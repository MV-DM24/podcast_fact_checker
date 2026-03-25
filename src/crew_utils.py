from __future__ import annotations

import json
import logging
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool

from src.search_utils import search_the_web_for_claim

LOGGER = logging.getLogger(__name__)


@tool
def SearchWebTool(claim_text: str) -> str:
    """Search the web for evidence about a claim.

    Args:
        claim_text: The claim text to search for.

    Returns:
        A formatted string containing relevant search snippets and URLs.
    """
    try:
        return search_the_web_for_claim(claim_text=claim_text)
    except Exception:
        LOGGER.exception("SearchWebTool failed for claim: %r", claim_text)
        return "Search Results: Search unavailable due to an error."


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract and parse the first JSON object found in a text blob."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Model output was empty; expected JSON.")

    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        raise ValueError("Could not locate a JSON object in the model output.")

    candidate = text[start_idx : end_idx + 1]
    return json.loads(candidate)


def verify_claim_with_crew(claim_text: str) -> dict[str, str]:
    """Verify a claim using a CrewAI research+judge pipeline.

    The researcher agent uses `SearchWebTool` to gather evidence.
    The judge agent evaluates the evidence and returns a strict JSON object
    with exactly three keys: `verdict`, `explanation`, `source`.

    Args:
        claim_text: The claim text to fact-check.

    Returns:
        A dictionary with keys: `verdict`, `explanation`, `source`.
    """
    # Steps:
    # - Initialize Gemini LLM.
    # - Create researcher + judge agents with the same LLM.
    # - Create tasks for evidence gathering and final JSON verdict.
    # - Run the crew sequentially.
    # - Parse the judge output JSON robustly.
    # - Fallback safely if parsing fails.
    researcher = Agent(
        role="Senior Fact-Checking Researcher",
        goal="Use the SearchWebTool to find concrete evidence about a given claim.",
        backstory=(
            "You are meticulous and evidence-driven. Use the provided tool to gather "
            "relevant snippets and URLs, then present the most directly relevant findings."
        ),
        tools=[SearchWebTool],
        llm="groq/llama-3.3-70b-versatile",
        allow_delegation=False,
    )

    judge = Agent(
        role="Impartial Fact-Checking Judge",
        goal="Look at the researcher's findings and output a final verdict.",
        backstory=(
            "You are neutral and strict. You must base your verdict only on the "
            "provided research findings, without using outside knowledge."
        ),
        llm="groq/llama-3.3-70b-versatile",
        allow_delegation=False,
    )

    research_task = Task(
        description=(
            "Using SearchWebTool, find concrete evidence for the following claim.\n\n"
            "Claim:\n{claim_text}\n\n"
            "Requirements:\n"
            "1) Use SearchWebTool and include the most relevant snippets and URLs.\n"
            "2) Summarize how each piece of evidence relates to the claim.\n"
            "3) Prefer direct, high-signal sources."
        ),
        expected_output="A concise set of evidence snippets and URLs relevant to the claim.",
        agent=researcher,
    )

    judge_task = Task(
        description=(
            "You will be given the researcher's findings for a claim.\n\n"
            "Claim:\n{claim_text}\n\n"
            "Researcher's findings:\n(Use the task context provided by the researcher.)\n\n"
            "Now output a JSON object with exactly three keys: verdict, explanation, source.\n"
            "Rules:\n"
            '1) verdict must be exactly one of: "True", "False", or "Unverified".\n'
            '2) explanation must be a brief neutral justification referencing the evidence.\n'
            "3) source must be the best matching URL from the research findings.\n"
            "4) Output JSON ONLY (no markdown, no extra keys, no surrounding text)."
        ),
        expected_output=(
            "A JSON object with exactly three keys: "
            'verdict ("True"|"False"|"Unverified"), explanation, source.'
        ),
        agent=judge,
        context=[research_task],
    )

    crew = Crew(
        agents=[researcher, judge],
        tasks=[research_task, judge_task],
        process=Process.sequential,
        verbose=True,
    )

    try:
        result = crew.kickoff(inputs={"claim_text": claim_text})
    except Exception:
        LOGGER.exception("CrewAI verification failed for claim: %r", claim_text)
        return {
            "verdict": "Unverified",
            "explanation": "CrewAI run failed; unable to verify the claim.",
            "source": "",
        }

    raw_output: Any = result
    try:
        # CrewAI's return shape can vary by version; normalize to a string.
        if hasattr(result, "raw"):
            raw_output = getattr(result, "raw")
        elif hasattr(result, "output"):
            raw_output = getattr(result, "output")
    except Exception:
        pass

    output_text = raw_output if isinstance(raw_output, str) else str(raw_output)

    try:
        parsed = _extract_json_object(output_text)
    except Exception:
        LOGGER.exception("Failed to parse judge JSON for claim: %r", claim_text)
        return {
            "verdict": "Unverified",
            "explanation": "Model output was not valid JSON; unable to verify the claim.",
            "source": "",
        }

    allowed_verdicts = {"True", "False", "Unverified"}
    verdict = parsed.get("verdict")
    if verdict not in allowed_verdicts:
        verdict = "Unverified"

    explanation = parsed.get("explanation")
    source = parsed.get("source")

    return {
        "verdict": str(verdict),
        "explanation": str(explanation or ""),
        "source": str(source or ""),
    }

