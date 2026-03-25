from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from groq import Groq

LOGGER = logging.getLogger(__name__)


def transcribe_audio_with_groq(audio_file_path: str) -> str:
    """Transcribe an audio file using Groq Whisper.

    This function calls Groq's `whisper-large-v3` model through the `groq` Python SDK.

    Args:
        audio_file_path: Absolute or relative path to an audio file (`.mp3`, `.m4a`, etc.).

    Returns:
        The transcription text.

    Raises:
        RuntimeError: If the API key is missing, the file can't be read, or the API call fails.
    """
    # Steps:
    # - Validate GROQ_API_KEY is present
    # - Validate audio file exists
    # - Call Groq audio transcription endpoint with whisper-large-v3
    # - Return response.text
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY environment variable. Add it to your .env file.")

    audio_path = Path(audio_file_path).expanduser().resolve()
    if not audio_path.exists() or not audio_path.is_file():
        raise RuntimeError(f"Audio file does not exist: '{audio_path}'")

    client = Groq(api_key=api_key)

    try:
        LOGGER.info("Transcribing with Groq Whisper (whisper-large-v3)...")
        with audio_path.open("rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
            )
    except Exception as exc:
        LOGGER.exception("Groq transcription failed for file: %s", audio_path)
        raise RuntimeError(f"Groq transcription failed for file: '{audio_path}'") from exc

    transcript_text = getattr(transcription, "text", None)
    if not isinstance(transcript_text, str) or not transcript_text.strip():
        raise RuntimeError("Groq returned an empty transcription.")

    return transcript_text.strip()


def extract_claims_from_transcript(transcript_text: str) -> list[dict]:
    """Extract verifiable factual claims from a transcript using Groq Llama 3.

    The model is instructed to return claims as structured JSON, and this function
    parses that output into Python objects.

    Args:
        transcript_text: Full transcript text produced by the transcription step.

    Returns:
        A list of dictionaries with schema: [{"claim": "<verifiable claim>"}].

    Raises:
        RuntimeError: If API key is missing, API call fails, or JSON output is invalid.
    """
    # Steps:
    # - Validate input transcript and API key
    # - Build a focused prompt for extracting only verifiable factual claims
    # - Call Groq chat completions with JSON mode enabled
    # - Parse response JSON robustly and normalize to list[dict]
    if not transcript_text or not transcript_text.strip():
        return []

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY environment variable. Add it to your .env file.")

    client = Groq(api_key=api_key)

    system_prompt = (
        "You are an expert fact-checking researcher. "
        "Read the provided podcast transcript and extract only verifiable factual claims. "
        "Ignore opinions, subjective judgments, speculation, rhetorical questions, and personal preferences. "
        "Return concise claims that can be validated with reliable sources."
    )
    user_prompt = (
        "Extract factual claims from the transcript below.\n\n"
        "Output requirements:\n"
        "1) Return valid JSON only.\n"
        '2) Use a top-level object with key "claims".\n'
        '3) "claims" must be a JSON array of objects.\n'
        '4) Each object must contain exactly one key: "claim".\n'
        "5) Do not include explanations or extra keys.\n\n"
        f"Transcript:\n{transcript_text}"
    )

    try:
        LOGGER.info("Extracting claims with Groq (llama3-70b-8192)...")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
    except Exception as exc:
        LOGGER.exception("Groq claim extraction failed.")
        raise RuntimeError("Groq claim extraction failed.") from exc

    try:
        response_content = response.choices[0].message.content
    except Exception as exc:
        LOGGER.exception("Groq claim extraction returned unexpected response shape.")
        raise RuntimeError("Groq claim extraction returned unexpected response shape.") from exc

    if not isinstance(response_content, str) or not response_content.strip():
        raise RuntimeError("Groq claim extraction returned empty content.")

    try:
        parsed_json = json.loads(response_content)
    except json.JSONDecodeError as exc:
        LOGGER.exception("Failed to parse Groq JSON output.")
        raise RuntimeError(f"Groq returned invalid JSON for claim extraction: {exc}") from exc

    claims_value = parsed_json.get("claims") if isinstance(parsed_json, dict) else None
    if not isinstance(claims_value, list):
        LOGGER.error("Groq JSON output missing 'claims' list: %s", parsed_json)
        raise RuntimeError("Groq JSON output must include a top-level 'claims' list.")

    normalized_claims: list[dict] = []
    for claim_item in claims_value:
        if isinstance(claim_item, dict):
            claim_text = claim_item.get("claim")
            if isinstance(claim_text, str) and claim_text.strip():
                normalized_claims.append({"claim": claim_text.strip()})

    return normalized_claims


def get_fact_check_verdict(claim_text: str, search_results: str) -> dict:
    """Get a neutral fact-check verdict for a claim using search evidence.

    The LLM is instructed to evaluate the claim using only the provided search
    results. The model must respond with a JSON object containing:
    `verdict` (one of "True", "False", "Unverified"),
    `explanation` (string), and
    `source` (string; the best URL from the search results).

    Args:
        claim_text: The claim to be evaluated.
        search_results: Pre-formatted web search snippets for evidence.

    Returns:
        A dictionary parsed from the model's JSON response.

    Raises:
        RuntimeError: If the Groq API call fails or the response is not valid JSON.
    """
    # Steps:
    # - Validate inputs and load GROQ_API_KEY
    # - Build a strict, neutral fact-check prompt
    # - Call Groq with JSON object response format enabled
    # - Parse and validate the returned JSON structure
    if not isinstance(claim_text, str) or not claim_text.strip():
        raise RuntimeError("claim_text must be a non-empty string.")

    if not isinstance(search_results, str):
        search_results = str(search_results)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY environment variable. Add it to your .env file.")

    client = Groq(api_key=api_key)

    system_prompt = (
        "You are a neutral fact-checking assistant. "
        "Your job is to assess whether a claim is supported by evidence. "
        "You must use ONLY the provided search results as evidence and must not use "
        "any outside knowledge."
    )
    user_prompt = (
        "Fact-checking task\n\n"
        f"Claim:\n{claim_text}\n\n"
        f"Search Results (use only these):\n{search_results}\n\n"
        "Rules:\n"
        "1) Base your decision exclusively on the provided search results.\n"
        '2) If the evidence clearly supports the claim, verdict must be "True".\n'
        '3) If the evidence clearly contradicts the claim, verdict must be "False".\n'
        '4) If the evidence is insufficient, unclear, or does not directly address the claim, '
        'verdict must be "Unverified".\n'
        "5) Choose the best URL from the provided search results as `source`.\n"
        "6) Respond with JSON ONLY (no markdown, no extra keys).\n"
        "JSON schema:\n"
        "{\n"
        '  "verdict": "True" | "False" | "Unverified",\n'
        '  "explanation": "string",\n'
        '  "source": "best URL from search results"\n'
        "}\n"
    )

    try:
        LOGGER.info("Requesting fact-check verdict from Groq (llama-3-70b-8192)...")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
    except Exception as exc:
        LOGGER.exception("Groq fact-check verdict failed.")
        raise RuntimeError("Groq fact-check verdict failed.") from exc

    try:
        response_content = response.choices[0].message.content
    except Exception as exc:
        LOGGER.exception("Groq fact-check verdict returned unexpected response shape.")
        raise RuntimeError("Groq fact-check verdict returned unexpected response shape.") from exc

    if not isinstance(response_content, str) or not response_content.strip():
        raise RuntimeError("Groq fact-check verdict returned empty content.")

    try:
        parsed_json = json.loads(response_content)
    except json.JSONDecodeError as exc:
        LOGGER.exception("Failed to parse Groq JSON output for fact-check verdict.")
        raise RuntimeError(f"Groq returned invalid JSON for fact-check verdict: {exc}") from exc

    if not isinstance(parsed_json, dict):
        raise RuntimeError("Groq fact-check verdict JSON output must be an object/dict.")

    allowed_verdicts = {"True", "False", "Unverified"}
    verdict_value = parsed_json.get("verdict")
    if verdict_value not in allowed_verdicts:
        # Keep it resilient: coerce to a default rather than failing the whole run.
        LOGGER.warning("Unexpected verdict value from model: %s", verdict_value)
        parsed_json["verdict"] = "Unverified"

    # Ensure required keys exist.
    for required_key in ("verdict", "explanation", "source"):
        if required_key not in parsed_json or not isinstance(parsed_json[required_key], str):
            parsed_json[required_key] = str(parsed_json.get(required_key, "") or "")

    return parsed_json
