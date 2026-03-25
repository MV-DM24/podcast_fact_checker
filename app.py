from __future__ import annotations

import streamlit as st

from main import process_video

st.set_page_config(page_title="Podcast Fact-Checker", layout="centered")

st.markdown(
    """
<div style="text-align: center; padding: 0.5rem 0 1.0rem 0;">
  <h1 style="margin-bottom: 0.25rem;">Podcast Fact-Checker</h1>
  <p style="color: rgba(255,255,255,0.75); margin-top: 0;">
    Paste a YouTube link and get a fast, evidence-based fact-check of the most verifiable claims.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

with st.expander("How it works", expanded=False):
    st.markdown(
        """
- **Download audio** from the video
- **Transcribe** the conversation
- **Extract claims** that are verifiable
- **Fact-check** each claim with web evidence
- **Cache results** so repeat links are instant
"""
    )

youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")


def _count_verdicts(results: list[dict]) -> dict[str, int]:
    """Count verdict categories for a list of fact-check results.

    Args:
        results: List of result dictionaries, each containing a `verdict` key.

    Returns:
        Dictionary with keys: total, true, false, unverified.
    """
    verdict_counts = {"total": 0, "true": 0, "false": 0, "unverified": 0}
    for item in results:
        if not isinstance(item, dict):
            continue
        verdict_counts["total"] += 1
        verdict = str(item.get("verdict", "Unverified")).strip()
        if verdict == "True":
            verdict_counts["true"] += 1
        elif verdict == "False":
            verdict_counts["false"] += 1
        else:
            verdict_counts["unverified"] += 1
    return verdict_counts


def _render_fact_check_item(result: dict, index: int) -> None:
    """Render a single fact-check result in the Streamlit UI.

    Args:
        result: A single fact-check result dictionary.
        index: 1-based position of the claim, used for display.
    """
    claim_text = str(result.get("claim", "")).strip()
    verdict_str = str(result.get("verdict", "Unverified")).strip()
    explanation = str(result.get("explanation", "")).strip()
    source_url = str(result.get("source", "")).strip()

    if verdict_str == "True":
        header = f"Claim {index} — ✅ True"
    elif verdict_str == "False":
        header = f"Claim {index} — ❌ False"
    else:
        header = f"Claim {index} — ⚠️ Unverified"

    with st.expander(header, expanded=index == 1):
        if claim_text:
            st.markdown(f"**Claim:** {claim_text}")

        if verdict_str == "True":
            st.success("Verdict: True")
        elif verdict_str == "False":
            st.error("Verdict: False")
        else:
            st.warning("Verdict: Unverified")

        if explanation:
            st.markdown("**Explanation**")
            st.write(explanation)

        if source_url:
            st.markdown(f"**Source:** [{source_url}]({source_url})")
        else:
            st.caption("Source: (not provided)")


if st.button("Analyze & Fact-Check", type="primary"):
    if not youtube_url or not youtube_url.strip():
        st.warning("Please paste a valid YouTube URL.")
        st.stop()

    with st.spinner(
        "Processing video... this can take a couple of minutes due to AI rate limits."
    ):
        try:
            results = process_video(youtube_url.strip())
        except Exception as exc:
            st.error("Sorry — the pipeline failed while processing this video.")
            st.caption(f"Error details: {exc}")
            st.stop()

    if not results:
        st.info("No claims were extracted or no fact-check results were produced.")
    else:
        counts = _count_verdicts(results)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total claims", str(counts["total"]))
        col2.metric("True", str(counts["true"]))
        col3.metric("False", str(counts["false"]))
        col4.metric("Unverified", str(counts["unverified"]))

        st.divider()

        for idx, result in enumerate(results, start=1):
            if isinstance(result, dict):
                _render_fact_check_item(result, index=idx)

