from __future__ import annotations

import streamlit as st

from main import process_video

st.set_page_config(page_title="AI Podcast Fact-Checker", layout="centered")
st.title("AI Podcast Fact-Checker")
st.write("Paste a YouTube URL to analyze the audio, extract a few factual claims, and run AI fact-checking.")

youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")


def _render_fact_check_item(result: dict) -> None:
    """Render a single fact-check result in the Streamlit UI."""
    claim_text = str(result.get("claim", "")).strip()
    verdict_str = str(result.get("verdict", "Unverified")).strip()
    explanation = str(result.get("explanation", "")).strip()
    source_url = str(result.get("source", "")).strip()

    if claim_text:
        st.markdown(f"### Claim: {claim_text}")

    if verdict_str == "True":
        st.success("Verdict: True")
    elif verdict_str == "False":
        st.error("Verdict: False")
    else:
        st.warning("Verdict: Unverified")

    if explanation:
        st.write(explanation)

    if source_url:
        st.markdown(f"Source URL: [{source_url}]({source_url})")
    else:
        st.caption("Source URL: (not provided)")


if st.button("Analyze & Fact-Check", type="primary"):
    if not youtube_url or not youtube_url.strip():
        st.warning("Please paste a valid YouTube URL.")
        st.stop()

    with st.spinner(
        "Processing video... this will take a couple of minutes due to AI rate limits."
    ):
        try:
            results = process_video(youtube_url.strip())
        except Exception as exc:
            st.error(f"Failed to process video: {exc}")
            st.stop()

    if not results:
        st.info("No claims were extracted or no fact-check results were produced.")
    else:
        for result in results:
            _render_fact_check_item(result)
            st.divider()

