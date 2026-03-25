from __future__ import annotations

import json
import time

from dotenv import load_dotenv

from main import _configure_logging
from src.audio_utils import download_youtube_audio
from src.crew_utils import verify_claim_with_crew
from src.llm_utils import extract_claims_from_transcript, transcribe_audio_with_groq


if __name__ == "__main__":
    _configure_logging()
    load_dotenv()

    # Hardcoded short YouTube URL for Milestone 1 smoke test.
    test_youtube_url = "https://youtu.be/6VDQHi04IX0?si=-VxxeYX2tFwRSzbe"

    print("Downloading audio...")
    audio_path = download_youtube_audio(youtube_url=test_youtube_url)
    print(f"Audio saved at: {audio_path}")

    print("Transcribing with Groq...")
    transcript = transcribe_audio_with_groq(audio_file_path=audio_path)

    print("Extracting factual claims...")
    extracted_claims = extract_claims_from_transcript(transcript_text=transcript)

    print("\n===== EXTRACTED CLAIMS =====\n")
    print(json.dumps(extracted_claims, indent=2, ensure_ascii=False))

    # Milestone 3: fact-check each extracted claim using web search + LLM verdict.
    fact_checked_claims: list[dict] = []
    for claim in extracted_claims[:3]:
        claim_text = claim.get("claim")
        if not isinstance(claim_text, str) or not claim_text.strip():
            continue

        print(f"Fact-checking: {claim_text!r}")

        verdict_dict = verify_claim_with_crew(claim_text=claim_text)
        combined_claim = {**claim, **verdict_dict}
        fact_checked_claims.append(combined_claim)
        time.sleep(20)

    print("\n===== FACT-CHECKED CLAIMS =====\n")
    print(json.dumps(fact_checked_claims, indent=2, ensure_ascii=False))

