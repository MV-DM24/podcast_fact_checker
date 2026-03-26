from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

from src.audio_utils import download_youtube_audio
from src.crew_utils import verify_claim_with_crew
from src.llm_utils import extract_claims_from_transcript, transcribe_audio_with_groq
from src.db_utils import check_if_video_processed, save_results_to_db, upload_audio_to_storage

LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure basic console logging for local runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def process_video(youtube_url: str, max_claims: int = 5) -> list[dict]:
    """Download, transcribe, extract claims, and fact-check them with AI.

    This runs the full Milestone 3 pipeline:
    1) Download audio from a YouTube URL.
    2) Transcribe audio to text using Groq.
    3) Extract verifiable factual claims from the transcript.
    4) Fact-check up to `max_claims` claims using CrewAI and web evidence.

    Args:
        youtube_url: A YouTube video URL.
        max_claims: Maximum number of extracted claims to fact-check.

    Returns:
        A list of dictionaries; each element contains the original `claim` plus
        `verdict`, `explanation`, and `source`.
    """
    # Steps (high-level):
    # - Check DB cache; return immediately if available
    # - Load environment variables (API keys)
    # - Download audio and transcribe it
    # - Upload audio to cloud storage (best-effort backup)
    # - Extract claims and slice to `max_claims`
    # - Verify each claim (sleep between calls to avoid rate limits)
    # - Save results to DB cache
    if not isinstance(youtube_url, str) or not youtube_url.strip():
        raise ValueError("youtube_url must be a non-empty string.")
    if not isinstance(max_claims, int) or max_claims < 1:
        raise ValueError("max_claims must be an integer >= 1.")

    try:
        load_dotenv()
    except Exception:
        LOGGER.exception("Failed to load .env; continuing with environment variables.")

    try:
        cached_results = check_if_video_processed(youtube_url=youtube_url.strip())
    except Exception as exc:
        LOGGER.exception("Failed to query Supabase cache for URL: %s", youtube_url)
        raise RuntimeError("Failed to check cache in the database.") from exc

    if cached_results is not None:
        if isinstance(cached_results, list):
            return cached_results
        # Be resilient to unexpected DB shapes, but don't silently return junk.
        LOGGER.warning("Cached results for URL=%s were not a list; ignoring cache.", youtube_url)

    try:
        audio_path = download_youtube_audio(youtube_url=youtube_url)
    except Exception as exc:
        LOGGER.exception("Audio download failed for URL: %s", youtube_url)
        raise RuntimeError("Audio download failed.") from exc

    try:
        youtube_video_id = Path(audio_path).stem
        upload_audio_to_storage(local_file_path=audio_path, youtube_video_id=youtube_video_id)
    except Exception:
        LOGGER.exception("Audio cloud backup failed; continuing without storage backup.")

    try:
        transcript = transcribe_audio_with_groq(audio_file_path=audio_path)
    except Exception as exc:
        LOGGER.exception("Transcription failed for audio: %s", audio_path)
        raise RuntimeError("Transcription failed.") from exc

    try:
        extracted_claims = extract_claims_from_transcript(transcript_text=transcript)
    except Exception as exc:
        LOGGER.exception("Claim extraction failed for transcript.")
        raise RuntimeError("Claim extraction failed.") from exc

    sliced_claims = extracted_claims[:max_claims]
    fact_checked_claims: list[dict] = []

    for claim in sliced_claims:
        claim_text = claim.get("claim")
        if not isinstance(claim_text, str) or not claim_text.strip():
            continue

        try:
            verdict_dict = verify_claim_with_crew(claim_text=claim_text)
        except Exception as exc:
            LOGGER.exception("Verification failed for claim: %r", claim_text)
            verdict_dict = {
                "verdict": "Unverified",
                "explanation": "Verification failed due to an internal error.",
                "source": "",
            }

        combined_claim = {**claim, **verdict_dict}
        fact_checked_claims.append(combined_claim)

        # Rate limit protection for repeated LLM calls.
        time.sleep(20)

    try:
        save_results_to_db(youtube_url=youtube_url.strip(), final_fact_checked_claims=fact_checked_claims)
    except Exception:
        LOGGER.exception("Failed to save results to Supabase cache; continuing without caching.")

    return fact_checked_claims
