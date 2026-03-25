from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from supabase import create_client

LOGGER = logging.getLogger(__name__)


def _get_supabase_client():
    """Create and return a Supabase client using environment variables.

    Returns:
        A Supabase client instance created via `supabase.create_client`.

    Raises:
        RuntimeError: If `SUPABASE_URL` or `SUPABASE_KEY` is missing.
    """
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY environment variables.")

    return create_client(supabase_url, supabase_key)


SUPABASE = _get_supabase_client()


def upload_audio_to_storage(local_file_path: str, youtube_video_id: str) -> str:
    """Upload a local `.m4a`/`.mp3` audio file to Supabase Storage.

    Files are uploaded to the `audio-files` bucket under the object key:
    `youtube/{youtube_video_id}/{youtube_video_id}{ext}`.

    Args:
        local_file_path: Path to a local audio file (`.m4a` or `.mp3`).
        youtube_video_id: The YouTube video ID used to namespace the object key.

    Returns:
        A public URL for the uploaded object.

    Raises:
        RuntimeError: If the file cannot be read or the upload fails.
        ValueError: If the inputs are invalid or the extension is not supported.
    """
    # Steps:
    # - Validate inputs and file extension
    # - Read the file bytes
    # - Upload into Supabase Storage bucket `audio-files`
    # - Return a public URL for the stored object
    if not isinstance(local_file_path, str) or not local_file_path.strip():
        raise ValueError("local_file_path must be a non-empty string.")
    if not isinstance(youtube_video_id, str) or not youtube_video_id.strip():
        raise ValueError("youtube_video_id must be a non-empty string.")

    audio_path = Path(local_file_path).expanduser().resolve()
    if not audio_path.exists() or not audio_path.is_file():
        raise RuntimeError(f"Audio file does not exist: '{audio_path}'")

    extension = audio_path.suffix.lower()
    if extension not in {".m4a", ".mp3"}:
        raise ValueError("Only .m4a and .mp3 files are supported for upload.")

    object_key = f"youtube/{youtube_video_id.strip()}/{youtube_video_id.strip()}{extension}"
    content_type = "audio/mp4" if extension == ".m4a" else "audio/mpeg"

    try:
        file_bytes = audio_path.read_bytes()
    except OSError as exc:
        LOGGER.exception("Failed to read audio file for upload: %s", audio_path)
        raise RuntimeError("Failed to read local audio file.") from exc

    try:
        SUPABASE.storage.from_("audio-files").upload(
            path=object_key,
            file=file_bytes,
            file_options={"content-type": content_type, "upsert": "true"},
        )
    except Exception as exc:
        LOGGER.exception("Supabase Storage upload failed (bucket=audio-files, key=%s).", object_key)
        raise RuntimeError("Supabase Storage upload failed.") from exc

    try:
        public_url: str = SUPABASE.storage.from_("audio-files").get_public_url(object_key)
    except Exception as exc:
        LOGGER.exception("Failed to generate public URL for key=%s.", object_key)
        raise RuntimeError("Failed to generate public URL for uploaded audio.") from exc

    if not isinstance(public_url, str) or not public_url.strip():
        raise RuntimeError("Supabase returned an empty public URL for the uploaded audio.")

    return public_url.strip()


def check_if_video_processed(youtube_url: str) -> Any | None:
    """Check whether a YouTube URL has already been processed.

    This queries the `fact_checks` table for an existing row matching the URL.

    Args:
        youtube_url: The YouTube video URL.

    Returns:
        The JSON `results` stored in the table if found, otherwise None.

    Raises:
        RuntimeError: If the database query fails.
        ValueError: If `youtube_url` is invalid.
    """
    # Steps:
    # - Validate input URL
    # - Query `fact_checks` for a row with matching url
    # - Return `results` JSON if present
    if not isinstance(youtube_url, str) or not youtube_url.strip():
        raise ValueError("youtube_url must be a non-empty string.")

    normalized_url = youtube_url.strip()

    try:
        response = (
            SUPABASE.table("fact_checks")
            .select("results")
            .eq("youtube_url", normalized_url)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        LOGGER.exception("Supabase query failed for url=%s", normalized_url)
        raise RuntimeError("Database query failed.") from exc

    data = getattr(response, "data", None)
    if not isinstance(data, list) or not data:
        return None

    first_row = data[0] if isinstance(data[0], dict) else None
    if not isinstance(first_row, dict):
        return None

    return first_row.get("results")


def save_results_to_db(youtube_url: str, final_fact_checked_claims: list[dict]) -> None:
    """Persist fact-check results for a YouTube URL into Supabase.

    This upserts a row into the `fact_checks` table, using the URL as the key.

    Args:
        youtube_url: The YouTube video URL used as the lookup key.
        final_fact_checked_claims: The list of claim dicts returned by the pipeline.

    Raises:
        RuntimeError: If the insert/upsert fails.
        ValueError: If inputs are invalid.
    """
    # Steps:
    # - Validate inputs
    # - Upsert row into `fact_checks` with url + results JSON
    if not isinstance(youtube_url, str) or not youtube_url.strip():
        raise ValueError("youtube_url must be a non-empty string.")
    if not isinstance(final_fact_checked_claims, list):
        raise ValueError("final_fact_checked_claims must be a list of dictionaries.")

    normalized_url = youtube_url.strip()
    row = {"youtube_url": normalized_url, "results": final_fact_checked_claims}

    try:
        SUPABASE.table("fact_checks").upsert(row, on_conflict="youtube_url").execute()
    except Exception as exc:
        LOGGER.exception("Supabase upsert failed for url=%s", normalized_url)
        raise RuntimeError("Failed to save results to the database.") from exc

