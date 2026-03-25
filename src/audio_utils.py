from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from yt_dlp import YoutubeDL

LOGGER = logging.getLogger(__name__)


def download_youtube_audio(youtube_url: str) -> str:
    """Download the best available audio from a YouTube URL.

    The audio is downloaded into the project's `data/` folder and saved as an
    `.m4a` or `.mp3` file when possible. If the best audio is downloaded as a
    different container (e.g., `.webm`) and `ffmpeg` is available, it will be
    converted to `.mp3`.

    Args:
        youtube_url: YouTube video URL to download audio from.

    Returns:
        Absolute path to the downloaded (or converted) audio file.

    Raises:
        RuntimeError: If the download fails or the final audio file cannot be located.
    """
    # Steps:
    # - Resolve project root and ensure data/ exists
    # - Use yt-dlp to download bestaudio to data/%(id)s.%(ext)s
    # - Determine the downloaded filepath from yt-dlp info dict
    # - If needed and ffmpeg exists, convert to mp3
    # - Return absolute path
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    try:
        data_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        LOGGER.exception("Failed to create data directory at %s", data_dir)
        raise RuntimeError(f"Failed to create data directory at '{data_dir}'.") from exc

    output_template = str(data_dir / "%(id)s.%(ext)s")

    def _progress_hook(progress: dict) -> None:
        status = progress.get("status")
        if status == "downloading":
            percent = progress.get("_percent_str", "").strip()
            speed = progress.get("_speed_str", "").strip()
            eta = progress.get("_eta_str", "").strip()
            LOGGER.info("Downloading audio... %s %s ETA %s", percent, speed, eta)
        elif status == "finished":
            filename = progress.get("filename")
            LOGGER.info("Download finished: %s", filename)

    ydl_opts: dict = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [_progress_hook],
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
    except Exception as exc:
        LOGGER.exception("yt-dlp failed for URL: %s", youtube_url)
        raise RuntimeError(f"Audio download failed for URL: {youtube_url}") from exc

    downloaded_path = _extract_downloaded_filepath(info_dict)
    if downloaded_path is None or not downloaded_path.exists():
        LOGGER.error("Could not locate downloaded audio file (info keys: %s)", list(info_dict.keys()))
        raise RuntimeError("Audio download completed but the output file could not be located.")

    if downloaded_path.suffix.lower() in {".mp3", ".m4a"}:
        return str(downloaded_path.resolve())

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        LOGGER.warning(
            "Downloaded audio is %s and ffmpeg is not available; keeping original file.",
            downloaded_path.suffix,
        )
        return str(downloaded_path.resolve())

    converted_path = downloaded_path.with_suffix(".mp3")
    try:
        _convert_to_mp3(ffmpeg_path=ffmpeg_path, input_path=downloaded_path, output_path=converted_path)
    except Exception as exc:
        LOGGER.exception("ffmpeg conversion failed; keeping original file: %s", downloaded_path)
        return str(downloaded_path.resolve())

    return str(converted_path.resolve())


def _extract_downloaded_filepath(info_dict: dict) -> Path | None:
    """Extract the output filepath from a yt-dlp info dict.

    Args:
        info_dict: The dictionary returned by `yt-dlp` after extraction/download.

    Returns:
        The `Path` to the downloaded file if found, else None.
    """
    # yt-dlp can return a "requested_downloads" list that includes "filepath".
    requested_downloads = info_dict.get("requested_downloads")
    if isinstance(requested_downloads, list) and requested_downloads:
        filepath = requested_downloads[0].get("filepath")
        if isinstance(filepath, str) and filepath:
            return Path(filepath)

    # Fallbacks used by some versions/configurations.
    filepath = info_dict.get("filepath") or info_dict.get("_filename")
    if isinstance(filepath, str) and filepath:
        return Path(filepath)

    return None


def _convert_to_mp3(ffmpeg_path: str, input_path: Path, output_path: Path) -> None:
    """Convert an audio file to mp3 using ffmpeg.

    Args:
        ffmpeg_path: Path to the ffmpeg executable.
        input_path: Input audio file path.
        output_path: Output mp3 file path.

    Raises:
        RuntimeError: If ffmpeg returns non-zero exit code or output isn't created.
    """
    LOGGER.info("Converting to mp3 with ffmpeg... (%s -> %s)", input_path.name, output_path.name)
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "libmp3lame",
        "-q:a",
        "2",
        str(output_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise RuntimeError("Failed to execute ffmpeg for audio conversion.") from exc

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit={result.returncode}): {result.stderr.strip()}")

    if not output_path.exists():
        raise RuntimeError("ffmpeg reported success but output mp3 was not created.")
