"""Voice command resolution + per-segment TTS cache.

The TTS step (Chatterbox on CPU) dominates job wall time. The copied
``generate_voice_en.py`` already skips a segment whose WAV exists
(``should_skip_existing``), but it cannot know whether the *text* or the voice
parameters changed since that WAV was generated. This module closes that gap:

- ``voice_signature``  -> stable hash of the resolved voice command + env, so a
  change of engine/voice/params invalidates every segment.
- ``prune_stale_audio`` -> called by the materializers *instead of* deleting the
  whole ``audio/`` directory. It removes only the WAV artifacts of segments whose
  text-or-params hash changed (or that disappeared), and records the new hashes
  in ``audio/en/cache.json``. Surviving WAVs are then reused by the generator,
  so a repair attempt that only rewrites two scenes only re-synthesizes two
  segments.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shlex
from pathlib import Path

from video_api.config import Settings

logger = logging.getLogger(__name__)

CACHE_FILE_NAME = "cache.json"


def voice_command_for_settings(settings: Settings) -> tuple[list[str], dict[str, str] | None]:
    engine = settings.voice_engine.strip().lower()
    if engine in {"chatterbox", "local", "command"}:
        return shlex.split(settings.voice_command), None
    if engine == "kokoro":
        # Kokoro is ~5x real-time on CPU (vs Chatterbox being GPU-oriented) and
        # supports EN + FR. Deps live in the worker image; no env needed.
        return (
            [
                "python",
                "generate_voice_en.py",
                "--engine",
                "kokoro",
                "--kokoro-lang",
                settings.voice_language,
                "--kokoro-voice",
                settings.kokoro_voice,
                "--tail-padding",
                f"{settings.voice_tail_padding:.3f}",
            ],
            None,
        )
    if engine in {"moss", "moss-tts", "moss_tts"}:
        return (
            [
                "python",
                "generate_voice_en.py",
                "--engine",
                "moss",
                "--moss-model",
                settings.moss_tts_model,
                "--moss-language",
                settings.voice_language,
                "--moss-dtype",
                settings.moss_tts_dtype,
                "--tail-padding",
                f"{settings.voice_tail_padding:.3f}",
            ],
            {
                "VIDEO_API_MOSS_TTS_MODEL": settings.moss_tts_model,
                "VIDEO_API_MOSS_TTS_LANGUAGE": settings.voice_language,
                "VIDEO_API_MOSS_TTS_VOICE": settings.moss_tts_voice,
                "VIDEO_API_MOSS_TTS_REFERENCE_AUDIO": settings.moss_tts_reference_audio,
                "VIDEO_API_MOSS_TTS_REFERENCE_TEXT": settings.moss_tts_reference_text,
                "VIDEO_API_MOSS_TTS_CONSISTENT_VOICE": "1" if settings.moss_tts_consistent_voice else "0",
                "VIDEO_API_MOSS_TTS_DEVICE": settings.moss_tts_device,
                "VIDEO_API_MOSS_TTS_DTYPE": settings.moss_tts_dtype,
                "VIDEO_API_MOSS_TTS_COMMAND": settings.moss_tts_command,
            },
        )
    if engine in {"moss-remote", "moss_remote", "remote-moss"}:
        # Remote GPU TTS server (apps/tts-server): the model is loaded there on
        # demand and freed from VRAM after its idle grace period; the worker
        # only uploads texts and downloads WAVs.
        if not settings.tts_server_url.strip():
            raise ValueError(
                "VIDEO_API_TTS_SERVER_URL is required when VIDEO_API_VOICE_ENGINE=moss-remote"
            )
        return (
            [
                "python",
                "generate_voice_en.py",
                "--engine",
                "moss-remote",
                "--moss-model",
                settings.moss_tts_model,
                "--moss-language",
                settings.voice_language,
                "--tail-padding",
                f"{settings.voice_tail_padding:.3f}",
            ],
            {
                "VIDEO_API_TTS_SERVER_URL": settings.tts_server_url,
                "VIDEO_API_TTS_SERVER_API_KEY": settings.tts_server_api_key,
                "VIDEO_API_TTS_SERVER_TIMEOUT": str(settings.tts_server_timeout_seconds),
                "VIDEO_API_MOSS_TTS_MODEL": settings.moss_tts_model,
                "VIDEO_API_MOSS_TTS_REFERENCE_AUDIO": settings.moss_tts_reference_audio,
                "VIDEO_API_MOSS_TTS_CONSISTENT_VOICE": "1" if settings.moss_tts_consistent_voice else "0",
            },
        )
    if engine in {"openai", "openai-compatible", "openai_compatible"}:
        return (
            [
                "python",
                "generate_voice_en.py",
                "--engine",
                "openai",
                "--tail-padding",
                f"{settings.voice_tail_padding:.3f}",
            ],
            {
                "OPENAI_BASE_URL": settings.openai_base_url or "",
                "OPENAI_API_KEY": settings.openai_api_key or "",
                "VIDEO_API_OPENAI_TTS_MODEL": settings.openai_tts_model,
                "VIDEO_API_OPENAI_TTS_VOICE": settings.openai_tts_voice,
                "VIDEO_API_OPENAI_TTS_FORMAT": settings.openai_tts_format,
                "VIDEO_API_OPENAI_TTS_SPEED": str(settings.openai_tts_speed),
            },
        )
    raise ValueError(
        "Unsupported VIDEO_API_VOICE_ENGINE="
        f"{settings.voice_engine!r}; expected 'chatterbox', 'kokoro', 'moss', "
        "'moss-remote' or 'openai'."
    )


# Excluded from the audio fingerprint: these change how the synthesis endpoint
# is reached, never what it sounds like. Rotating an API key or moving the TTS
# server to another host must not invalidate every cached segment.
_SIGNATURE_ENV_EXCLUDE = ("API_KEY", "VIDEO_API_TTS_SERVER_URL", "VIDEO_API_TTS_SERVER_TIMEOUT")


def voice_signature(settings: Settings) -> str:
    """Stable fingerprint of everything that shapes the synthesized audio."""
    args, env = voice_command_for_settings(settings)
    fingerprint_env = {
        key: value
        for key, value in (env or {}).items()
        if not any(marker in key for marker in _SIGNATURE_ENV_EXCLUDE)
    }
    payload = json.dumps({"args": args, "env": fingerprint_env}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def segment_fingerprint(text: str, signature: str) -> str:
    payload = signature + "\x00" + " ".join(text.split())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _segment_files(audio_dir: Path, key: str) -> list[Path]:
    return [
        audio_dir / f"{key}.wav",
        audio_dir / f"{key}.wav.part",
        # Remove legacy MP3s left by jobs created before the PCM-only pipeline.
        audio_dir / f"{key}.mp3",
        audio_dir / f"{key}.padded.wav",
        *audio_dir.glob(f"{key}.openai.*"),
    ]


def prune_stale_audio(video_dir: Path, segments: list[dict], signature: str) -> dict[str, int]:
    """Invalidate cached segment audio whose text or voice params changed.

    *segments* is the ``segments_en.json`` shape: ``[{"key", "text", ...}]``.
    Returns counters (for logs/tests): {"reused": n, "invalidated": n}.
    """
    audio_dir = video_dir / "audio" / "en"
    cache_path = audio_dir / CACHE_FILE_NAME
    old_cache: dict[str, str] = {}
    if cache_path.exists():
        try:
            old_cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            old_cache = {}

    new_cache: dict[str, str] = {}
    reused = 0
    invalidated = 0
    current_keys = set()
    for segment in segments:
        key = segment["key"]
        current_keys.add(key)
        fingerprint = segment_fingerprint(segment["text"], signature)
        new_cache[key] = fingerprint
        wav_path = audio_dir / f"{key}.wav"
        if old_cache.get(key) == fingerprint and wav_path.exists():
            reused += 1
            continue
        if wav_path.exists():
            invalidated += 1
        for path in _segment_files(audio_dir, key):
            path.unlink(missing_ok=True)

    # Segments that no longer exist (scene removed/renamed by a repair attempt).
    for key in set(old_cache) - current_keys:
        for path in _segment_files(audio_dir, key):
            path.unlink(missing_ok=True)

    audio_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(new_cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info(
        "voice.cache video_dir=%s reused=%d invalidated=%d total=%d",
        video_dir,
        reused,
        invalidated,
        len(segments),
    )
    return {"reused": reused, "invalidated": invalidated}
