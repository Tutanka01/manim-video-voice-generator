"""Environment-driven settings for the GPU TTS server."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DEFAULT_MODEL_REVISION = "cdd3b911b1585e3f2dbc7775ef10f9926f58850a"
DEFAULT_CODEC_MODEL = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
DEFAULT_CODEC_REVISION = "3cd226ba2947efa357ef453bcad111b6eafba782"


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # Comma-separated API keys for Authorization: Bearer / X-API-Key. Empty
    # (default) disables authentication: acceptable only on a trusted LAN/VPN;
    # the server logs a loud warning at startup.
    api_keys: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            key.strip() for key in os.getenv("TTS_SERVER_API_KEYS", "").split(",") if key.strip()
        )
    )
    model_id: str = field(
        default_factory=lambda: os.getenv("TTS_SERVER_MODEL", "OpenMOSS-Team/MOSS-TTS-v1.5")
    )
    model_revision: str = field(
        default_factory=lambda: os.getenv(
            "TTS_SERVER_MODEL_REVISION",
            DEFAULT_MODEL_REVISION,
        )
    )
    codec_model_id: str = field(
        default_factory=lambda: os.getenv("TTS_SERVER_CODEC_MODEL", DEFAULT_CODEC_MODEL)
    )
    codec_revision: str = field(
        default_factory=lambda: os.getenv(
            "TTS_SERVER_CODEC_REVISION",
            DEFAULT_CODEC_REVISION,
        )
    )
    # OCI digest injected by the deployment (for example
    # ghcr.io/org/image@sha256:...). If absent, the engine creates a boot-scoped
    # identity so persistent cache entries cannot cross an unknown image.
    image_digest: str = field(
        default_factory=lambda: os.getenv("TTS_SERVER_IMAGE_DIGEST", "").strip()
    )
    device: str = field(default_factory=lambda: os.getenv("TTS_SERVER_DEVICE", "auto"))
    dtype: str = field(default_factory=lambda: os.getenv("TTS_SERVER_DTYPE", "auto"))
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("TTS_SERVER_DATA_DIR", "/data"))
    )
    max_new_tokens: int = field(
        default_factory=lambda: int(os.getenv("TTS_SERVER_MAX_NEW_TOKENS", "4096"))
    )
    # Number of same-reference segments generated in one batched forward pass.
    # 1 keeps the strictly-sequential behaviour. On a bandwidth-bound GPU (e.g.
    # DGX Spark / GB10) autoregressive decode is memory-bound, so batching reads
    # the weights once for several segments and can be several times faster.
    # Raise carefully and watch VRAM; validate audio before trusting it in prod.
    batch_size: int = field(default_factory=lambda: max(1, int(os.getenv("TTS_SERVER_BATCH_SIZE", "1"))))
    # Per-segment guard rails: a segment is one narration paragraph, not a book.
    max_text_chars: int = field(
        default_factory=lambda: int(os.getenv("TTS_SERVER_MAX_TEXT_CHARS", "2000"))
    )
    max_segments: int = field(default_factory=lambda: int(os.getenv("TTS_SERVER_MAX_SEGMENTS", "64")))
    # Decoded size limit of the uploaded voice reference WAV.
    max_reference_bytes: int = field(
        default_factory=lambda: int(os.getenv("TTS_SERVER_MAX_REFERENCE_MB", "32")) * 1024 * 1024
    )
    # Terminal jobs (audio included) are deleted after this delay; the video
    # worker downloads the WAVs right after completion, so 48h is generous.
    job_ttl_hours: float = field(
        default_factory=lambda: float(os.getenv("TTS_SERVER_JOB_TTL_HOURS", "48"))
    )
    # Content-addressed WAV cache retention. 0 disables pruning.
    cache_ttl_days: float = field(
        default_factory=lambda: float(os.getenv("TTS_SERVER_CACHE_TTL_DAYS", "30"))
    )
    # Lazy-load grace: the ~8B checkpoint is unloaded from VRAM after this many
    # seconds without any synthesis, then loaded again on the next miss. The
    # synthesis profile and CAS keys survive an unload. 0 keeps the model
    # permanently warm (previous behavior).
    model_idle_seconds: float = field(
        default_factory=lambda: float(os.getenv("TTS_SERVER_MODEL_IDLE_SECONDS", "900"))
    )
    # 1 = deterministic fake engine (silent WAVs, no torch import). Used by the
    # test suite and to exercise the API on a machine without GPU.
    fake_engine: bool = field(default_factory=lambda: _bool_env("TTS_SERVER_FAKE_ENGINE", False))
    log_level: str = field(default_factory=lambda: os.getenv("TTS_SERVER_LOG_LEVEL", "INFO"))

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"


@lru_cache
def get_settings() -> Settings:
    return Settings()
