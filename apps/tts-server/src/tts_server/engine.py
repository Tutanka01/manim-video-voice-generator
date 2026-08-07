"""MOSS-TTS engine: the ~8B checkpoint is loaded lazily on first use and
unloaded again after an idle grace period, so the GPU server does not keep
16-18 GB of VRAM allocated around the clock. GPU access is serialized by a lock.

The synthesis logic mirrors the reference implementation in
``videos/linux-fondamentaux/002-c-est-quoi-un-syscall/generate_voice_en.py``
(device/dtype/attention selection, language names, reference-audio cloning) so
the remote voice sounds exactly like the local ``moss`` engine. The synthesis
profile (device/dtype/attention/sample-rate) is resolved without loading the
weights, so the content-addressed audio cache keeps working while the model is
cold; ``TTS_SERVER_MODEL_IDLE_SECONDS`` controls the idle grace before unload
(0 keeps the model permanently warm).
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import importlib.util
import logging
import platform
import re
import threading
import time
import uuid
import wave
from pathlib import Path

from tts_server import __version__
from tts_server.config import Settings

logger = logging.getLogger(__name__)

MOSS_LANGUAGE_NAMES = {
    "zh": "Chinese",
    "yue": "Cantonese",
    "en": "English",
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "nl": "Dutch",
    "es": "Spanish",
    "fr": "French",
    "fi": "Finnish",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mk": "Macedonian",
    "ms": "Malay",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sw": "Swahili",
    "sv": "Swedish",
    "tl": "Tagalog",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
}


def moss_language_name(language: str) -> str:
    language = (language or "").strip()
    code = language.lower().replace("_", "-").split("-", 1)[0]
    if code not in MOSS_LANGUAGE_NAMES:
        supported = ", ".join(sorted(MOSS_LANGUAGE_NAMES))
        raise ValueError(f"MOSS-TTS does not support language {language!r}. Supported codes: {supported}")
    return MOSS_LANGUAGE_NAMES[code]


# MOSS-TTS-v1.5 emits audio codes at ~12.5 tokens per second of speech.
AUDIO_TOKENS_PER_SECOND = 12.5
# Deliberately LOW chars/second and a headroom factor so the token budget always
# over-covers a legitimate segment (slow speech, and CJK packs more audio per
# character). The goal is only to bound *runaway* generations that never emit an
# end token — not to truncate real narration. A 300-char segment needs ~250
# tokens; this grants ~1100, while a global cap of 4096 would let a runaway burn
# ~5 minutes of audio.
_MIN_CHARS_PER_SECOND = 5.0
_TOKEN_HEADROOM = 1.5
_MIN_NEW_TOKENS = 256
MOSS_GENERATION_PARAMETERS = {
    "text_temperature": 1.5,
    "text_top_p": 1.0,
    "text_top_k": 50,
    "audio_temperature": 1.7,
    "audio_top_p": 0.8,
    "audio_top_k": 25,
    "audio_repetition_penalty": 1.0,
}
_COMMIT_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_OCI_DIGEST_RE = re.compile(r"(?:^|@)(sha256:[0-9a-f]{64})$")


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _pinned_commit(value: str, variable: str) -> str:
    revision = value.strip().lower()
    if not _COMMIT_REVISION_RE.fullmatch(revision):
        raise ValueError(f"{variable} must be an explicit 40-character commit SHA")
    return revision


def _image_identity(configured: str) -> str:
    match = _OCI_DIGEST_RE.search(configured.strip().lower())
    if match:
        return match.group(1)
    nonce = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
    if configured:
        logger.warning(
            "TTS_SERVER_IMAGE_DIGEST is not an immutable sha256 digest; "
            "persistent CAS reuse is isolated to this process boot"
        )
    else:
        logger.warning(
            "TTS_SERVER_IMAGE_DIGEST is unset; persistent CAS reuse is "
            "isolated to this process boot"
        )
    return f"boot-sha256:{nonce}"


def estimate_new_tokens(text: str, ceiling: int) -> int:
    """Per-segment ``max_new_tokens`` derived from the text length.

    Bounded by ``ceiling`` (the configured hard cap) above and ``_MIN_NEW_TOKENS``
    below so short segments still get a usable budget.
    """
    seconds = max(1.0, len(text) / _MIN_CHARS_PER_SECOND)
    estimate = int(seconds * AUDIO_TOKENS_PER_SECOND * _TOKEN_HEADROOM)
    # The hard ceiling always wins, even over the floor (a ceiling below the
    # floor is unusual but must still be honoured).
    return min(ceiling, max(_MIN_NEW_TOKENS, estimate))


class EngineNotReady(RuntimeError):
    pass


class BaseEngine:
    name = "base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model_revision = settings.model_revision.strip().lower()
        self._codec_revision = settings.codec_revision.strip().lower()
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._load_error: str | None = None
        self._synthesis_profile: dict | None = None
        self._image_digest = _image_identity(settings.image_digest)
        # Lazy-load / idle-unload bookkeeping.
        self._load_in_progress = threading.Event()
        self._start_lock = threading.Lock()
        self._load_thread: threading.Thread | None = None
        self._last_load_attempt = 0.0
        self._last_activity = time.monotonic()
        self._stop = threading.Event()
        self._idle_seconds = max(0.0, settings.model_idle_seconds)

    # -- lifecycle -----------------------------------------------------------
    # After a failed load a caller may retry on demand, at most once per
    # cooldown, so a transient error does not brick the server until restart.
    _LOAD_RETRY_COOLDOWN = 60.0

    def start(self) -> None:
        """Boot the engine into a cold-but-usable state.

        The synthesis profile (device/dtype/attention/sample-rate) is resolved
        without loading the checkpoint, so the content-addressed audio cache
        already works while VRAM stays free. Weights are loaded by the first
        synthesis and unloaded again after ``model_idle_seconds`` without use.
        """
        threading.Thread(target=self._probe_safely, name="engine-probe", daemon=True).start()
        if self._idle_seconds > 0:
            watchdog = threading.Thread(
                target=self._watchdog_loop, name="engine-idle", daemon=True
            )
            watchdog.start()
        logger.info(
            "engine.boot engine=%s model=%s lazy=on idle_timeout=%.0fs",
            self.name,
            self.settings.model_id,
            self._idle_seconds,
        )

    def _probe_safely(self) -> None:
        """Resolve the synthesis profile in the background; never loads weights."""
        try:
            self._initialize()
            self._synthesis_profile = self._build_synthesis_profile()
        except Exception as error:  # noqa: BLE001 - surfaced via state/ensure_ready
            self._load_error = f"{type(error).__name__}: {error}"
            logger.exception("engine.probe.failed model=%s", self.settings.model_id)
        else:
            logger.info(
                "engine.idle engine=%s model=%s (profile ready, weights lazy)",
                self.name,
                self.settings.model_id,
            )

    def _initialize(self) -> None:  # pragma: no cover - overridden
        pass

    def _load_safely(self) -> None:
        self._load_in_progress.set()
        try:
            self._load()
            self._synthesis_profile = self._build_synthesis_profile()
        except Exception as error:  # noqa: BLE001 - surfaced via state/ensure_ready
            self._load_error = f"{type(error).__name__}: {error}"
            logger.exception("engine.load.failed model=%s", self.settings.model_id)
        else:
            self._ready.set()
            logger.info("engine.ready engine=%s model=%s", self.name, self.settings.model_id)
        finally:
            self._load_in_progress.clear()

    def _load(self) -> None:  # pragma: no cover - overridden
        pass

    def _profile_details(self) -> dict:
        return {
            "engine": self.name,
            "device": "unknown",
            "dtype": "unknown",
            "attention": "unknown",
            "generation": {},
            "audio_format": {},
        }

    def _build_synthesis_profile(self) -> dict:
        profile = {
            "generator": {
                "name": "promptloom-moss-tts-server",
                "version": __version__,
                "profile_version": 2,
            },
            "model": {
                "repo": self.settings.model_id,
                "revision": self._model_revision,
                "remote_code_revision": self._model_revision,
                "codec_repo": self.settings.codec_model_id,
                "codec_revision": self._codec_revision,
            },
            "image_digest": self._image_digest,
            "runtime": {
                "python": platform.python_version(),
                "torch": _package_version("torch"),
                "transformers": _package_version("transformers"),
                "soundfile": _package_version("soundfile"),
                "flash_attn": _package_version("flash-attn"),
                "torchaudio": _package_version("torchaudio"),
                "torchcodec": _package_version("torchcodec"),
                "huggingface_hub": _package_version("huggingface-hub"),
                "numpy": _package_version("numpy"),
            },
        }
        profile.update(self._profile_details())
        return profile

    def synthesis_profile(self) -> dict | None:
        if self._synthesis_profile is None:
            return None
        return copy.deepcopy(self._synthesis_profile)

    @property
    def state(self) -> str:
        if self._load_error:
            return "error"
        if self._ready.is_set():
            return "ready"
        if self._load_in_progress.is_set():
            return "loading"
        return "idle"

    @property
    def model_loaded(self) -> bool:
        return self._ready.is_set()

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def _ensure_load_thread(self) -> None:
        """Start (or reuse) the single background load thread when not ready.

        A failed load is retried on demand, at most once per cooldown, so a
        transient error does not leave the server bricked until restart.
        """
        if self._ready.is_set():
            return
        with self._start_lock:
            if self._ready.is_set():
                return
            if self._load_error:
                if time.monotonic() - self._last_load_attempt < self._LOAD_RETRY_COOLDOWN:
                    return
                self._load_error = None
            thread = self._load_thread
            if thread is None or not thread.is_alive():
                self._last_load_attempt = time.monotonic()
                self._load_thread = threading.Thread(
                    target=self._load_safely, name="engine-load", daemon=True
                )
                self._load_thread.start()

    def ensure_ready(self, timeout: float | None = None) -> None:
        """Block until the model is loaded; load it lazily from an idle state
        and raise if loading failed/timed out."""
        self._ensure_load_thread()
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self._ready.wait(timeout=1.0):
            if self._load_error:
                raise EngineNotReady(f"model failed to load: {self._load_error}")
            if deadline is not None and time.monotonic() > deadline:
                raise EngineNotReady("timed out waiting for the model to load")
        if self._load_error:
            raise EngineNotReady(f"model failed to load: {self._load_error}")
        self._note_activity()

    def _note_activity(self) -> None:
        """Re-arm the idle grace period, called under the engine lock at the
        end of every synthesis so an in-flight run is never unloaded."""
        self._last_activity = time.monotonic()

    def unload(self) -> None:
        """Release the model's GPU memory after an idle period.

        A running synthesis holds the engine lock, so this never tears the
        model down mid-generation, and the synthesis profile (hence the CAS
        keys) survives the unload.
        """
        with self._lock:
            if not self._ready.is_set():
                return
            if self._idle_seconds > 0 and (
                time.monotonic() - self._last_activity < self._idle_seconds
            ):
                return
            self._ready.clear()
            self._unload()
        logger.info("engine.unloaded engine=%s state=%s", self.name, self.state)

    def _unload(self) -> None:  # pragma: no cover - overridden
        pass

    def _watchdog_loop(self) -> None:
        tick = max(1.0, min(30.0, self._idle_seconds / 2.0))
        while not self._stop.wait(timeout=tick):
            try:
                if not self._ready.is_set() or self._load_in_progress.is_set():
                    continue
                if time.monotonic() - self._last_activity >= self._idle_seconds:
                    idle = time.monotonic() - self._last_activity
                    logger.info(
                        "engine.unload.idle engine=%s idle=%.0fs timeout=%.0fs",
                        self.name,
                        idle,
                        self._idle_seconds,
                    )
                    self.unload()
            except Exception:  # noqa: BLE001 - the watchdog must never die
                logger.exception("engine.unload.error")

    # -- synthesis -----------------------------------------------------------
    def synthesize(self, text: str, language: str, reference: Path | None, out_path: Path) -> None:
        self.ensure_ready()
        with self._lock:
            self._synthesize(text, language, reference, out_path)
            self._note_activity()

    def synthesize_batch(
        self,
        texts: list[str],
        language: str,
        reference: Path | None,
        out_paths: list[Path],
    ) -> None:
        """Generate several segments that share one reference in a single pass.

        All items in a batch use the *same* cloning reference (the job's anchor),
        which is the only grouping the pipeline ever needs. The base
        implementation renders them one by one; engines with true batched
        generation override :meth:`_synthesize_batch`.
        """
        if len(texts) != len(out_paths):
            raise ValueError("texts and out_paths must have the same length")
        self.ensure_ready()
        with self._lock:
            self._synthesize_batch(texts, language, reference, out_paths)
            self._note_activity()

    def _synthesize(self, text: str, language: str, reference: Path | None, out_path: Path) -> None:
        raise NotImplementedError

    def _synthesize_batch(
        self,
        texts: list[str],
        language: str,
        reference: Path | None,
        out_paths: list[Path],
    ) -> None:
        for text, out_path in zip(texts, out_paths):
            self._synthesize(text, language, reference, out_path)

    def info(self) -> dict:
        return {
            "engine": self.name,
            "model": self.settings.model_id,
            "model_revision": self._model_revision,
            "codec_model": self.settings.codec_model_id,
            "codec_revision": self._codec_revision,
            "image_digest": self._image_digest,
            "state": self.state,
            "model_loaded": self.model_loaded,
            "idle_timeout_seconds": self._idle_seconds,
        }


def _select_torch_device(requested: str) -> str:
    import torch

    requested = (requested or "auto").strip().lower()
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _select_moss_dtype(requested: str, device: str):
    import torch

    requested = (requested or "auto").strip().lower()
    if requested in {"float32", "fp32"}:
        return torch.float32
    if requested in {"float16", "fp16"}:
        return torch.float16
    if requested in {"bfloat16", "bf16"}:
        return torch.bfloat16
    # The checkpoint is BF16; keeping it avoids materializing the 8B params as
    # fp32 (double the memory) on CPU and CUDA alike.
    return torch.bfloat16 if device in {"cpu", "cuda"} else torch.float32


def _resolve_attn_implementation(device: str, dtype) -> str:
    import torch

    if (
        device == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    if device == "cuda":
        return "sdpa"
    return "eager"


class MossEngine(BaseEngine):
    name = "moss"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._processor = None
        self._model = None
        self._device: str | None = None
        self._dtype = None
        self._dtype_name: str | None = None
        self._attention: str | None = None
        self._sample_rate: int | None = None
        self._init_lock = threading.Lock()

    def _initialize(self) -> None:
        """Resolve every immutable aspect of the engine without occupying VRAM.

        Pins the model/codec revisions, downloads the codec and processor,
        resolves device/dtype/attention and the sampling rate, and leaves the
        synthesis profile available so the CAS cache already works while the
        8B checkpoint stays cold. The audio tokenizer and model are only moved
        to the device when :meth:`_load` actually needs to generate.
        """
        with self._init_lock:
            if self._processor is not None and self._device is not None:
                return
            model_revision = _pinned_commit(
                self.settings.model_revision,
                "TTS_SERVER_MODEL_REVISION",
            )
            codec_revision = _pinned_commit(
                self.settings.codec_revision,
                "TTS_SERVER_CODEC_REVISION",
            )
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoProcessor

            device = _select_torch_device(self.settings.device)
            if device == "cuda":
                torch.backends.cuda.enable_cudnn_sdp(False)
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
            dtype = _select_moss_dtype(self.settings.dtype, device)
            attn_implementation = _resolve_attn_implementation(device, dtype)
            logger.info(
                "engine.probe model=%s device=%s dtype=%s attn=%s",
                self.settings.model_id,
                device,
                dtype,
                attn_implementation,
            )
            codec_path = snapshot_download(
                repo_id=self.settings.codec_model_id,
                revision=codec_revision,
            )
            # Build a pinned, checkpoint-free snapshot of the model repo so the
            # processor loads its remote code/config/tokenizer straight from
            # disk. Recent transformers forwards `revision`/`code_revision`
            # kwargs into the MOSS processor's own `from_pretrained`, which then
            # passes them to ProcessorMixin.__init__ ("Unexpected keyword
            # argument revision"). Passing a local snapshot path keeps everything
            # pinned to the exact commit while the ~16 GB checkpoint shards stay
            # lazy (downloaded/loaded on the first synthesis only).
            model_snapshot = snapshot_download(
                repo_id=self.settings.model_id,
                revision=model_revision,
                ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.pth", "*.onnx"],
            )
            processor = AutoProcessor.from_pretrained(
                model_snapshot,
                codec_path=codec_path,
                trust_remote_code=True,
            )
            # NOTE: processor.audio_tokenizer is intentionally NOT moved to the
            # device here; that only happens in _load so VRAM stays free while
            # the model is cold.
            self._processor = processor
            self._device = device
            self._dtype = dtype
            self._dtype_name = str(dtype)
            self._attention = attn_implementation
            self._sample_rate = int(processor.model_config.sampling_rate)

    def _load(self) -> None:
        if self._processor is None or self._device is None:
            self._initialize()
            self._synthesis_profile = self._build_synthesis_profile()
        from transformers import AutoModel

        processor = self._processor
        device = self._device
        dtype = self._dtype
        attn_implementation = self._attention
        logger.info(
            "engine.load.start model=%s device=%s dtype=%s attn=%s",
            self.settings.model_id,
            device,
            dtype,
            attn_implementation,
        )
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)
        model = AutoModel.from_pretrained(
            self.settings.model_id,
            revision=self._model_revision,
            code_revision=self._model_revision,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        self._model = model

    def _unload(self) -> None:
        import gc

        import torch

        self._processor = None
        self._model = None
        gc.collect()
        if self._device == "cuda":
            torch.cuda.empty_cache()

    def _profile_details(self) -> dict:
        return {
            "engine": self.name,
            "device": self._device,
            "dtype": self._dtype_name,
            "attention": self._attention,
            "generation": {
                "sampling": dict(MOSS_GENERATION_PARAMETERS),
                "batching": {
                    "configured_batch_size": max(1, self.settings.batch_size),
                    "policy": "ordered-same-reference-v1",
                    "token_budget": "maximum-estimate-in-batch-v1",
                },
                "max_new_tokens_policy": {
                    "ceiling": self.settings.max_new_tokens,
                    "audio_tokens_per_second": AUDIO_TOKENS_PER_SECOND,
                    "minimum_chars_per_second": _MIN_CHARS_PER_SECOND,
                    "headroom": _TOKEN_HEADROOM,
                    "floor": _MIN_NEW_TOKENS,
                },
            },
            "audio_format": {
                "container": "wav",
                "codec": "pcm_s16le",
                "sample_rate": self._sample_rate,
                "channels": 1,
            },
        }

    def _synthesize(self, text: str, language: str, reference: Path | None, out_path: Path) -> None:
        self._run([text], language, reference, [out_path])

    def _synthesize_batch(
        self,
        texts: list[str],
        language: str,
        reference: Path | None,
        out_paths: list[Path],
    ) -> None:
        self._run(texts, language, reference, out_paths)

    def _run(
        self,
        texts: list[str],
        language: str,
        reference: Path | None,
        out_paths: list[Path],
    ) -> None:
        import soundfile as sf
        import torch

        processor, model = self._processor, self._model
        device = next(model.parameters()).device
        language_name = moss_language_name(language)
        reference_list = [str(reference)] if reference else None
        conversations = [
            [processor.build_user_message(text=text, language=language_name, reference=reference_list)]
            for text in texts
        ]
        # A batch runs until its longest sequence stops, so the budget is the max
        # over its segments; short ones still stop early on their own end token.
        max_new = max(estimate_new_tokens(text, self.settings.max_new_tokens) for text in texts)
        with torch.no_grad():
            batch = processor(conversations, mode="generation")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new,
                **MOSS_GENERATION_PARAMETERS,
            )
            messages = processor.decode(outputs)
        if len(messages) != len(out_paths):
            raise RuntimeError(
                f"MOSS TTS returned {len(messages)} messages for {len(out_paths)} segments."
            )
        sampling_rate = processor.model_config.sampling_rate
        cap_seconds = max_new / AUDIO_TOKENS_PER_SECOND
        for message, text, out_path in zip(messages, texts, out_paths):
            if message is None or not message.audio_codes_list:
                raise RuntimeError("MOSS TTS returned no decoded audio.")
            samples = message.audio_codes_list[0].to(torch.float32).cpu().numpy()
            # Written via soundfile, not torchaudio.save: torchaudio >= 2.9
            # delegates saving to the optional `torchcodec` package, absent from
            # the image. PCM16 keeps the WAV readable by the stdlib `wave`
            # duration probe and halves the size vs float32.
            sf.write(str(out_path), samples, sampling_rate, subtype="PCM_16")
            if len(samples) / float(sampling_rate) >= 0.95 * cap_seconds:
                logger.warning(
                    "engine.generate.hit_token_cap chars=%d max_new=%d seconds=%.1f "
                    "(audio may be truncated or the model failed to stop)",
                    len(text),
                    max_new,
                    len(samples) / float(sampling_rate),
                )

    def info(self) -> dict:
        data = super().info()
        data.update(
            {
                "device": self._device,
                "dtype": self._dtype_name,
                "attention": self._attention,
            }
        )
        try:
            import torch

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                data["gpu"] = torch.cuda.get_device_name(0)
                data["vram_free_gb"] = round(free / 1024**3, 1)
                data["vram_total_gb"] = round(total / 1024**3, 1)
        except Exception:  # noqa: BLE001 - health info is best-effort
            pass
        return data


class FakeEngine(BaseEngine):
    """Deterministic stand-in for tests and GPU-less smoke runs.

    Writes silent PCM16 WAVs whose duration scales with the text length, and
    records every call (text, language, reference) for assertions.
    """

    name = "fake"
    sample_rate = 24000

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.calls: list[tuple[str, str, str]] = []

    def _load(self) -> None:
        return None

    def _profile_details(self) -> dict:
        return {
            "engine": self.name,
            "device": "fake",
            "dtype": "pcm16",
            "attention": "none",
            "generation": {
                "algorithm": "deterministic-silence-v1",
            },
            "audio_format": {
                "container": "wav",
                "codec": "pcm_s16le",
                "sample_rate": self.sample_rate,
                "channels": 1,
            },
        }

    def _synthesize(self, text: str, language: str, reference: Path | None, out_path: Path) -> None:
        moss_language_name(language)
        self.calls.append((text, language, str(reference) if reference else ""))
        seconds = max(0.3, len(text) / 15.0)
        frames = int(seconds * self.sample_rate)
        with wave.open(str(out_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.sample_rate)
            handle.writeframes(b"\x00\x00" * frames)


def create_engine(settings: Settings) -> BaseEngine:
    if settings.fake_engine:
        return FakeEngine(settings)
    return MossEngine(settings)
