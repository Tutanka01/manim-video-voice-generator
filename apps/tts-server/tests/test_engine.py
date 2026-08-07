from __future__ import annotations

import sys
import threading
import time
import types

import pytest

import tts_server.engine as engine_module
from tts_server.cache import AudioCache
from tts_server.config import (
    DEFAULT_CODEC_REVISION,
    DEFAULT_MODEL_REVISION,
    Settings,
)
from tts_server.engine import (
    MOSS_GENERATION_PARAMETERS,
    EngineNotReady,
    FakeEngine,
    MossEngine,
    estimate_new_tokens,
)


def test_short_text_gets_the_floor() -> None:
    assert estimate_new_tokens("Hi.", ceiling=4096) == 256


def test_estimate_scales_with_length_and_stays_below_ceiling() -> None:
    short = estimate_new_tokens("word " * 20, ceiling=4096)
    long = estimate_new_tokens("word " * 200, ceiling=4096)
    assert 256 < short < long <= 4096


def test_estimate_is_capped_by_the_ceiling() -> None:
    # A very long segment can never exceed the configured hard cap.
    assert estimate_new_tokens("x" * 100_000, ceiling=1024) == 1024


def test_short_text_never_exceeds_a_low_ceiling() -> None:
    # The floor must not override an explicitly lower hard cap.
    assert estimate_new_tokens("Hi.", ceiling=128) == 128


def test_default_model_and_codec_revisions_are_immutable_commits() -> None:
    assert DEFAULT_MODEL_REVISION == "cdd3b911b1585e3f2dbc7775ef10f9926f58850a"
    assert DEFAULT_CODEC_REVISION == "3cd226ba2947efa357ef453bcad111b6eafba782"
    assert len(DEFAULT_MODEL_REVISION) == 40
    assert len(DEFAULT_CODEC_REVISION) == 40
    int(DEFAULT_MODEL_REVISION, 16)
    int(DEFAULT_CODEC_REVISION, 16)


def test_fake_engine_builds_an_immutable_complete_profile(tmp_path) -> None:
    settings = Settings(
        fake_engine=True,
        data_dir=tmp_path,
        image_digest=f"registry/tts@sha256:{'1' * 64}",
    )
    engine = FakeEngine(settings)
    engine._load_safely()

    profile = engine.synthesis_profile()
    assert profile is not None
    assert profile["model"]["revision"] == DEFAULT_MODEL_REVISION
    assert profile["model"]["remote_code_revision"] == DEFAULT_MODEL_REVISION
    assert profile["model"]["codec_revision"] == DEFAULT_CODEC_REVISION
    assert profile["image_digest"] == f"sha256:{'1' * 64}"
    assert profile["engine"] == "fake"
    assert profile["dtype"] == "pcm16"
    assert profile["attention"] == "none"
    assert profile["audio_format"] == {
        "container": "wav",
        "codec": "pcm_s16le",
        "sample_rate": 24000,
        "channels": 1,
    }

    profile["model"]["revision"] = "mutated"
    assert engine.synthesis_profile()["model"]["revision"] == DEFAULT_MODEL_REVISION


def test_missing_or_mutable_image_identity_is_isolated_per_boot(tmp_path) -> None:
    missing_a = FakeEngine(Settings(fake_engine=True, data_dir=tmp_path / "a"))
    missing_b = FakeEngine(Settings(fake_engine=True, data_dir=tmp_path / "b"))
    mutable = FakeEngine(
        Settings(
            fake_engine=True,
            data_dir=tmp_path / "c",
            image_digest="registry/tts:latest",
        )
    )
    for engine in (missing_a, missing_b, mutable):
        engine._load_safely()

    identities = {
        engine.synthesis_profile()["image_digest"]
        for engine in (missing_a, missing_b, mutable)
    }
    assert len(identities) == 3
    assert all(identity.startswith("boot-sha256:") for identity in identities)


def test_fake_and_moss_profiles_cannot_share_an_engine_profile(tmp_path) -> None:
    settings = Settings(
        fake_engine=True,
        data_dir=tmp_path,
        image_digest=f"sha256:{'2' * 64}",
    )
    fake = FakeEngine(settings)
    fake._load_safely()
    moss = MossEngine(settings)
    moss._device = "cuda"
    moss._dtype_name = "torch.bfloat16"
    moss._attention = "sdpa"
    moss._sample_rate = 24000

    fake_profile = fake.synthesis_profile()
    moss_profile = moss._build_synthesis_profile()
    assert AudioCache.engine_profile_id(fake_profile) != AudioCache.engine_profile_id(
        moss_profile
    )


@pytest.mark.parametrize("revision", ["", "main", "v1.5", "a" * 39, "g" * 40])
def test_mutable_or_invalid_model_revisions_are_rejected(
    tmp_path,
    revision: str,
) -> None:
    settings = Settings(
        data_dir=tmp_path,
        model_revision=revision,
        image_digest=f"sha256:{'3' * 64}",
    )

    with pytest.raises(ValueError, match="explicit 40-character commit SHA"):
        MossEngine(settings)._load()


@pytest.mark.parametrize("revision", ["", "main", "v1.5", "a" * 39, "g" * 40])
def test_mutable_or_invalid_codec_revisions_are_rejected(
    tmp_path,
    revision: str,
) -> None:
    settings = Settings(
        data_dir=tmp_path,
        codec_revision=revision,
        image_digest=f"sha256:{'3' * 64}",
    )

    with pytest.raises(ValueError, match="explicit 40-character commit SHA"):
        MossEngine(settings)._load()


def test_moss_load_pins_weights_remote_code_and_codec(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    class _AudioTokenizer:
        def to(self, device):
            calls["tokenizer_device"] = device
            return self

    processor = types.SimpleNamespace(
        audio_tokenizer=_AudioTokenizer(),
        model_config=types.SimpleNamespace(sampling_rate=24000),
    )

    class _Model:
        def to(self, device):
            calls["model_device"] = device
            return self

        def eval(self):
            calls["model_eval"] = True

    class _AutoProcessor:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["processor"] = (model_id, kwargs)
            return processor

    class _AutoModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["model"] = (model_id, kwargs)
            return _Model()

    huggingface_hub = types.ModuleType("huggingface_hub")

    def snapshot_download(**kwargs):
        calls["codec"] = kwargs
        return "/immutable/codec-snapshot"

    huggingface_hub.snapshot_download = snapshot_download
    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = _AutoProcessor
    transformers.AutoModel = _AutoModel
    torch = types.ModuleType("torch")

    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(engine_module, "_select_torch_device", lambda requested: "cpu")
    monkeypatch.setattr(
        engine_module,
        "_select_moss_dtype",
        lambda requested, device: "torch.bfloat16",
    )
    monkeypatch.setattr(
        engine_module,
        "_resolve_attn_implementation",
        lambda device, dtype: "eager",
    )

    settings = Settings(
        data_dir=tmp_path,
        image_digest=f"sha256:{'4' * 64}",
    )
    engine = MossEngine(settings)
    engine._load()

    assert calls["codec"] == {
        "repo_id": settings.codec_model_id,
        "revision": DEFAULT_CODEC_REVISION,
    }
    processor_model, processor_kwargs = calls["processor"]
    assert processor_model == settings.model_id
    assert processor_kwargs == {
        "revision": DEFAULT_MODEL_REVISION,
        "code_revision": DEFAULT_MODEL_REVISION,
        "codec_path": "/immutable/codec-snapshot",
        "trust_remote_code": True,
    }
    model_id, model_kwargs = calls["model"]
    assert model_id == settings.model_id
    assert model_kwargs["revision"] == DEFAULT_MODEL_REVISION
    assert model_kwargs["code_revision"] == DEFAULT_MODEL_REVISION
    assert model_kwargs["trust_remote_code"] is True
    details = engine._profile_details()
    assert details["generation"]["sampling"] == MOSS_GENERATION_PARAMETERS
    assert details["generation"]["batching"] == {
        "configured_batch_size": 1,
        "policy": "ordered-same-reference-v1",
        "token_budget": "maximum-estimate-in-batch-v1",
    }
    assert details["audio_format"] == {
        "container": "wav",
        "codec": "pcm_s16le",
        "sample_rate": 24000,
        "channels": 1,
    }


# -- lazy lifecycle ----------------------------------------------------------


def test_lazy_engine_starts_idle_and_loads_on_demand(tmp_path) -> None:
    settings = Settings(
        fake_engine=True,
        data_dir=tmp_path,
        image_digest=f"sha256:{'5' * 64}",
    )
    engine = FakeEngine(settings)
    engine._probe_safely()

    assert engine.state == "idle"
    assert engine.model_loaded is False
    assert engine.synthesis_profile() is not None
    assert engine.info()["idle_timeout_seconds"] == 900

    engine.ensure_ready()
    assert engine.state == "ready"
    assert engine.model_loaded is True


def test_lazy_engine_never_loads_when_idle_timeout_is_zero(tmp_path) -> None:
    settings = Settings(
        fake_engine=True,
        data_dir=tmp_path,
        model_idle_seconds=0,
        image_digest=f"sha256:{'6' * 64}",
    )
    engine = FakeEngine(settings)
    engine.start()
    time.sleep(0.05)
    engine.ensure_ready()
    assert engine.state == "ready"
    # A zero timeout keeps the model warm; unload is still possible manually.
    engine._last_activity = time.monotonic() - 10
    engine.unload()
    assert engine.state == "idle"


def test_unload_is_respected_only_after_grace_and_keeps_profile(tmp_path) -> None:
    settings = Settings(
        fake_engine=True,
        data_dir=tmp_path,
        image_digest=f"sha256:{'7' * 64}",
    )
    engine = FakeEngine(settings)
    engine._probe_safely()
    engine.ensure_ready()

    engine.unload()
    assert engine.state == "ready"  # grace period still open

    engine._last_activity = time.monotonic() - engine._idle_seconds - 5
    engine.unload()
    assert engine.state == "idle"
    assert engine.model_loaded is False
    assert engine.synthesis_profile() is not None

    engine.ensure_ready()
    assert engine.state == "ready"


def test_watchdog_unloads_after_idle_timeout(tmp_path) -> None:
    settings = Settings(
        fake_engine=True,
        data_dir=tmp_path / "d",
        model_idle_seconds=0.2,
        image_digest=f"sha256:{'8' * 64}",
    )
    engine = FakeEngine(settings)
    engine._probe_safely()
    engine.ensure_ready()
    assert engine.state == "ready"

    watchdog = threading.Thread(target=engine._watchdog_loop, daemon=True)
    watchdog.start()
    time.sleep(1.3)
    try:
        assert engine.state == "idle"
    finally:
        engine._stop.set()
        watchdog.join(timeout=2)


def test_failed_load_retries_on_demand(tmp_path, monkeypatch) -> None:
    settings = Settings(
        fake_engine=True,
        data_dir=tmp_path,
        image_digest=f"sha256:{'9' * 64}",
    )
    engine = FakeEngine(settings)
    attempts = {"n": 0}
    original_load = engine._load

    def flaky_load() -> None:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("transient failure")
        return original_load()

    monkeypatch.setattr(engine, "_load", flaky_load)
    engine._probe_safely()

    with pytest.raises(EngineNotReady, match="failed to load"):
        engine.ensure_ready(timeout=2)
    assert engine.state == "error"

    # Skip the retry cooldown, then the next request recovers.
    engine._last_load_attempt = 0
    engine.ensure_ready(timeout=2)
    assert engine.state == "ready"
    assert attempts["n"] == 2
