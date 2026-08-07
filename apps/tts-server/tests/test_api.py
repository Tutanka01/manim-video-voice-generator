from __future__ import annotations

import base64
import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tts_server.config import Settings
from tts_server.main import create_app

AUTH = {"Authorization": "Bearer test-key"}


@pytest.fixture()
def client(tmp_path: Path):
    settings = Settings(
        api_keys=("test-key",),
        fake_engine=True,
        data_dir=tmp_path / "data",
        model_id="OpenMOSS-Team/MOSS-TTS-v1.5",
        image_digest=f"sha256:{'1' * 64}",
    )
    app = create_app(settings)
    with TestClient(app) as test_client:
        _wait_for_ready(test_client)
        yield test_client


def _wait_for_ready(test_client: TestClient, deadline_seconds: float = 10.0) -> None:
    # With lazy loading the health check answers 200 as soon as the engine is
    # operational (idle or ready). Wait instead for the synthesis-profile probe
    # so healthz assertions and cache keys are deterministic.
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        if test_client.app.state.engine.synthesis_profile() is not None:
            return
        time.sleep(0.05)
    raise AssertionError("engine probe never completed")


def _wait_for_job(test_client: TestClient, job_id: str, deadline_seconds: float = 30.0) -> dict:
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        response = test_client.get(f"/v1/jobs/{job_id}", headers=AUTH)
        assert response.status_code == 200
        state = response.json()
        if state["status"] in {"completed", "failed"}:
            return state
        time.sleep(0.05)
    raise AssertionError("job never reached a terminal state")


def _batch_payload(**overrides) -> dict:
    payload = {
        "language": "en",
        "consistent_voice": True,
        "segments": [
            {"key": "Scene1_IntroEN", "text": "The kernel sits between hardware and programs."},
            {"key": "Scene2_SyscallEN", "text": "A system call crosses that boundary."},
        ],
    }
    payload.update(overrides)
    return payload


def test_healthz_reports_engine_without_auth(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == "fake"
    assert body["state"] == "idle"
    assert body["model_loaded"] is False
    assert body["idle_timeout_seconds"] == 900
    assert body["auth"] is True
    assert body["model_revision"] == "cdd3b911b1585e3f2dbc7775ef10f9926f58850a"
    assert len(body["engine_profile_id"]) == 64


def test_model_loads_on_demand_and_unloads_after_idle(client: TestClient) -> None:
    assert client.get("/healthz").json()["state"] == "idle"

    response = client.post("/v1/tts/batch", json=_batch_payload(), headers=AUTH)
    state = _wait_for_job(client, response.json()["job_id"])
    assert state["status"] == "completed"

    engine = client.app.state.engine
    assert engine.state == "ready"
    assert engine.model_loaded is True

    # A fresh unload() call is a no-op while the grace period is open.
    engine.unload()
    assert engine.state == "ready"

    # Pretend the grace period elapsed, then unload.
    engine._last_activity = time.monotonic() - engine._idle_seconds - 5
    engine.unload()
    health = client.get("/healthz").json()
    assert health["state"] == "idle"
    assert health["model_loaded"] is False
    assert health["engine_profile_id"]  # profile survives the unload
    assert client.get("/healthz").status_code == 200

    # The next cache miss lazily reloads and synthesizes again.
    engine.ensure_ready()
    assert engine.state == "ready"
    second = client.post("/v1/tts/batch", json=_batch_payload(), headers=AUTH)
    _wait_for_job(client, second.json()["job_id"])


def test_endpoints_require_api_key(client: TestClient) -> None:
    assert client.post("/v1/tts/batch", json=_batch_payload()).status_code == 401
    assert client.get("/v1/jobs/whatever").status_code == 401
    assert client.post("/v1/tts", json={"text": "hi"}).status_code == 401
    wrong = {"Authorization": "Bearer nope"}
    assert client.post("/v1/tts/batch", json=_batch_payload(), headers=wrong).status_code == 401


def test_batch_job_completes_and_serves_wav(client: TestClient) -> None:
    response = client.post("/v1/tts/batch", json=_batch_payload(), headers=AUTH)
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    state = _wait_for_job(client, job_id)
    assert state["status"] == "completed"
    assert [segment["status"] for segment in state["segments"]] == ["done", "done"]
    assert all(segment["duration_seconds"] > 0 for segment in state["segments"])
    assert all(len(segment["synthesis_profile_id"]) == 64 for segment in state["segments"])
    assert state["model_revision"] == "cdd3b911b1585e3f2dbc7775ef10f9926f58850a"

    wav = client.get(state["segments"][0]["wav_url"], headers=AUTH)
    assert wav.status_code == 200
    assert wav.headers["content-type"].startswith("audio/wav")
    assert wav.content[:4] == b"RIFF"


def test_batch_mp3_is_encoded_only_when_requested(client: TestClient) -> None:
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not available in this environment")
    response = client.post("/v1/tts/batch", json=_batch_payload(), headers=AUTH)
    job_id = response.json()["job_id"]
    state = _wait_for_job(client, job_id)
    first = state["segments"][0]
    mp3_path = client.app.state.jobs.job_dir(job_id) / "Scene1_IntroEN.mp3"

    assert "mp3_url" in first
    assert not mp3_path.exists()

    mp3 = client.get(first["mp3_url"], headers=AUTH)
    assert mp3.status_code == 200
    assert mp3.headers["content-type"].startswith("audio/mpeg")
    assert mp3_path.exists()

    reused = client.get(first["mp3_url"], headers=AUTH)
    assert reused.status_code == 200
    assert reused.content == mp3.content


def test_consistent_voice_anchors_following_segments(client: TestClient) -> None:
    response = client.post("/v1/tts/batch", json=_batch_payload(), headers=AUTH)
    job_id = response.json()["job_id"]
    _wait_for_job(client, job_id)

    calls = client.app.state.engine.calls
    assert calls[0][2] == ""  # first segment has no reference
    assert calls[1][2].endswith("Scene1_IntroEN.wav")  # second is anchored to it


def test_uploaded_reference_is_used_for_all_segments(client: TestClient) -> None:
    reference = base64.b64encode(b"RIFFfake-reference").decode("ascii")
    payload = _batch_payload(reference_audio_b64=reference)
    response = client.post("/v1/tts/batch", json=payload, headers=AUTH)
    job_id = response.json()["job_id"]
    _wait_for_job(client, job_id)

    calls = client.app.state.engine.calls
    assert all(call[2].endswith("reference.wav") for call in calls[-2:])


def test_second_identical_batch_hits_the_cache(client: TestClient) -> None:
    first = client.post("/v1/tts/batch", json=_batch_payload(), headers=AUTH)
    _wait_for_job(client, first.json()["job_id"])
    synth_calls = len(client.app.state.engine.calls)

    second = client.post("/v1/tts/batch", json=_batch_payload(), headers=AUTH)
    assert second.json()["status"] == "completed"
    state = _wait_for_job(client, second.json()["job_id"])

    assert state["status"] == "completed"
    assert all(segment["cached"] for segment in state["segments"])
    assert len(client.app.state.engine.calls) == synth_calls  # no new synthesis


def test_batching_preserves_anchor_and_completes(tmp_path: Path) -> None:
    settings = Settings(
        api_keys=("test-key",),
        fake_engine=True,
        data_dir=tmp_path / "data",
        model_id="OpenMOSS-Team/MOSS-TTS-v1.5",
        batch_size=4,
        image_digest=f"sha256:{'1' * 64}",
    )
    app = create_app(settings)
    with TestClient(app) as client:
        _wait_for_ready(client)
        payload = _batch_payload(
            segments=[
                {"key": "Scene1_IntroEN", "text": "The kernel sits between hardware and programs."},
                {"key": "Scene2_SyscallEN", "text": "A system call crosses that boundary."},
                {"key": "Scene3_ReturnEN", "text": "Then control returns to the program."},
            ]
        )
        response = client.post("/v1/tts/batch", json=payload, headers=AUTH)
        job_id = response.json()["job_id"]
        state = _wait_for_job(client, job_id)

        assert state["status"] == "completed"
        assert [segment["status"] for segment in state["segments"]] == ["done", "done", "done"]

        calls = client.app.state.engine.calls
        # First segment defines the anchor (no reference); the batched rest share it.
        assert calls[0][2] == ""
        assert calls[1][2].endswith("Scene1_IntroEN.wav")
        assert calls[2][2].endswith("Scene1_IntroEN.wav")


def test_sync_endpoint_returns_wav_bytes(client: TestClient) -> None:
    response = client.post(
        "/v1/tts", json={"text": "Hello kernel.", "language": "en"}, headers=AUTH
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content[:4] == b"RIFF"
    assert len(response.headers["x-synthesis-profile-id"]) == 64


def test_second_identical_sync_request_hits_the_hardened_cache(client: TestClient) -> None:
    payload = {"text": "Cache this exact message.\nPlease.", "language": "en"}
    first = client.post("/v1/tts", json=payload, headers=AUTH)
    calls = len(client.app.state.engine.calls)
    second = client.post("/v1/tts", json=payload, headers=AUTH)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.content == first.content
    assert (
        second.headers["x-synthesis-profile-id"]
        == first.headers["x-synthesis-profile-id"]
    )
    assert len(client.app.state.engine.calls) == calls


def test_model_mismatch_is_rejected(client: TestClient) -> None:
    payload = _batch_payload(model="some-other/model")
    response = client.post("/v1/tts/batch", json=payload, headers=AUTH)
    assert response.status_code == 409


def test_model_revision_mismatch_is_rejected(client: TestClient) -> None:
    payload = _batch_payload(model_revision="a" * 40)
    response = client.post("/v1/tts/batch", json=payload, headers=AUTH)
    assert response.status_code == 409


def test_unsupported_language_is_rejected(client: TestClient) -> None:
    payload = _batch_payload(language="xx")
    response = client.post("/v1/tts/batch", json=payload, headers=AUTH)
    assert response.status_code == 422


def test_duplicate_segment_keys_are_rejected(client: TestClient) -> None:
    payload = _batch_payload(
        segments=[{"key": "Scene1", "text": "a"}, {"key": "Scene1", "text": "b"}]
    )
    response = client.post("/v1/tts/batch", json=payload, headers=AUTH)
    assert response.status_code == 422


def test_invalid_reference_base64_is_rejected(client: TestClient) -> None:
    payload = _batch_payload(reference_audio_b64="not!!base64")
    response = client.post("/v1/tts/batch", json=payload, headers=AUTH)
    assert response.status_code == 422


def test_traversal_filenames_are_rejected(client: TestClient) -> None:
    response = client.post("/v1/tts/batch", json=_batch_payload(), headers=AUTH)
    job_id = response.json()["job_id"]
    _wait_for_job(client, job_id)
    assert client.get(f"/v1/jobs/{job_id}/audio/job.json", headers=AUTH).status_code == 404
    assert client.get(f"/v1/jobs/{job_id}/audio/..%2Fjob.json", headers=AUTH).status_code == 404
