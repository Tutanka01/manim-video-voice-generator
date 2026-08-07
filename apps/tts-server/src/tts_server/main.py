"""FastAPI app: batch TTS jobs, sync synthesis, downloads, health.

One container = one lazily-loaded model: the weights are freed from VRAM after
an idle grace period (``TTS_SERVER_MODEL_IDLE_SECONDS``) and loaded again on
the next cache miss. The API is consumed by the video-api worker
(``generate_voice_en.py --engine moss-remote``) over a trusted LAN/VPN,
authenticated with ``Authorization: Bearer <key>`` (or ``X-API-Key``).
"""
from __future__ import annotations

import base64
import binascii
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from tts_server import __version__
from tts_server.cache import AudioCache
from tts_server.config import Settings, get_settings
from tts_server.engine import EngineNotReady, create_engine, moss_language_name
from tts_server.jobs import JobStore
from tts_server.schemas import BatchRequest, JobCreated, SyncRequest

logger = logging.getLogger(__name__)

# Budget granted to a synchronous synthesis for a cold model: with lazy
# loading the first request after an idle period has to download and load the
# ~16 GB checkpoint (the boot healthcheck historically granted 30 minutes).
_SYNC_LOAD_BUDGET_SECONDS = 1800.0


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level.upper())
    engine = create_engine(settings)
    cache = AudioCache(settings.cache_dir)
    jobs = JobStore(settings, engine, cache)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not settings.api_keys:
            logger.warning(
                "TTS_SERVER_API_KEYS is empty: authentication is DISABLED. "
                "Only acceptable on a trusted LAN/VPN."
            )
        engine.start()
        jobs.start()
        yield
        jobs.stop()

    app = FastAPI(title="MOSS TTS Server", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.engine = engine
    app.state.jobs = jobs

    def require_api_key(request: Request) -> None:
        if not settings.api_keys:
            return
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            token = header[7:].strip()
        else:
            token = request.headers.get("x-api-key", "").strip()
        if token not in settings.api_keys:
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    def _validate_language(language: str) -> None:
        try:
            moss_language_name(language)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    def _validate_model(requested: str | None, requested_revision: str | None) -> None:
        if requested and requested != settings.model_id:
            raise HTTPException(
                status_code=409,
                detail=f"this server runs {settings.model_id!r}, not {requested!r}",
            )
        if requested_revision and requested_revision.lower() != settings.model_revision.lower():
            raise HTTPException(
                status_code=409,
                detail=(
                    f"this server runs model revision {settings.model_revision!r}, "
                    f"not {requested_revision!r}"
                ),
            )

    def _decode_reference(encoded: str | None) -> bytes | None:
        if not encoded:
            return None
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as error:
            raise HTTPException(
                status_code=422, detail="reference_audio_b64 is not valid base64"
            ) from error
        if len(raw) > settings.max_reference_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"reference audio exceeds {settings.max_reference_bytes} bytes",
            )
        return raw

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        info = engine.info()
        engine_profile = engine.synthesis_profile()
        if engine_profile is not None:
            info["engine_profile_id"] = cache.engine_profile_id(engine_profile)
        info.update(
            {
                "version": __version__,
                "queue_depth": jobs.queue_depth(),
                "auth": bool(settings.api_keys),
            }
        )
        status_code = 200 if engine.state != "error" else 503
        if engine.state == "error":
            info["error"] = engine.load_error
        return JSONResponse(status_code=status_code, content=info)

    @app.post(
        "/v1/tts/batch",
        response_model=JobCreated,
        status_code=202,
        dependencies=[Depends(require_api_key)],
    )
    def create_batch(body: BatchRequest) -> JobCreated:
        _validate_model(body.model, body.model_revision)
        _validate_language(body.language)
        if len(body.segments) > settings.max_segments:
            raise HTTPException(
                status_code=413, detail=f"too many segments (max {settings.max_segments})"
            )
        keys = [segment.key for segment in body.segments]
        if len(set(keys)) != len(keys):
            raise HTTPException(status_code=422, detail="duplicate segment keys")
        for segment in body.segments:
            if len(segment.text) > settings.max_text_chars:
                raise HTTPException(
                    status_code=413,
                    detail=f"segment {segment.key!r} exceeds {settings.max_text_chars} characters",
                )
        reference = _decode_reference(body.reference_audio_b64)
        job = jobs.create(
            language=body.language,
            consistent_voice=body.consistent_voice,
            segments=[(segment.key, segment.text) for segment in body.segments],
            reference_bytes=reference,
        )
        return JobCreated(job_id=job.id, status=job.status, status_url=f"/v1/jobs/{job.id}")

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(require_api_key)])
    def job_status(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return jobs.public_state(job)

    @app.get("/v1/jobs/{job_id}/audio/{filename}", dependencies=[Depends(require_api_key)])
    def download_audio(job_id: str, filename: str) -> FileResponse:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        allowed = {f"{segment.key}.wav" for segment in job.segments} | {
            f"{segment.key}.mp3" for segment in job.segments
        }
        if filename not in allowed:
            raise HTTPException(status_code=404, detail="unknown audio file")
        if filename.endswith(".mp3"):
            try:
                path = jobs.ensure_mp3(job, filename.removesuffix(".mp3"))
            except FileNotFoundError as error:
                raise HTTPException(status_code=404, detail="audio not generated yet") from error
            except RuntimeError as error:
                raise HTTPException(status_code=503, detail=str(error)) from error
        else:
            path = jobs.job_dir(job_id) / filename
        if not path.exists():
            raise HTTPException(status_code=404, detail="audio not generated yet")
        media_type = "audio/wav" if filename.endswith(".wav") else "audio/mpeg"
        return FileResponse(path, media_type=media_type, filename=filename)

    @app.post("/v1/tts", dependencies=[Depends(require_api_key)])
    def synthesize_sync(body: SyncRequest) -> Response:
        _validate_model(body.model, body.model_revision)
        _validate_language(body.language)
        if len(body.text) > settings.max_text_chars:
            raise HTTPException(
                status_code=413, detail=f"text exceeds {settings.max_text_chars} characters"
            )
        reference = _decode_reference(body.reference_audio_b64)
        with tempfile.TemporaryDirectory(prefix="tts-sync-") as tmp_dir:
            reference_path: Path | None = None
            if reference:
                reference_path = Path(tmp_dir) / "reference.wav"
                reference_path.write_bytes(reference)
            try:
                # The model is lazy-loaded on the first request after an idle
                # period, so give a cold load (download + weights) a generous
                # budget before failing this test-only synchronous endpoint.
                engine.ensure_ready(timeout=_SYNC_LOAD_BUDGET_SECONDS)
            except EngineNotReady as error:
                raise HTTPException(status_code=503, detail=str(error)) from error
            engine_profile = engine.synthesis_profile()
            if engine_profile is None:
                raise HTTPException(status_code=503, detail="synthesis profile is unavailable")
            profile_id, fingerprint = cache.identity(
                engine_profile=engine_profile,
                language=body.language,
                text=body.text,
                reference_hash=cache.file_hash(reference_path),
            )
            cached = cache.lookup(fingerprint)
            if cached is not None:
                return Response(
                    content=cached.read_bytes(),
                    media_type="audio/wav",
                    headers={"X-Synthesis-Profile-ID": profile_id},
                )
            out_path = Path(tmp_dir) / "out.wav"
            engine.synthesize(body.text, body.language, reference_path, out_path)
            cache.store(fingerprint, out_path)
            return Response(
                content=out_path.read_bytes(),
                media_type="audio/wav",
                headers={"X-Synthesis-Profile-ID": profile_id},
            )

    return app
