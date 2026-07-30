"""CorridorKey Web UI — FastAPI application.

Provides REST API + SSE for video upload, processing, and download.
Uses a background worker thread (``webui.worker.JobWorker``) for GPU-bound
pipeline execution so the event loop stays responsive.

Start with:   uv run uvicorn webui.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_UPLOAD_SIZE = 8 * 1024 * 1024 * 1024  # 8 GB

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CorridorKey Web UI",
    description="Browser-based green/blue screen keying powered by CorridorKey",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (CSS, JS)
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Global job store + worker (initialized at startup)
# ---------------------------------------------------------------------------
_jobs: dict[str, "Job"] = {}
_jobs_lock = asyncio.Lock()
_worker: "JobWorker | None" = None

from webui.models import InferenceParams, Job, JobState, ProgressReport  # noqa: E402
from webui.worker import JobWorker  # noqa: E402


@app.on_event("startup")
async def startup() -> None:
    global _worker
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    # Recover jobs from persisted state
    await _recover_jobs()

    # Start the background worker
    _worker = JobWorker()
    _worker.start()
    logger.info("CorridorKey Web UI started")


@app.on_event("shutdown")
async def shutdown() -> None:
    if _worker:
        _worker.stop()


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Serve the main web UI."""
    html_path = TEMPLATES_DIR / "index.html"
    if html_path.is_file():
        return html_path.read_text(encoding="utf-8")
    return "<h1>CorridorKey Web UI</h1><p>index.html not found — check the templates directory.</p>"


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)) -> dict:
    """Upload a video file. Returns a ``job_id`` for subsequent operations."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported format '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    upload_path = job_dir / f"input{ext}"
    # Stream to disk with size limit
    total = 0
    with open(upload_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1 MiB chunks
            total += len(chunk)
            if total > MAX_UPLOAD_SIZE:
                f.close()
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(413, "File too large (max 8 GB)")
            f.write(chunk)

    job = Job(
        job_id=job_id,
        state=JobState.UPLOADED,
        original_filename=file.filename,
        uploaded_path=str(upload_path),
        workspace_dir=str(job_dir),
        created_at=_utcnow_iso(),
    )

    # Probe input video to get bitrate for the UI default
    try:
        from backend.ffmpeg_tools import probe_video
        info = probe_video(str(upload_path))
        job.input_bitrate = info.get("bit_rate", 0)
    except Exception:
        logger.warning("Could not probe input video bitrate for job %s", job_id)
        job.input_bitrate = 0

    async with _jobs_lock:
        _jobs[job_id] = job

    # Initialize SSE queue
    job._sse_queue = asyncio.Queue(maxsize=200)

    _persist_job(job)
    logger.info("Upload complete: %s → job %s (%d bytes)", file.filename, job_id, total)

    return job.to_dict()


# ---------------------------------------------------------------------------
# Start processing
# ---------------------------------------------------------------------------

@app.post("/api/jobs/{job_id}/start")
async def start_job(
    job_id: str,
    screen_color: Annotated[str, Form()] = "auto",
    input_is_linear: Annotated[bool, Form()] = False,
    despill_strength: Annotated[int, Form()] = 5,       # 0–10 slider
    auto_despeckle: Annotated[bool, Form()] = True,
    despeckle_size: Annotated[int, Form()] = 400,
    image_size: Annotated[int, Form()] = 2048,
    refiner_scale: Annotated[float, Form()] = 1.0,
    generate_comp: Annotated[bool, Form()] = False,
    gpu_post_processing: Annotated[bool, Form()] = False,
    output_bitrate: Annotated[int, Form()] = 0,  # 0 = same as input
    mask_mode: Annotated[str, Form()] = "hybrid",  # "hybrid" | "ai" | "fast"
    smart_shrink: Annotated[bool, Form()] = True,
    output_format: Annotated[str, Form()] = "webm",  # "webm" | "mov"
) -> dict:
    """Configure parameters and enqueue the job for processing."""
    async with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, f"Job {job_id} not found")

        if job.state not in (JobState.UPLOADED, JobState.FAILED):
            raise HTTPException(409, f"Job {job_id} is already {job.state.value}")

        # Validate
        if screen_color not in ("auto", "green", "blue"):
            raise HTTPException(400, f"Invalid screen_color '{screen_color}'")
        if mask_mode not in ("hybrid", "ai", "fast"):
            raise HTTPException(400, f"Invalid mask_mode '{mask_mode}'")
        if output_format not in ("webm", "mov"):
            raise HTTPException(400, f"Invalid output_format '{output_format}'")
        if not 0 <= despill_strength <= 10:
            raise HTTPException(400, "despill_strength must be 0–10")
        if image_size not in (512, 1024, 2048):
            raise HTTPException(400, "image_size must be 512, 1024, or 2048")

        job.params = InferenceParams(
            screen_color=screen_color,
            input_is_linear=input_is_linear,
            despill_strength=despill_strength / 10.0,
            auto_despeckle=auto_despeckle,
            despeckle_size=despeckle_size,
            image_size=image_size,
            refiner_scale=refiner_scale,
            generate_comp=generate_comp,
            gpu_post_processing=gpu_post_processing,
            mask_mode=mask_mode,
            smart_shrink=smart_shrink,
            output_format=output_format,
        )

        # Reset error state if retrying
        job.error = None
        job.state = JobState.UPLOADED
        job.output_bitrate = output_bitrate

    _persist_job(job)

    # Submit to worker (synchronous — thread-safe queue)
    if _worker is None:
        raise HTTPException(500, "Worker not initialized")
    _worker.submit(job)

    return job.to_dict()


# ---------------------------------------------------------------------------
# Status & progress
# ---------------------------------------------------------------------------

@app.get("/api/jobs/{job_id}/status")
async def job_status(job_id: str) -> dict:
    """Get the current status and progress of a job."""
    async with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    qpos = 1 if job.state == JobState.QUEUED else 0
    return job.make_progress_report(queue_position=qpos).to_dict()


@app.get("/api/jobs")
async def list_jobs() -> list[dict]:
    """List all jobs (queue overview)."""
    async with _jobs_lock:
        all_jobs = list(_jobs.values())
        all_jobs.sort(key=lambda j: j.created_at, reverse=True)
        return [j.to_dict() for j in all_jobs]


# ---------------------------------------------------------------------------
# Server-Sent Events (real-time progress)
# ---------------------------------------------------------------------------

@app.get("/api/events/{job_id}")
async def job_events(job_id: str, request: Request) -> StreamingResponse:
    """SSE stream for real-time progress updates on a specific job."""
    async with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    if job._sse_queue is None:
        job._sse_queue = asyncio.Queue(maxsize=200)

    q = job._sse_queue

    async def event_generator():
        # Send initial state
        report = job.make_progress_report()
        yield f"data: {json.dumps(report.to_dict())}\n\n"

        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    report = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # Heartbeat to keep connection alive
                    yield ": heartbeat\n\n"
                    continue

                yield f"data: {json.dumps(report.to_dict())}\n\n"

                if report.state in (JobState.COMPLETED, JobState.FAILED):
                    break
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

@app.get("/api/jobs/{job_id}/download")
async def download_job(job_id: str) -> FileResponse:
    """Download the processed output video."""
    async with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    if job.state != JobState.COMPLETED:
        raise HTTPException(409, f"Job {job_id} is not completed (state: {job.state.value})")

    if not job.output_path or not os.path.exists(job.output_path):
        raise HTTPException(404, "Output file not found — it may have been cleaned up")

    download_name = os.path.basename(job.output_path)
    media_type = (
        "application/zip" if job.output_path.endswith(".zip") else
        "video/webm" if job.output_path.endswith(".webm") else
        "video/quicktime" if job.output_path.endswith(".mov") else
        "video/mp4" if job.output_path.endswith(".mp4") else
        "application/octet-stream"
    )
    return FileResponse(
        job.output_path,
        media_type=media_type,
        filename=download_name,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow_iso() -> str:
    """Return current UTC time as an ISO 8601 string (with Z suffix)."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _persist_job(job: Job) -> None:
    """Save job metadata to disk so state survives restarts (thread-safe)."""
    job_file = UPLOADS_DIR / job.job_id / "job.json"
    try:
        job_file.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")
    except Exception:
        logger.warning("Failed to persist job %s", job.job_id)


def _scan_for_output(job_dir: str) -> str | None:
    """Walk a job directory and return the path to the best output file.

    Prefers *_output.webm over other webm/mp4 files, and skips input.*
    files at the workspace root.
    Returns None if no output file is found.
    """
    candidates: list[tuple[int, str]] = []  # (score, path)
    for root, _dirs, files in os.walk(job_dir):
        for f in files:
            if not f.endswith((".webm", ".mp4", ".mov", ".zip")):
                continue
            full = os.path.join(root, f)
            # Skip input files (bare input.* at workspace root or clip root level)
            if os.path.basename(f).startswith("input."):
                continue
            if f.endswith("_output.zip"):
                return full  # Smart Shrink ZIP — best match
            if f.endswith("_output.webm") or f.endswith("_output.mov"):
                return full  # best match — return immediately
            if f.endswith("_comp.mp4"):
                candidates.append((2, full))
            elif f.endswith(".zip"):
                candidates.append((3, full))  # Smart Shrink ZIP
            elif f.endswith(".webm"):
                candidates.append((1, full))
            elif f.endswith(".mov"):
                candidates.append((1, full))
            elif f.endswith(".mp4"):
                candidates.append((0, full))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    return None


async def _recover_jobs() -> None:
    """Scan uploads/ for saved jobs and restore them into memory."""
    if not UPLOADS_DIR.is_dir():
        return

    for job_dir in UPLOADS_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        job_file = job_dir / "job.json"
        if not job_file.is_file():
            continue

        try:
            data = json.loads(job_file.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Skipping corrupt job.json in %s", job_dir.name)
            continue

        job_id = data.get("job_id", job_dir.name)
        job = Job(
            job_id=job_id,
            state=JobState(data.get("state", "uploaded")),
            original_filename=data.get("original_filename", ""),
            uploaded_path=data.get("uploaded_path", ""),
            workspace_dir=str(job_dir),
            output_path=data.get("output_filename"),
            input_bitrate=data.get("input_bitrate", 0),
            output_bitrate=data.get("output_bitrate", 0),
            created_at=data.get("created_at", ""),
            error=data.get("error"),
        )

        # Do not recover jobs that are still "processing" — worker died mid-job
        if job.state == JobState.PROCESSING:
            job.state = JobState.FAILED
            job.error = "Server was restarted while processing"
        elif job.state == JobState.QUEUED:
            job.state = JobState.UPLOADED  # Re-queue won't happen automatically

        # Auto-detect output files regardless of persisted state.
        # Old code never persisted job completion, so many jobs with valid
        # output files on disk still show "uploaded" / missing output_path.
        _detected = _scan_for_output(str(job_dir))
        if _detected:
            job.output_path = _detected
            job.state = JobState.COMPLETED
            job.error = None
            logger.info("Detected output for job %s: %s", job_id, _detected)

        job._sse_queue = asyncio.Queue(maxsize=200)
        async with _jobs_lock:
            _jobs[job_id] = job
        logger.info("Recovered job %s (state: %s)", job_id, job.state.value)

