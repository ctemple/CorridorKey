"""Background job processor for the CorridorKey Web UI.

Runs in a dedicated thread, processing jobs one at a time (GPU serialization).
Communicates progress back to the main event loop via thread-safe callbacks.
"""

from __future__ import annotations

import logging
import os

# OpenCV EXR support must be enabled before importing cv2 (also set in mask_utils.py)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import queue
import shutil
import threading
import traceback
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from webui.models import Job

logger = logging.getLogger(__name__)


class JobWorker:
    """Serial job processor running in a background thread.

    Jobs are enqueued via :meth:`submit`. The worker picks them up one at a
    time, runs the full CorridorKey pipeline (organize → alpha → inference →
    stitch), and reports progress via each job's SSE queue.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[Job | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._main_loop = None  # set at start()
        self._running = False
        self._current_job: Job | None = None

    @property
    def current_job_id(self) -> str | None:
        if self._current_job:
            return self._current_job.job_id
        return None

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    def submit(self, job: Job) -> None:
        """Enqueue a job for processing (thread-safe — call from any thread)."""
        from webui.models import JobState

        job.state = JobState.QUEUED
        self._queue.put(job)
        self._push_progress(job)
        logger.info("Job %s enqueued (queue size: %d)", job.job_id, self._queue.qsize())

    def start(self) -> None:
        """Start the background worker thread."""
        if self._running:
            return
        self._running = True
        import asyncio as _asyncio

        self._main_loop = _asyncio.get_running_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="corridorkey-worker")
        self._thread.start()
        logger.info("Job worker started")

    def stop(self) -> None:
        """Signal the worker to stop and join the thread."""
        self._running = False
        # Push a sentinel to unblock the queue
        self._queue.put(None)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=30)
        logger.info("Job worker stopped")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _push_progress(self, job: Job, queue_position: int = 0) -> None:
        """Send a progress update to the job's SSE queue (thread-safe)."""
        if job._sse_queue is None or self._main_loop is None:
            return
        report = job.make_progress_report(queue_position=queue_position)
        try:
            self._main_loop.call_soon_threadsafe(job._sse_queue.put_nowait, report)
        except Exception:
            pass  # SSE queue might be full or closed — ignore

    @staticmethod
    def _persist_job_sync(job: Job) -> None:
        """Persist job state to disk (called from worker thread)."""
        import json as _json
        import pathlib as _pl

        job_file = _pl.Path(job.workspace_dir) / "job.json"
        try:
            job_file.write_text(_json.dumps(job.to_dict(), indent=2), encoding="utf-8")
        except Exception:
            logger.warning("Failed to persist job %s", job.job_id, exc_info=True)

    def _run_loop(self) -> None:
        """Main worker loop running in the background thread."""
        from webui.models import JobState

        logger.info("Worker thread started")

        while self._running:
            try:
                job = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if job is None:  # sentinel
                break

            self._current_job = job
            job.state = JobState.PROCESSING
            self._push_progress(job, queue_position=0)

            try:
                self._process_job(job)
            except Exception as exc:
                logger.error("Job %s failed:\n%s", job.job_id, traceback.format_exc())
                job.state = JobState.FAILED
                job.error = str(exc)
                self._push_progress(job)
                self._persist_job_sync(job)
            finally:
                self._current_job = None

        self._running = False
        logger.info("Worker thread stopped")

    def _process_job(self, job: Job) -> None:
        """Run the full pipeline on a single job (blocking — called from worker thread)."""
        from webui.models import JobState, SubStep

        # --- Step 1: Organize clip directory ---
        job.sub_step = SubStep.ORGANIZING
        job.sub_step_label = "Organizing clip structure…"
        self._push_progress(job)

        clip_name = _sanitize_clip_name(job.original_filename)
        input_path = os.path.join(job.workspace_dir, clip_name)
        os.makedirs(input_path, exist_ok=True)

        # Move uploaded video → Input.ext
        ext = os.path.splitext(job.original_filename)[1] or ".mp4"
        input_video = os.path.join(input_path, f"Input{ext}")
        if not os.path.exists(input_video):
            shutil.move(job.uploaded_path, input_video)

        # Create hint directories
        alpha_dir = os.path.join(input_path, "AlphaHint")
        os.makedirs(alpha_dir, exist_ok=True)
        os.makedirs(os.path.join(input_path, "VideoMamaMaskHint"), exist_ok=True)

        # --- Step 2: Generate alpha hints ---
        mask_mode = job.params.mask_mode if job.params else "hybrid"
        mode_labels = {"hybrid": "Hybrid (fast + AI fallback)", "ai": "AI (BiRefNet)", "fast": "Fast (classical)"}
        job.sub_step = SubStep.ALPHA_GENERATION
        job.sub_step_label = f"Generating alpha hints — {mode_labels.get(mask_mode, mask_mode)}…"
        self._push_progress(job)

        from clip_manager import ClipEntry, run_birefnet
        from device_utils import resolve_device

        clip = ClipEntry(clip_name, input_path)
        clip.find_assets()

        if clip.input_asset is None:
            raise RuntimeError(f"Could not find input asset for {clip_name}")

        device = resolve_device()
        total_frames = clip.input_asset.frame_count

        # ── AI-only mode: use existing batched BiRefNet pipeline ──
        if mask_mode == "ai":
            def _on_ai_frame(frame_idx: int, _num_frames: int) -> None:
                job.current_frame = frame_idx + 1
                job.total_frames = total_frames
                if (frame_idx + 1) % 5 == 0 or frame_idx + 1 >= total_frames:
                    self._push_progress(job)

            run_birefnet(
                [clip],
                device=device,
                usage="General",
                dilate_radius=0,
                on_clip_start=lambda _name, _n: None,
                on_frame_complete=_on_ai_frame,
            )

        # ── Fast / Hybrid mode: per-frame mask generation ──
        else:
            import cv2 as _cv2
            from webui.mask_utils import fast_chromascreen_mask as _fcsm, estimate_screen_color as _esc

            # Detect screen color from first frame
            resolved_color = "green"
            _cap = _cv2.VideoCapture(clip.input_asset.path)
            _ret, _frame_bgr = _cap.read()
            _cap.release()
            if _ret:
                _frame_f32 = _cv2.cvtColor(_frame_bgr, _cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                resolved_color = _esc(_frame_f32)
            logger.info("Detected screen color for '%s': %s", clip_name, resolved_color)

            # Load BiRefNet only for hybrid mode (as fallback)
            birefnet_handler = None
            if mask_mode == "hybrid":
                from BiRefNetModule.wrapper import BiRefNetHandler
                logger.info("Loading BiRefNet for hybrid fallback on %s…", device)
                birefnet_handler = BiRefNetHandler(device=device, usage="General")

            # Per-frame loop
            _cap = _cv2.VideoCapture(clip.input_asset.path)
            frame_idx = 0
            fast_count = 0
            ai_count = 0
            _stem = os.path.splitext(os.path.basename(clip.input_asset.path))[0]

            try:
                while True:
                    _ret, _frame_bgr = _cap.read()
                    if not _ret:
                        break

                    _frame_rgb = _cv2.cvtColor(_frame_bgr, _cv2.COLOR_BGR2RGB)
                    _frame_f32 = _frame_rgb.astype(np.float32) / 255.0

                    mask = None

                    if mask_mode == "fast":
                        mask, _conf, _ = _fcsm(_frame_f32, resolved_color)
                        if mask is None:
                            raise RuntimeError(
                                f"Fast mask failed on frame {frame_idx}. "
                                "Try 'Hybrid' or 'AI' mode."
                            )
                        fast_count += 1

                    else:  # hybrid
                        mask, conf, _ = _fcsm(_frame_f32, resolved_color)
                        if mask is None or conf < 0.7:
                            # AI fallback
                            _mask_u8 = birefnet_handler.process_frame(_frame_rgb)
                            mask = _mask_u8.astype(np.float32) / 255.0
                            ai_count += 1
                        else:
                            fast_count += 1

                    # Save mask
                    _mask_u8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
                    _save_path = os.path.join(alpha_dir, f"{_stem}_alpha_{frame_idx:05d}.png")
                    _cv2.imwrite(_save_path, _mask_u8)

                    frame_idx += 1
                    job.current_frame = frame_idx
                    job.total_frames = total_frames
                    if frame_idx % 5 == 0 or frame_idx >= total_frames:
                        self._push_progress(job)

            finally:
                _cap.release()
                if birefnet_handler is not None:
                    birefnet_handler.cleanup()

            logger.info(
                "Alpha generation for '%s': %d frames (fast=%d, ai=%d, mode=%s)",
                clip_name, frame_idx, fast_count, ai_count, mask_mode,
            )

        # Re-scan to pick up generated alphas
        clip.alpha_asset = None
        clip.find_assets()

        if clip.alpha_asset is None:
            raise RuntimeError("Alpha generation completed but no AlphaHint found")

        # --- Step 3: Run inference ---
        job.sub_step = SubStep.INFERENCE
        job.sub_step_label = "Running neural network inference…"
        job.total_frames = min(clip.input_asset.frame_count, clip.alpha_asset.frame_count)
        job.current_frame = 0
        self._push_progress(job)

        if job.params is None:
            from webui.models import InferenceParams
            job.params = InferenceParams()

        settings = job.params.to_inference_settings()

        from clip_manager import run_inference

        # Progress callbacks (called from worker thread → push to SSE)
        def on_clip_start(clip_name_inner: str, num_frames: int) -> None:
            job.total_frames = num_frames
            job.current_frame = 0
            job.sub_step_label = f"Processing {clip_name_inner}…"
            self._push_progress(job)

        def on_frame_complete(frame_idx: int, num_frames: int) -> None:
            job.current_frame = frame_idx + 1
            job.total_frames = num_frames
            # Only push every few frames to avoid overwhelming the SSE queue
            if (frame_idx + 1) % 5 == 0 or frame_idx + 1 >= num_frames:
                self._push_progress(job)

        run_inference(
            [clip],
            device=device,
            settings=settings,
            on_clip_start=on_clip_start,
            on_frame_complete=on_frame_complete,
        )

        output_dir = os.path.join(input_path, "Output")

        # --- Step 4: Smart Shrink (optional) ---
        smart_shrink_bbox: tuple[int, int, int, int] | None = None
        if job.params and job.params.smart_shrink:
            matte_dir = os.path.join(output_dir, "Matte")
            if os.path.isdir(matte_dir):
                from webui.mask_utils import compute_content_bbox

                smart_shrink_bbox = compute_content_bbox(
                    matte_dir, alpha_threshold=1.0 / 255.0
                )
                if smart_shrink_bbox is not None:
                    logger.info(
                        "Smart Shrink: crop=%dx%d+%d+%d",
                        smart_shrink_bbox[2], smart_shrink_bbox[3],
                        smart_shrink_bbox[0], smart_shrink_bbox[1],
                    )
                else:
                    logger.info("Smart Shrink: no benefit, skipping")

        # --- Step 5: Stitch output ---
        job.sub_step = SubStep.STITCHING
        job.sub_step_label = "Stitching output video…"
        self._push_progress(job)

        output_format = (job.params.output_format if job.params else "webm")
        if clip.input_asset and clip.input_asset.type == "video":
            if output_format == "mov":
                out_path = os.path.join(output_dir, f"{clip_name}_output.mov")
                self._stitch_mov_alpha(clip, output_dir, out_path, job.output_bitrate, crop=smart_shrink_bbox)
            else:
                out_path = os.path.join(output_dir, f"{clip_name}_output.webm")
                self._stitch_webm_alpha(clip, output_dir, out_path, job.output_bitrate, crop=smart_shrink_bbox)
            if os.path.exists(out_path):
                job.output_path = out_path
        else:
            self._package_sequence(output_dir)

        if job.output_path is None:
            # Fallback: look for any webm/mov/mp4 in output tree
            for root, _dirs, files in os.walk(output_dir):
                for f in files:
                    if f.endswith((".webm", ".mov")):
                        job.output_path = os.path.join(root, f)
                        break
                if job.output_path:
                    break

        # ── Clean up intermediate per-frame files to free disk space ──
        self._cleanup_intermediate_files(input_path)

        job.state = JobState.COMPLETED
        job.sub_step = None
        job.sub_step_label = ""
        self._push_progress(job)
        self._persist_job_sync(job)
        logger.info("Job %s completed: %s", job.job_id, job.output_path)

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stitch_webm_alpha(
        clip, output_dir: str, out_path: str, output_bitrate: int = 0,
        crop: tuple[int, int, int, int] | None = None,
    ) -> None:
        """Stitch FG + Matte EXR frames into a WebM with VP8 alpha + audio.

        Uses ffmpeg to combine the foreground (RGB) and matte (alpha)
        EXR sequences via the alphamerge filter, preserving the original
        video's audio, framerate, and resolution.

        Args:
            output_bitrate: User-selected bitrate in bps (0 = use input bitrate).
            crop: Optional ``(x, y, w, h)`` to crop both FG and Matte
                sequences before alphamerge (Smart Shrink).
        """
        import subprocess

        try:
            from backend.ffmpeg_tools import find_ffmpeg, find_ffprobe, probe_video
        except ImportError:
            logger.warning("ffmpeg_tools not available — skipping WebM stitch")
            return

        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            logger.warning("ffmpeg not found — skipping WebM stitch")
            return

        fg_dir = os.path.join(output_dir, "FG")
        matte_dir = os.path.join(output_dir, "Matte")

        if not os.path.isdir(fg_dir) or not os.path.isdir(matte_dir):
            logger.warning("FG or Matte directory missing — cannot stitch WebM with alpha")
            return

        fg_files = sorted(f for f in os.listdir(fg_dir) if f.lower().endswith(".exr"))
        matte_files = sorted(f for f in os.listdir(matte_dir) if f.lower().endswith(".exr"))
        if not fg_files or not matte_files:
            logger.warning("No EXR frames found in FG/Matte — cannot stitch")
            return

        # Determine frame pattern
        first = fg_files[0]
        stem = os.path.splitext(first)[0]
        if stem.isdigit():
            pattern = f"%0{len(stem)}d.exr"
        else:
            pattern = f"{stem}.exr"

        # Probe input video for fps, audio, and bitrate
        fps = 24.0
        has_audio = False
        bit_rate = 0
        try:
            info = probe_video(clip.input_asset.path)
            fps = info.get("fps", 24.0)
            has_audio = info.get("duration", 0) > 0
            bit_rate = info.get("bit_rate", 0)
        except Exception:
            logger.warning("Could not probe input video — using fps=24, no audio")

        # Determine video bitrate: user-selected value takes priority;
        # fall back to probed input bitrate, then ffmpeg default (CRF-based).
        effective_bitrate = output_bitrate if output_bitrate > 0 else bit_rate

        # Build ffmpeg command:
        # [0:v] FG frames (RGB)  [1:v] Matte frames (grayscale)  [2:a] original audio
        # alphamerge → RGBA → libvpx yuva420p + libvorbis audio
        if crop is not None:
            cx, cy, cw, ch = crop
            filter_complex = f"[0:v]crop={cw}:{ch}:{cx}:{cy}[fg];[1:v]crop={cw}:{ch}:{cx}:{cy}[matte];[fg][matte]alphamerge[out]"
        else:
            filter_complex = "[0:v][1:v]alphamerge[out]"

        cmd = [
            ffmpeg,
            "-framerate", str(fps),
            "-start_number", "0",
            "-i", os.path.join(fg_dir, pattern),
            "-framerate", str(fps),
            "-start_number", "0",
            "-i", os.path.join(matte_dir, pattern),
            "-i", clip.input_asset.path,
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-map", "2:a?",
            "-c:v", "libvpx",
            "-pix_fmt", "yuva420p",
            "-deadline", "good",
            "-cpu-used", "0",
            "-auto-alt-ref", "0",
            *(["-b:v", str(effective_bitrate)] if effective_bitrate > 0 else []),
            "-c:a", "libvorbis",
            "-shortest",
            out_path,
            "-y",
        ]

        logger.info(
            "Stitching WebM with alpha: FG + Matte → %s @ %s fps, bitrate=%s%s",
            out_path, fps, effective_bitrate or "auto",
            f", crop={crop[2]}x{crop[3]}+{crop[0]}+{crop[1]}" if crop else "",
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            if result.returncode != 0:
                # Log last 500 chars of stderr for debugging
                logger.error("ffmpeg WebM stitch failed: %s", result.stderr[-500:])
            else:
                logger.info("WebM with alpha written: %s", out_path)
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg WebM stitch timed out")
        except Exception as exc:
            logger.error("ffmpeg WebM stitch error: %s", exc)

    @staticmethod
    def _stitch_mov_alpha(
        clip, output_dir: str, out_path: str, output_bitrate: int = 0,
        crop: tuple[int, int, int, int] | None = None,
    ) -> None:
        """Stitch FG + Matte EXR frames into a MOV with ProRes 4444 alpha + audio.

        Uses ffmpeg to combine the foreground (RGB) and matte (alpha)
        EXR sequences via the alphamerge filter, encoding to Apple ProRes 4444
        (yuva444p10le) which preserves the alpha channel for professional VFX
        pipelines. Audio is copied from the original input.

        Args:
            output_bitrate: Ignored for ProRes (codec uses fixed profile-based
                bitrate). Kept for API consistency with _stitch_webm_alpha.
            crop: Optional ``(x, y, w, h)`` to crop both FG and Matte
                sequences before alphamerge (Smart Shrink).
        """
        import subprocess

        try:
            from backend.ffmpeg_tools import find_ffmpeg, probe_video
        except ImportError:
            logger.warning("ffmpeg_tools not available — skipping MOV stitch")
            return

        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            logger.warning("ffmpeg not found — skipping MOV stitch")
            return

        fg_dir = os.path.join(output_dir, "FG")
        matte_dir = os.path.join(output_dir, "Matte")

        if not os.path.isdir(fg_dir) or not os.path.isdir(matte_dir):
            logger.warning("FG or Matte directory missing — cannot stitch MOV with alpha")
            return

        fg_files = sorted(f for f in os.listdir(fg_dir) if f.lower().endswith(".exr"))
        matte_files = sorted(f for f in os.listdir(matte_dir) if f.lower().endswith(".exr"))
        if not fg_files or not matte_files:
            logger.warning("No EXR frames found in FG/Matte — cannot stitch")
            return

        # Determine frame pattern
        first = fg_files[0]
        stem = os.path.splitext(first)[0]
        if stem.isdigit():
            pattern = f"%0{len(stem)}d.exr"
        else:
            pattern = f"{stem}.exr"

        # Probe input video for fps and audio
        fps = 24.0
        has_audio = False
        try:
            info = probe_video(clip.input_asset.path)
            fps = info.get("fps", 24.0)
            has_audio = info.get("duration", 0) > 0
        except Exception:
            logger.warning("Could not probe input video — using fps=24, no audio")

        # Build filter: crop → alphamerge
        if crop is not None:
            cx, cy, cw, ch = crop
            filter_complex = (
                f"[0:v]crop={cw}:{ch}:{cx}:{cy}[fg];"
                f"[1:v]crop={cw}:{ch}:{cx}:{cy}[matte];"
                f"[fg][matte]alphamerge[out]"
            )
        else:
            filter_complex = "[0:v][1:v]alphamerge[out]"

        cmd = [
            ffmpeg,
            "-framerate", str(fps),
            "-start_number", "0",
            "-i", os.path.join(fg_dir, pattern),
            "-framerate", str(fps),
            "-start_number", "0",
            "-i", os.path.join(matte_dir, pattern),
            "-i", clip.input_asset.path,
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-map", "2:a?",
            "-c:v", "prores_ks",
            "-profile:v", "4",        # 4444 with alpha
            "-pix_fmt", "yuva444p10le",
            "-c:a", "pcm_s16le",       # PCM audio in MOV container
            "-shortest",
            out_path,
            "-y",
        ]

        logger.info(
            "Stitching MOV with alpha: FG + Matte → %s @ %s fps, ProRes 4444%s",
            out_path, fps,
            f", crop={crop[2]}x{crop[3]}+{crop[0]}+{crop[1]}" if crop else "",
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            if result.returncode != 0:
                logger.error("ffmpeg MOV stitch failed: %s", result.stderr[-500:])
            else:
                logger.info("MOV with alpha written: %s", out_path)
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg MOV stitch timed out")
        except Exception as exc:
            logger.error("ffmpeg MOV stitch error: %s", exc)

    @staticmethod
    def _package_sequence(output_dir: str) -> None:
        """For image sequence output — collect FG/Matte/Comp into a zip."""
        import zipfile

        zip_path = os.path.join(output_dir, "output.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(output_dir):
                for f in files:
                    if f == "output.zip":
                        continue
                    full = os.path.join(root, f)
                    arcname = os.path.relpath(full, output_dir)
                    zf.write(full, arcname)
        logger.info("Packaged output to %s", zip_path)

    @staticmethod
    def _cleanup_intermediate_files(input_path: str) -> None:
        """Remove intermediate per-frame files after successful video generation.

        Keeps: Input video, output videos (*_output.webm, *_comp.mp4),
               and job.json (at workspace level).
        Removes: AlphaHint/, VideoMamaMaskHint/, Output/FG/, Output/Matte/,
                 Output/Comp/, Output/Processed/.
        """
        dirs_to_remove = [
            os.path.join(input_path, "AlphaHint"),
            os.path.join(input_path, "VideoMamaMaskHint"),
            os.path.join(input_path, "Output", "FG"),
            os.path.join(input_path, "Output", "Matte"),
            os.path.join(input_path, "Output", "Comp"),
            os.path.join(input_path, "Output", "Processed"),
        ]

        for dir_path in dirs_to_remove:
            if os.path.isdir(dir_path):
                try:
                    shutil.rmtree(dir_path)
                    logger.info("Cleaned up intermediate directory: %s", dir_path)
                except OSError as exc:
                    logger.warning("Failed to clean up %s: %s", dir_path, exc)


def _sanitize_clip_name(filename: str) -> str:
    """Sanitize a filename into a safe clip directory name."""
    name = os.path.splitext(os.path.basename(filename))[0]
    safe = "".join(c if c.isalnum() or c in "._- " else "_" for c in name)
    safe = safe.strip().replace(" ", "_")
    if not safe:
        safe = "clip"
    return safe
