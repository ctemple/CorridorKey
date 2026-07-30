"""Job data model, state machine, and SSE event types for the CorridorKey Web UI."""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from typing import Any


class JobState(str, enum.Enum):
    UPLOADED = "uploaded"       # Video uploaded, awaiting configuration
    QUEUED = "queued"           # In processing queue
    PROCESSING = "processing"   # Worker is actively processing
    COMPLETED = "completed"     # Done, output ready for download
    FAILED = "failed"           # Error occurred


class SubStep(str, enum.Enum):
    ORGANIZING = "organizing"
    ALPHA_GENERATION = "alpha_generation"
    INFERENCE = "inference"
    STITCHING = "stitching"


@dataclass
class InferenceParams:
    """Mirrors clip_manager.InferenceSettings — serialisable for the frontend."""
    screen_color: str = "auto"          # "auto" | "green" | "blue"
    input_is_linear: bool = False
    despill_strength: float = 0.5       # 0.0–1.0 (mapped from 0–10 slider)
    auto_despeckle: bool = True
    despeckle_size: int = 400
    image_size: int = 2048              # 512 | 1024 | 2048
    refiner_scale: float = 1.0
    generate_comp: bool = False
    gpu_post_processing: bool = False
    mask_mode: str = "hybrid"           # "hybrid" | "ai" | "fast"
    smart_shrink: bool = True           # auto-crop to content bounding box before stitch
    output_format: str = "webm"         # "webm" | "mov"

    def to_inference_settings(self):
        """Convert to clip_manager.InferenceSettings."""
        from clip_manager import InferenceSettings
        return InferenceSettings(
            screen_color=self.screen_color,
            input_is_linear=self.input_is_linear,
            despill_strength=self.despill_strength,
            auto_despeckle=self.auto_despeckle,
            despeckle_size=self.despeckle_size,
            image_size=self.image_size,
            refiner_scale=self.refiner_scale,
            generate_comp=self.generate_comp,
            gpu_post_processing=self.gpu_post_processing,
        )


@dataclass
class ProgressReport:
    """Progress snapshot sent to the frontend via SSE."""
    job_id: str
    state: JobState
    sub_step: SubStep | None = None
    sub_step_label: str = ""
    current_frame: int = 0
    total_frames: int = 0
    queue_position: int = 0
    error: str | None = None
    output_filename: str | None = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "state": self.state.value if isinstance(self.state, JobState) else self.state,
            "sub_step": self.sub_step.value if self.sub_step else None,
            "sub_step_label": self.sub_step_label,
            "current_frame": self.current_frame,
            "total_frames": self.total_frames,
            "queue_position": self.queue_position,
            "error": self.error,
            "output_filename": self.output_filename,
        }

    def to_sse_json(self) -> str:
        import json
        return json.dumps(self.to_dict())


@dataclass
class Job:
    """Tracks one complete job through the pipeline."""
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: JobState = JobState.UPLOADED
    original_filename: str = ""
    uploaded_path: str = ""          # Path to uploaded video file
    workspace_dir: str = ""          # Per-job working directory
    params: InferenceParams | None = None
    output_path: str | None = None   # Final downloadable output
    input_bitrate: int = 0           # Probed input video bitrate (bps)
    output_bitrate: int = 0          # User-selected output bitrate (bps, 0 = same as input)
    created_at: str = ""             # ISO 8601 timestamp of job creation
    error: str | None = None

    # Runtime progress (not persisted)
    sub_step: SubStep | None = None
    sub_step_label: str = ""
    current_frame: int = 0
    total_frames: int = 0
    _sse_queue: Any = None  # asyncio.Queue for SSE events

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "state": self.state.value,
            "original_filename": self.original_filename,
            "sub_step": self.sub_step.value if self.sub_step else None,
            "sub_step_label": self.sub_step_label,
            "current_frame": self.current_frame,
            "total_frames": self.total_frames,
            "error": self.error,
            "output_filename": self.output_path,
            "input_bitrate": self.input_bitrate,
            "output_bitrate": self.output_bitrate,
            "created_at": self.created_at,
            "params": {
                "screen_color": self.params.screen_color if self.params else "auto",
                "input_is_linear": self.params.input_is_linear if self.params else False,
                "despill_strength": int((self.params.despill_strength if self.params else 0.5) * 10),
                "auto_despeckle": self.params.auto_despeckle if self.params else True,
                "despeckle_size": self.params.despeckle_size if self.params else 400,
                "image_size": self.params.image_size if self.params else 2048,
                "refiner_scale": self.params.refiner_scale if self.params else 1.0,
                "generate_comp": self.params.generate_comp if self.params else True,
                "gpu_post_processing": self.params.gpu_post_processing if self.params else False,
                "mask_mode": self.params.mask_mode if self.params else "hybrid",
                "smart_shrink": self.params.smart_shrink if self.params else False,
                "output_format": self.params.output_format if self.params else "webm",
            } if self.params else None,
        }

    def make_progress_report(self, queue_position: int = 0) -> ProgressReport:
        return ProgressReport(
            job_id=self.job_id,
            state=self.state,
            sub_step=self.sub_step,
            sub_step_label=self.sub_step_label,
            current_frame=self.current_frame,
            total_frames=self.total_frames,
            queue_position=queue_position,
            error=self.error,
            output_filename=self.output_path,
        )
