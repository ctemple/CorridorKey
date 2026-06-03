"""Fast classical chroma-screen mask utilities (pure numpy + OpenCV).

Ported from the CorridorKey HuggingFace Space for the local Web UI.
No GPU, ONNX, or PyTorch required — runs entirely on CPU in milliseconds.
"""

from __future__ import annotations

import os

# OpenCV EXR support must be enabled before importing cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np
import cv2


def estimate_screen_color(
    frame_f32: np.ndarray,
    alpha_hint: np.ndarray | None = None,
) -> str:
    """Detect green vs blue screen from background pixels.

    Args:
        frame_f32: RGB image as float32 in [0, 1], shape (H, W, 3).
        alpha_hint: Optional alpha mask float32 [0, 1] to identify background.

    Returns:
        ``"green"`` or ``"blue"``.
    """
    if alpha_hint is not None:
        if alpha_hint.ndim == 3:
            alpha_hint = alpha_hint[:, :, 0]
        bg_mask = alpha_hint < 0.3
    else:
        h, w = frame_f32.shape[:2]
        ph, pw = max(int(h * 0.05), 4), max(int(w * 0.05), 4)
        bg_mask = np.zeros((h, w), dtype=bool)
        bg_mask[:ph, :pw] = True
        bg_mask[:ph, -pw:] = True
        bg_mask[-ph:, :pw] = True
        bg_mask[-ph:, -pw:] = True

    if bg_mask.mean() < 0.01:
        return "green"

    bg = frame_f32[bg_mask]
    mean_g = float(bg[:, 1].mean())
    mean_b = float(bg[:, 2].mean())

    if abs(mean_g - mean_b) < 0.05:
        return "green"
    return "blue" if mean_b > mean_g else "green"


def fast_chromascreen_mask(
    frame_rgb_f32: np.ndarray,
    screen_color: str = "auto",
) -> tuple[np.ndarray | None, float, str]:
    """Fast classical mask for green or blue screens using HSV thresholding.

    Samples corner pixels to estimate the dominant background colour, then
    thresholds the frame in HSV space to produce a binary mask.  Falls back
    gracefully when no green/blue screen is detected.

    Args:
        frame_rgb_f32: RGB image as float32 in [0, 1], shape (H, W, 3).
        screen_color: ``"green"``, ``"blue"``, or ``"auto"``.

    Returns:
        Tuple of ``(mask_f32, confidence, detected_color)`` where:
        * ``mask_f32`` is a float32 alpha mask in [0, 1] (or ``None`` on
          failure).
        * ``confidence`` is a float in [0, 1] estimating mask quality.
        * ``detected_color`` is the resolved screen colour string.
    """
    h, w = frame_rgb_f32.shape[:2]
    ph = max(int(h * 0.05), 4)
    pw = max(int(w * 0.05), 4)

    corners = np.concatenate(
        [
            frame_rgb_f32[:ph, :pw].reshape(-1, 3),
            frame_rgb_f32[:ph, -pw:].reshape(-1, 3),
            frame_rgb_f32[-ph:, :pw].reshape(-1, 3),
            frame_rgb_f32[-ph:, -pw:].reshape(-1, 3),
        ],
        axis=0,
    )
    bg_color = np.median(corners, axis=0)

    is_green = (bg_color[1] > bg_color[0] + 0.05) and (bg_color[1] > bg_color[2] + 0.05)
    is_blue = (bg_color[2] > bg_color[0] + 0.05) and (bg_color[2] > bg_color[1] + 0.05)

    if screen_color == "green" and not is_green:
        return None, 0.0, "green"
    if screen_color == "blue" and not is_blue:
        return None, 0.0, "blue"
    if screen_color == "auto" and not is_green and not is_blue:
        return None, 0.0, "green"

    if screen_color != "auto":
        detected = screen_color
    else:
        detected = "blue" if (is_blue and not is_green) else "green"

    # HSV thresholding
    frame_u8 = (np.clip(frame_rgb_f32, 0, 1) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2HSV)

    if detected == "blue":
        screen_mask = cv2.inRange(hsv, (100, 40, 40), (130, 255, 255))
    else:
        screen_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))

    fg_mask = cv2.bitwise_not(screen_mask)
    fg_mask = cv2.morphologyEx(
        fg_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)

    mask_f32 = fg_mask.astype(np.float32) / 255.0

    # Confidence: how far the mask is from ambiguous (0.5)
    confidence = 1.0 - 2.0 * float(np.mean(np.minimum(mask_f32, 1.0 - mask_f32)))

    return mask_f32, confidence, detected


def compute_content_bbox(
    matte_dir: str, alpha_threshold: float = 0.0
) -> tuple[int, int, int, int] | None:
    """Compute the global bounding box of non-transparent content across all Matte EXR frames.

    Scans every ``.exr`` file in *matte_dir*, finds pixels where alpha >
    *alpha_threshold*, and returns the union bounding box across all frames.
    Width and height are always rounded up to even numbers.

    Args:
        matte_dir: Directory containing grayscale EXR matte frames.
        alpha_threshold: Minimum alpha value to consider a pixel non-transparent.
            Default 0.0 (any non-zero pixel).  Increase to 1/255 to ignore
            sub-LSB noise that some ML models produce in background regions.

    Returns:
        ``(x, y, width, height)`` crop rectangle, or ``None`` if no
        non-transparent content was found or the bounding box already
        covers the full frame.
    """
    import logging
    import os

    _log = logging.getLogger(__name__)

    if not os.path.isdir(matte_dir):
        _log.warning("Smart Shrink: matte_dir not found: %s", matte_dir)
        return None

    exr_files = sorted(
        f for f in os.listdir(matte_dir) if f.lower().endswith(".exr")
    )
    if not exr_files:
        _log.warning("Smart Shrink: no EXR files in %s", matte_dir)
        return None

    _log.info("Smart Shrink: scanning %d matte frames (threshold=%.6f)…", len(exr_files), alpha_threshold)

    global_min_x: int | None = None
    global_min_y: int | None = None
    global_max_x: int | None = None
    global_max_y: int | None = None
    frame_h: int = 0
    frame_w: int = 0
    total_nonzero: int = 0
    frames_with_content: int = 0
    first_frame_stats: bool = False

    for fname in exr_files:
        fpath = os.path.join(matte_dir, fname)
        img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if img is None:
            _log.warning("Smart Shrink: cv2.imread returned None for %s", fname)
            continue

        h, w = img.shape[:2]
        frame_h, frame_w = h, w

        if img.ndim == 3 and img.shape[2] > 1:
            alpha = img[:, :, 0].astype(np.float32)
        else:
            alpha = img.astype(np.float32)

        # Log first-frame diagnostics
        if not first_frame_stats:
            first_frame_stats = True
            _log.info(
                "Smart Shrink: frame[0] shape=%s dtype=%s min=%.6f max=%.6f mean=%.6f",
                img.shape, img.dtype, alpha.min(), alpha.max(), alpha.mean(),
            )
            # Edge checks
            _log.info(
                "Smart Shrink: frame[0] top-row-max=%.6f bot-row-max=%.6f "
                "left-col-max=%.6f right-col-max=%.6f",
                alpha[0, :].max(), alpha[h - 1, :].max(),
                alpha[:, 0].max(), alpha[:, w - 1].max(),
            )

        # Find non-transparent pixels
        ys, xs = np.where(alpha > alpha_threshold)
        if len(xs) == 0:
            continue  # fully transparent frame

        frames_with_content += 1
        total_nonzero += len(xs)

        min_x, max_x = int(xs.min()), int(xs.max())
        min_y, max_y = int(ys.min()), int(ys.max())

        if global_min_x is None:
            global_min_x, global_min_y = min_x, min_y
            global_max_x, global_max_y = max_x, max_y
        else:
            global_min_x = min(global_min_x, min_x)
            global_min_y = min(global_min_y, min_y)
            global_max_x = max(global_max_x, max_x)
            global_max_y = max(global_max_y, max_y)

    if global_min_x is None:
        _log.warning("Smart Shrink: all frames fully transparent — skipping")
        return None

    merged_w = global_max_x - global_min_x + 1
    merged_h = global_max_y - global_min_y + 1

    _log.info(
        "Smart Shrink: %d/%d frames have content, avg %.0f px/frame, "
        "merged bbox=(%d,%d,%d,%d) frame=%dx%d",
        frames_with_content, len(exr_files),
        total_nonzero / max(frames_with_content, 1),
        global_min_x, global_min_y, merged_w, merged_h,
        frame_w, frame_h,
    )

    # Check if bounding box already covers the full frame (no benefit)
    if (
        global_min_x == 0
        and global_min_y == 0
        and global_max_x == frame_w - 1
        and global_max_y == frame_h - 1
    ):
        _log.info("Smart Shrink: bbox covers full frame — skipping")
        return None

    x = global_min_x
    y = global_min_y
    w = global_max_x - global_min_x + 1
    h = global_max_y - global_min_y + 1

    # Ensure even dimensions (required by many video codecs)
    if w % 2 != 0:
        w += 1
    if h % 2 != 0:
        h += 1

    _log.info("Smart Shrink: final crop=%dx%d+%d+%d (was %dx%d)", w, h, x, y, frame_w, frame_h)
    return (x, y, w, h)