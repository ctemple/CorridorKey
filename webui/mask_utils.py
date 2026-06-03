"""Fast classical chroma-screen mask utilities (pure numpy + OpenCV).

Ported from the CorridorKey HuggingFace Space for the local Web UI.
No GPU, ONNX, or PyTorch required — runs entirely on CPU in milliseconds.
"""

from __future__ import annotations

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