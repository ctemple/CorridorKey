"""Fast classical chroma key mask generator.

Provides a lightweight, CPU-only alternative to BiRefNet for screen
color keying.  Uses HSV-based chroma keying with automatic (corner-sample)
or explicit screen-colour detection.

Ported from the CorridorKey HuggingFace Space.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Hue ranges for green / blue screen detection (OpenCV H in [0, 180)).
_GREEN_HUE_RANGE = (35, 85)
_BLUE_HUE_RANGE = (100, 130)

# Confidence threshold below which Hybrid mode falls back to BiRefNet.
HYBRID_CONFIDENCE_THRESHOLD = 0.7


def fast_chromascreen_mask(
    frame: np.ndarray,
    screen_color: str = "auto",
) -> tuple[np.ndarray | None, float]:
    """Generate a chroma-key mask from a single frame using classical CV.

    Args:
        frame: Input image in BGR uint8 or float32 format (H, W, 3).
        screen_color: ``"auto"`` (detect from corners), ``"green"``, or ``"blue"``.

    Returns:
        ``(mask, confidence)`` tuple.

        - *mask*: ``float32`` ``[H, W]`` array in ``[0.0, 1.0]``, or *None* if no
          screen colour could be determined.
        - *confidence*: ``float`` in ``[0.0, 1.0]``.  Higher = cleaner separation.
          Roughly ``1.0 - 2 * mean(min(mask, 1-mask))``.
    """
    h, w = frame.shape[:2]

    # -- detect screen colour from corner samples (when in auto mode) --------
    if screen_color == "auto":
        screen_color = _detect_screen_color_from_corners(frame)
        if screen_color is None:
            logger.warning("fast_chromascreen_mask: could not detect screen colour")
            return None, 0.0
        logger.debug("fast_chromascreen_mask: detected screen='%s'", screen_color)

    # -- convert to HSV ------------------------------------------------------
    if frame.dtype == np.float32 or frame.dtype == np.float64:
        # Normalise float → uint8
        frame_u8 = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        frame_u8 = frame

    hsv = cv2.cvtColor(frame_u8, cv2.COLOR_BGR2HSV)

    # -- threshold in HSV space ----------------------------------------------
    saturation_min = 30
    value_min = 30

    if screen_color == "green":
        lower = np.array([_GREEN_HUE_RANGE[0], saturation_min, value_min])
        upper = np.array([_GREEN_HUE_RANGE[1], 255, 255])
    else:  # blue
        lower = np.array([_BLUE_HUE_RANGE[0], saturation_min, value_min])
        upper = np.array([_BLUE_HUE_RANGE[1], 255, 255])

    mask_u8 = cv2.inRange(hsv, lower, upper)

    # -- morphological close to fill small holes -----------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)

    # -- slight blur to soften edges -----------------------------------------
    mask_u8 = cv2.GaussianBlur(mask_u8, (5, 5), 0)

    # -- convert to float [0, 1] ---------------------------------------------
    mask_f32 = mask_u8.astype(np.float32) / 255.0

    # -- confidence: how far the average pixel is from the 0.5 boundary ------
    #   conf = 1 - 2 * mean(min(m, 1-m))
    #   Perfect binary mask → mean(min(m,1-m)) → 0  → conf → 1
    #   All pixels at 0.5   → mean(min(m,1-m)) → 0.5 → conf → 0
    mean_ambiguity = float(np.mean(np.minimum(mask_f32, 1.0 - mask_f32)))
    confidence = 1.0 - 2.0 * mean_ambiguity

    return mask_f32.astype(np.float32), max(0.0, min(1.0, confidence))


def _detect_screen_color_from_corners(frame: np.ndarray) -> str | None:
    """Inspect the four corner regions and return ``"green"`` or ``"blue"``.

    Returns *None* when the corners are too small or ambiguous.
    """
    h, w = frame.shape[:2]
    corner_size = max(16, min(h, w) // 20)

    corners = [
        frame[:corner_size, :corner_size],                        # top-left
        frame[:corner_size, w - corner_size :],                   # top-right
        frame[h - corner_size :, :corner_size],                   # bottom-left
        frame[h - corner_size :, w - corner_size :],              # bottom-right
    ]

    samples = []
    for region in corners:
        if region.size == 0:
            continue
        mean_bgr = region.mean(axis=(0, 1))  # B, G, R
        samples.append(mean_bgr)

    if not samples:
        return None

    mean_colour = np.mean(samples, axis=0)
    mean_g = float(mean_colour[1])
    mean_b = float(mean_colour[0])

    # Require a meaningful difference before committing.
    if abs(mean_g - mean_b) < 5.0:
        logger.debug(
            "_detect_screen_color_from_corners: G=%.1f B=%.1f too close",
            mean_g, mean_b,
        )
        return None

    return "green" if mean_g > mean_b else "blue"