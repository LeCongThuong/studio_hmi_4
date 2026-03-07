"""Optional OpenCV import helper."""

from __future__ import annotations

try:
    import cv2 as cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


def require_cv2(feature: str):
    if cv2 is None:
        raise ImportError(
            f"OpenCV (`cv2`) is required for {feature}. "
            "Install `opencv-python` or `opencv-python-headless` in this environment."
        )
    return cv2
