"""Camera movement detection via SSIM comparison against reference frames.

Stores a reference frame per camera and compares new frames using
Structural Similarity Index (SSIM). When SSIM drops below threshold,
the camera is flagged as having moved, invalidating existing ROIs.
"""

import json
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# SSIM constants (following Wang et al. 2004)
_C1 = (0.01 * 255) ** 2
_C2 = (0.03 * 255) ** 2
_COMPARE_SIZE = (360, 240)


def _compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute mean SSIM between two grayscale images.

    Uses Gaussian-weighted local statistics (11x11 window, sigma=1.5)
    per the original SSIM paper. Returns a scalar in [-1, 1].
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    numerator = (2 * mu1_mu2 + _C1) * (2 * sigma12 + _C2)
    denominator = (mu1_sq + mu2_sq + _C1) * (sigma1_sq + sigma2_sq + _C2)

    ssim_map = numerator / denominator
    return float(ssim_map.mean())


def _prepare_frame(frame: np.ndarray) -> np.ndarray:
    """Convert to grayscale and resize for SSIM comparison."""
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    return cv2.resize(gray, _COMPARE_SIZE, interpolation=cv2.INTER_AREA)


class CameraWatchdog:
    """Detects camera movement by comparing frames to stored references."""

    def __init__(self, reference_dir: str, ssim_threshold: float = 0.60):
        self._reference_dir = Path(reference_dir)
        self._reference_dir.mkdir(parents=True, exist_ok=True)
        self._ssim_threshold = ssim_threshold
        # In-memory cache of prepared reference frames (grayscale, resized)
        self._ref_cache: dict[str, np.ndarray] = {}

    def store_reference(self, camera_id: str, frame: np.ndarray) -> None:
        """Save a reference frame for the given camera."""
        ref_path = self._reference_dir / f"{camera_id}.jpg"
        cv2.imwrite(str(ref_path), frame)

        meta_path = self._reference_dir / f"{camera_id}_meta.json"
        meta = {
            "camera_id": camera_id,
            "stored_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "shape": list(frame.shape),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        # Update in-memory cache
        self._ref_cache[camera_id] = _prepare_frame(frame)
        logger.info("Stored reference frame for camera %s", camera_id)

    def _load_reference(self, camera_id: str) -> np.ndarray | None:
        """Load reference frame from cache or disk."""
        if camera_id in self._ref_cache:
            return self._ref_cache[camera_id]

        ref_path = self._reference_dir / f"{camera_id}.jpg"
        if not ref_path.exists():
            return None

        frame = cv2.imread(str(ref_path))
        if frame is None:
            return None

        prepared = _prepare_frame(frame)
        self._ref_cache[camera_id] = prepared
        return prepared

    def check_movement(
        self, camera_id: str, current_frame: np.ndarray
    ) -> dict:
        """Compare current frame against reference.

        Returns:
            {"moved": bool, "ssim": float | None}
            - moved=True if SSIM < threshold (camera has moved)
            - ssim=None on first call (no reference to compare against)
        """
        ref = self._load_reference(camera_id)

        if ref is None:
            # No reference yet — store this frame as baseline
            self.store_reference(camera_id, current_frame)
            return {"moved": False, "ssim": None}

        current_prepared = _prepare_frame(current_frame)
        ssim_score = _compute_ssim(ref, current_prepared)
        moved = ssim_score < self._ssim_threshold

        if moved:
            logger.warning(
                "Camera %s movement detected: SSIM=%.3f (threshold=%.2f)",
                camera_id,
                ssim_score,
                self._ssim_threshold,
            )

        return {"moved": moved, "ssim": round(ssim_score, 4)}
