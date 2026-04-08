"""Tests for camera movement detection via SSIM."""

import json
import os
import tempfile

import cv2
import numpy as np
import pytest

from backend.services.camera_watchdog import CameraWatchdog


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def watchdog(tmp_dir):
    return CameraWatchdog(reference_dir=tmp_dir, ssim_threshold=0.60)


def _make_highway_frame(seed: int = 42) -> np.ndarray:
    """Generate a synthetic 720x480 highway-like frame."""
    rng = np.random.RandomState(seed)
    frame = np.zeros((480, 720, 3), dtype=np.uint8)
    # Gray road surface
    frame[200:480, :, :] = 100
    # White lane markings
    frame[300:310, 150:570, :] = 255
    frame[350:360, 150:570, :] = 255
    # Some noise to simulate texture
    noise = rng.randint(0, 20, frame.shape, dtype=np.uint8)
    frame = cv2.add(frame, noise)
    return frame


def _make_shifted_frame(seed: int = 42) -> np.ndarray:
    """Generate a frame that simulates camera pan (shifted scene)."""
    rng = np.random.RandomState(seed)
    frame = np.zeros((480, 720, 3), dtype=np.uint8)
    # Road is now in a completely different position (camera panned)
    frame[0:280, :, :] = 100
    # Lane markings shifted
    frame[100:110, 200:620, :] = 255
    frame[150:160, 200:620, :] = 255
    noise = rng.randint(0, 20, frame.shape, dtype=np.uint8)
    frame = cv2.add(frame, noise)
    return frame


class TestCameraWatchdog:
    def test_store_reference_saves_file(self, watchdog, tmp_dir):
        frame = _make_highway_frame()
        watchdog.store_reference("C805", frame)
        assert os.path.exists(os.path.join(tmp_dir, "C805.jpg"))

    def test_store_reference_saves_metadata(self, watchdog, tmp_dir):
        frame = _make_highway_frame()
        watchdog.store_reference("C805", frame)
        meta_path = os.path.join(tmp_dir, "C805_meta.json")
        assert os.path.exists(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        assert "stored_at" in meta
        assert meta["camera_id"] == "C805"

    def test_check_movement_same_frame_not_moved(self, watchdog):
        """Same frame should produce SSIM close to 1.0 → no movement."""
        frame = _make_highway_frame()
        watchdog.store_reference("C805", frame)
        result = watchdog.check_movement("C805", frame)
        assert result["moved"] is False
        assert result["ssim"] > 0.95

    def test_check_movement_shifted_frame_moved(self, watchdog):
        """Panned camera should produce low SSIM → movement detected."""
        ref_frame = _make_highway_frame()
        shifted_frame = _make_shifted_frame()
        watchdog.store_reference("C805", ref_frame)
        result = watchdog.check_movement("C805", shifted_frame)
        assert result["moved"] is True
        assert result["ssim"] < 0.60

    def test_check_movement_no_reference_stores_it(self, watchdog, tmp_dir):
        """First check with no reference should store the frame and report no movement."""
        frame = _make_highway_frame()
        result = watchdog.check_movement("C805", frame)
        assert result["moved"] is False
        assert result["ssim"] is None  # No comparison possible
        # Reference should now be stored
        assert os.path.exists(os.path.join(tmp_dir, "C805.jpg"))

    def test_check_movement_minor_variation_not_moved(self, watchdog):
        """Traffic/lighting variation should not trigger movement."""
        ref_frame = _make_highway_frame(seed=42)
        # Same structure but different noise (simulates lighting change)
        varied_frame = _make_highway_frame(seed=99)
        watchdog.store_reference("C805", ref_frame)
        result = watchdog.check_movement("C805", varied_frame)
        assert result["moved"] is False
        assert result["ssim"] > 0.60

    def test_custom_threshold(self, tmp_dir):
        """A stricter threshold should be more sensitive to changes."""
        strict_watchdog = CameraWatchdog(reference_dir=tmp_dir, ssim_threshold=0.95)
        ref_frame = _make_highway_frame(seed=42)
        varied_frame = _make_highway_frame(seed=99)
        strict_watchdog.store_reference("C805", ref_frame)
        result = strict_watchdog.check_movement("C805", varied_frame)
        # With strict threshold, even minor noise might trigger
        # The key assertion is that threshold is respected
        assert (result["ssim"] < 0.95) == result["moved"]

    def test_update_reference_after_movement(self, watchdog):
        """After confirming movement, updating reference should reset detection."""
        ref_frame = _make_highway_frame()
        shifted_frame = _make_shifted_frame()
        watchdog.store_reference("C805", ref_frame)

        result = watchdog.check_movement("C805", shifted_frame)
        assert result["moved"] is True

        # Update reference to new position
        watchdog.store_reference("C805", shifted_frame)
        result = watchdog.check_movement("C805", shifted_frame)
        assert result["moved"] is False

    def test_multiple_cameras_independent(self, watchdog):
        """Each camera has independent reference frames."""
        frame_a = _make_highway_frame(seed=42)
        frame_b = _make_shifted_frame(seed=42)

        watchdog.store_reference("C805", frame_a)
        watchdog.store_reference("C809", frame_b)

        # Each camera compared to its own reference
        result_a = watchdog.check_movement("C805", frame_a)
        result_b = watchdog.check_movement("C809", frame_b)
        assert result_a["moved"] is False
        assert result_b["moved"] is False
