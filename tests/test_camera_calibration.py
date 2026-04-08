"""Tests for camera orientation estimation from traffic flow analysis."""

import json
import math
import tempfile

import numpy as np
import pytest

from backend.services.camera_calibration import (
    FlowAccumulator,
    estimate_camera_azimuth,
    _circular_mean_axis,
)


class TestCircularMeanAxis:
    """Test the circular statistics helper for finding dominant flow axis."""

    def test_horizontal_flow(self):
        """Vehicles moving left and right → axis at 90° (east-west)."""
        angles = [90.0] * 50 + [270.0] * 50  # East + West
        axis = _circular_mean_axis(np.array(angles))
        assert abs(axis - 90.0) < 5.0 or abs(axis - 270.0) < 5.0

    def test_vertical_flow(self):
        """Vehicles moving up and down → axis at 0°/180° (north-south)."""
        angles = [0.0] * 50 + [180.0] * 50
        axis = _circular_mean_axis(np.array(angles))
        # 0° and 360° are equivalent
        assert min(axis, 360.0 - axis) < 5.0 or abs(axis - 180.0) < 5.0

    def test_diagonal_flow(self):
        """Vehicles at 45° and 225° → axis at 45°."""
        angles = [45.0] * 50 + [225.0] * 50
        axis = _circular_mean_axis(np.array(angles))
        assert abs(axis - 45.0) < 5.0 or abs(axis - 225.0) < 5.0

    def test_noisy_flow(self):
        """Dominant axis should survive moderate noise."""
        rng = np.random.RandomState(42)
        # East-west flow with ±15° noise
        angles_east = 90.0 + rng.normal(0, 15, 100)
        angles_west = 270.0 + rng.normal(0, 15, 100)
        angles = np.concatenate([angles_east, angles_west]) % 360
        axis = _circular_mean_axis(angles)
        # Should be near 90° or 270°
        assert min(abs(axis - 90.0), abs(axis - 270.0)) < 15.0


class TestFlowAccumulator:
    def test_accumulate_vectors(self):
        acc = FlowAccumulator(min_vehicles=5)
        for i in range(10):
            acc.add_velocity(dx=10.0, dy=0.0)  # Rightward motion
        assert acc.count == 10

    def test_not_ready_below_minimum(self):
        acc = FlowAccumulator(min_vehicles=50)
        for i in range(10):
            acc.add_velocity(dx=10.0, dy=0.0)
        assert not acc.is_ready

    def test_ready_above_minimum(self):
        acc = FlowAccumulator(min_vehicles=5)
        for i in range(10):
            acc.add_velocity(dx=10.0, dy=0.0)
        assert acc.is_ready

    def test_ignores_stationary(self):
        """Very small velocities should be ignored (parked cars, noise)."""
        acc = FlowAccumulator(min_vehicles=5)
        acc.add_velocity(dx=0.5, dy=0.3)  # Tiny motion → skip
        acc.add_velocity(dx=10.0, dy=0.0)  # Real motion
        assert acc.count == 1

    def test_get_dominant_axis(self):
        """Rightward motion in pixel space → dominant axis ~90° (image convention)."""
        acc = FlowAccumulator(min_vehicles=5)
        rng = np.random.RandomState(42)
        for _ in range(30):
            acc.add_velocity(dx=10.0 + rng.normal(0, 1), dy=rng.normal(0, 1))
        for _ in range(30):
            acc.add_velocity(dx=-10.0 + rng.normal(0, 1), dy=rng.normal(0, 1))
        axis = acc.get_dominant_axis()
        assert axis is not None
        # dx>0, dy~0 → atan2(dx, -dy) ≈ 90° (pixel convention: 0=up)
        assert min(abs(axis - 90.0), abs(axis - 270.0)) < 15.0


class TestEstimateCameraAzimuth:
    def test_camera_looking_north(self):
        """Camera looking north: pixel-up = geographic north.
        Flow axis in pixels ~0° matches road bearing ~0° → azimuth_offset ~0°.
        """
        pixel_flow_axis = 0.0  # Vehicles moving up/down in image
        road_bearing = 0.0  # Road runs north-south
        offset = estimate_camera_azimuth(pixel_flow_axis, road_bearing)
        assert abs(offset) < 5.0 or abs(offset - 360.0) < 5.0

    def test_camera_rotated_90(self):
        """Camera rotated 90° clockwise: pixel-up = geographic east.
        Vehicles on a N-S road (bearing 0°) would appear to move left-right
        in the image (pixel axis ~90°).
        azimuth_offset = road_bearing - pixel_axis = 0 - 90 = -90 → 270°
        """
        pixel_flow_axis = 90.0
        road_bearing = 0.0
        offset = estimate_camera_azimuth(pixel_flow_axis, road_bearing)
        assert abs(offset - 270.0) < 5.0 or abs(offset + 90.0) < 5.0

    def test_camera_looking_east(self):
        """Camera looking east: pixel-up = geographic east.
        E-W road (bearing 90°) vehicles move up/down (pixel axis 0°).
        azimuth_offset = 90 - 0 = 90°
        """
        pixel_flow_axis = 0.0
        road_bearing = 90.0
        offset = estimate_camera_azimuth(pixel_flow_axis, road_bearing)
        assert abs(offset - 90.0) < 5.0
