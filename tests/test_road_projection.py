"""Tests for geometric road projection (pinhole camera model)."""

import math

import numpy as np
import pytest

from backend.services.road_projection import (
    CameraModel,
    project_road_to_roi,
)


class TestCameraModel:
    """Test the simplified pinhole camera model."""

    def test_point_directly_ahead_projects_to_center(self):
        """A point straight ahead of the camera should project near image center."""
        model = CameraModel(
            cam_east=0.0, cam_north=0.0,
            azimuth_deg=0.0,  # Looking north
            height_m=10.0, tilt_deg=30.0, fov_deg=60.0,
            image_width=720, image_height=480,
        )
        # Point 50m directly north, on the ground
        px, py = model.world_to_pixel(east=0.0, north=50.0, up=0.0)
        # Should be near horizontal center
        assert abs(px - 360) < 50  # Within 50px of center

    def test_point_to_the_right_projects_right(self):
        """A point to the right of the camera should project to the right half."""
        model = CameraModel(
            cam_east=0.0, cam_north=0.0,
            azimuth_deg=0.0,  # Looking north
            height_m=10.0, tilt_deg=30.0, fov_deg=60.0,
            image_width=720, image_height=480,
        )
        # Point 50m north, 20m to the east (right when looking north)
        px_right, _ = model.world_to_pixel(east=20.0, north=50.0, up=0.0)
        px_center, _ = model.world_to_pixel(east=0.0, north=50.0, up=0.0)
        assert px_right > px_center

    def test_closer_point_projects_lower(self):
        """A closer ground point should project lower in the image (camera looks down)."""
        model = CameraModel(
            cam_east=0.0, cam_north=0.0,
            azimuth_deg=0.0, height_m=10.0,
            tilt_deg=30.0, fov_deg=60.0,
            image_width=720, image_height=480,
        )
        _, py_near = model.world_to_pixel(east=0.0, north=20.0, up=0.0)
        _, py_far = model.world_to_pixel(east=0.0, north=100.0, up=0.0)
        # Near point should have larger y (lower in image)
        assert py_near > py_far

    def test_behind_camera_returns_none(self):
        """Points behind the camera should not project."""
        model = CameraModel(
            cam_east=0.0, cam_north=0.0,
            azimuth_deg=0.0, height_m=10.0,
            tilt_deg=30.0, fov_deg=60.0,
            image_width=720, image_height=480,
        )
        result = model.world_to_pixel(east=0.0, north=-50.0, up=0.0)
        assert result is None

    def test_azimuth_rotation(self):
        """Changing azimuth should rotate the projection."""
        # Camera looking east (azimuth 90°): a point due east should be centered
        model = CameraModel(
            cam_east=0.0, cam_north=0.0,
            azimuth_deg=90.0, height_m=10.0,
            tilt_deg=30.0, fov_deg=60.0,
            image_width=720, image_height=480,
        )
        px, py = model.world_to_pixel(east=50.0, north=0.0, up=0.0)
        assert px is not None
        assert abs(px - 360) < 50  # Near horizontal center


class TestProjectRoadToROI:
    def test_produces_polygon(self):
        """Should produce a closed polygon with at least 4 vertices."""
        cam = CameraModel(
            cam_east=500000.0, cam_north=4980000.0,
            azimuth_deg=0.0, height_m=10.0,
            tilt_deg=30.0, fov_deg=60.0,
            image_width=720, image_height=480,
        )
        # Road points running north, directly ahead
        road_utm = [
            [500000.0, 4980030.0],
            [500000.0, 4980060.0],
            [500000.0, 4980090.0],
            [500000.0, 4980120.0],
        ]
        polygon = project_road_to_roi(cam, road_utm, num_lanes=4, lane_width_m=3.7)
        assert polygon is not None
        assert len(polygon) >= 4

    def test_wider_road_produces_wider_roi(self):
        """More lanes → wider ROI polygon."""
        cam = CameraModel(
            cam_east=500000.0, cam_north=4980000.0,
            azimuth_deg=0.0, height_m=10.0,
            tilt_deg=30.0, fov_deg=60.0,
            image_width=720, image_height=480,
        )
        road_utm = [
            [500000.0, 4980040.0],
            [500000.0, 4980080.0],
        ]
        roi_narrow = project_road_to_roi(cam, road_utm, num_lanes=2, lane_width_m=3.7)
        roi_wide = project_road_to_roi(cam, road_utm, num_lanes=6, lane_width_m=3.7)
        assert roi_narrow is not None
        assert roi_wide is not None

        # Compute bounding box widths
        narrow_xs = [p[0] for p in roi_narrow]
        wide_xs = [p[0] for p in roi_wide]
        assert (max(wide_xs) - min(wide_xs)) > (max(narrow_xs) - min(narrow_xs))

    def test_road_behind_camera_returns_none(self):
        """A road entirely behind the camera should return None."""
        cam = CameraModel(
            cam_east=500000.0, cam_north=4980000.0,
            azimuth_deg=0.0, height_m=10.0,
            tilt_deg=30.0, fov_deg=60.0,
            image_width=720, image_height=480,
        )
        # Road behind camera (south)
        road_utm = [
            [500000.0, 4979900.0],
            [500000.0, 4979950.0],
        ]
        polygon = project_road_to_roi(cam, road_utm, num_lanes=4, lane_width_m=3.7)
        assert polygon is None
