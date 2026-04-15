"""Tests for trajectory-based ROI generation via clustering."""

import math
from collections import deque

import numpy as np
import pytest

from backend.services.trajectory_cluster import (
    TrajectorySummary,
    TrajectoryAccumulator,
    TrajectoryClusterService,
    _build_direction_targets,
    _buffered_union_polygon,
    _circ_dist,
    _iqr_filter,
    _polygon_to_coords,
    _route_priority,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trails(
    n: int,
    center_x: float,
    center_y: float,
    dx: float,
    dy: float,
    spread: float = 20.0,
    trail_len: int = 10,
    rng: np.random.RandomState | None = None,
) -> dict[int, deque]:
    """Generate synthetic trails for n vehicles moving in (dx, dy) direction.

    Each trail has `trail_len` points centered around (center_x, center_y)
    with spatial spread.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    trails: dict[int, deque] = {}
    for i in range(n):
        cx = center_x + rng.normal(0, spread)
        cy = center_y + rng.normal(0, spread)
        trail = deque(maxlen=50)
        for t in range(trail_len):
            frac = t / trail_len
            trail.append((
                int(cx + dx * frac + rng.normal(0, 2)),
                int(cy + dy * frac + rng.normal(0, 2)),
            ))
        trails[i] = trail
    return trails


def _make_roads_ew(bearing: float = 90.0, distance_m: float = 100.0) -> list[dict]:
    """Create a pair of EB/WB road entries for an east-west highway."""
    return [
        {
            "route_label": "I-94",
            "cardinal": "EB",
            "bearing_deg": bearing,
            "bearing_to_road_deg": bearing,
            "distance_m": distance_m,
        },
        {
            "route_label": "I-94",
            "cardinal": "WB",
            "bearing_deg": bearing,
            "bearing_to_road_deg": bearing,
            "distance_m": distance_m,
        },
    ]


# ---------------------------------------------------------------------------
# Unit tests: _route_priority and _build_direction_targets
# ---------------------------------------------------------------------------

class TestRoutePriority:
    def test_interstate_highest(self):
        assert _route_priority("I-94") < _route_priority("US 12")
        assert _route_priority("I 94") < _route_priority("US 12")

    def test_us_over_state(self):
        assert _route_priority("US 12") < _route_priority("MN 51")

class TestBuildDirectionTargets:
    def test_merges_cosigned_routes(self):
        """Co-signed routes at same distance should merge, keeping interstate."""
        roads = [
            {"route_label": "US 12", "cardinal": "EB", "bearing_deg": 95.9, "distance_m": 26.3},
            {"route_label": "I 94", "cardinal": "EB", "bearing_deg": 122.5, "distance_m": 26.3},
            {"route_label": "US 52", "cardinal": "EB", "bearing_deg": 315.8, "distance_m": 26.3},
            {"route_label": "US 12", "cardinal": "WB", "bearing_deg": 95.9, "distance_m": 50.3},
            {"route_label": "I 94", "cardinal": "WB", "bearing_deg": 122.5, "distance_m": 50.3},
            {"route_label": "US 52", "cardinal": "WB", "bearing_deg": 315.8, "distance_m": 50.3},
        ]
        targets = _build_direction_targets(roads, azimuth=0.0)
        labels = {(t["route_label"], t["cardinal"]) for t in targets}
        # Should merge to I 94 EB + I 94 WB (interstate wins over US)
        assert ("I 94", "EB") in labels
        assert ("I 94", "WB") in labels
        assert ("US 12", "EB") not in labels
        assert ("US 52", "EB") not in labels

    def test_different_distances_not_merged(self):
        """Routes at different distances are different physical roads."""
        roads = [
            {"route_label": "I 94", "cardinal": "EB", "bearing_deg": 90.0, "distance_m": 26.3},
            {"route_label": "MN 51", "cardinal": "NB", "bearing_deg": 0.0, "distance_m": 101.8},
        ]
        targets = _build_direction_targets(roads, azimuth=0.0)
        labels = {t["route_label"] for t in targets}
        assert "I 94" in labels
        assert "MN 51" in labels


# ---------------------------------------------------------------------------
# Unit tests: _circ_dist
# ---------------------------------------------------------------------------

class TestCircDist:
    def test_same_angle(self):
        assert _circ_dist(90.0, 90.0) == 0.0

    def test_opposite(self):
        assert abs(_circ_dist(0.0, 180.0) - 180.0) < 0.01

    def test_wrap_around(self):
        assert abs(_circ_dist(350.0, 10.0) - 20.0) < 0.01

    def test_symmetric(self):
        assert abs(_circ_dist(30.0, 100.0) - _circ_dist(100.0, 30.0)) < 0.01


# ---------------------------------------------------------------------------
# Unit tests: _iqr_filter
# ---------------------------------------------------------------------------

class TestIQRFilter:
    def test_no_outliers(self):
        """Tightly clustered points should all pass."""
        members = [TrajectorySummary(100 + i, 200 + i, 90.0) for i in range(50)]
        filtered = _iqr_filter(members)
        assert len(filtered) == 50

    def test_removes_outliers(self):
        """Extreme outlier should be removed."""
        members = [TrajectorySummary(100 + i * 0.5, 200, 90.0) for i in range(50)]
        # Add outlier far from the cluster
        members.append(TrajectorySummary(900.0, 200, 90.0))
        filtered = _iqr_filter(members)
        assert len(filtered) == 50  # Outlier removed
        assert all(x < 200 for x, y in filtered)


# ---------------------------------------------------------------------------
# Unit tests: _convex_hull_polygon
# ---------------------------------------------------------------------------

class TestBufferedUnionPolygon:
    def test_clustered_points(self):
        """Clustered points should produce a valid polygon."""
        points = [(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)]
        result = _buffered_union_polygon(points, 720, 480)
        assert result is not None
        assert result.geom_type == "Polygon"
        assert result.area > 0

    def test_collinear_points_still_work(self):
        """Collinear points with buffer should still produce a polygon (unlike convex hull)."""
        points = [(100.0, 200.0), (200.0, 200.0), (300.0, 200.0)]
        result = _buffered_union_polygon(points, 720, 480)
        assert result is not None
        assert result.geom_type == "Polygon"

    def test_clips_to_image_bounds(self):
        """Points near edges should be clipped to image bounds."""
        points = [(-10.0, 240.0), (730.0, 240.0)]
        result = _buffered_union_polygon(points, 720, 480)
        assert result is not None
        # Should be within image bounds
        minx, miny, maxx, maxy = result.bounds
        assert minx >= 0
        assert miny >= 0
        assert maxx <= 720
        assert maxy <= 480

    def test_single_point_produces_circle(self):
        """A single point buffered should produce a circular polygon."""
        points = [(360.0, 240.0)]
        result = _buffered_union_polygon(points, 720, 480)
        assert result is not None
        assert result.geom_type == "Polygon"


# ---------------------------------------------------------------------------
# Unit tests: TrajectoryAccumulator
# ---------------------------------------------------------------------------

class TestTrajectoryAccumulator:
    def test_add_trails_basic(self):
        """Trails with enough length should be accepted."""
        acc = TrajectoryAccumulator("C001")
        rng = np.random.RandomState(42)
        trails = _make_trails(30, 300, 200, dx=50, dy=0, rng=rng)
        added = acc.add_trails(trails)
        assert added == 30
        assert acc.count == 30

    def test_skips_short_trails(self):
        """Trails with fewer than 5 points should be skipped."""
        acc = TrajectoryAccumulator("C001")
        trails = {0: deque([(100, 100), (105, 100), (110, 100)])}  # Only 3 points
        added = acc.add_trails(trails)
        assert added == 0

    def test_skips_stationary(self):
        """Trails with near-zero displacement should be skipped."""
        acc = TrajectoryAccumulator("C001")
        trails = {0: deque([(100, 100)] * 10)}  # No movement
        added = acc.add_trails(trails)
        assert added == 0

    def test_deduplicates_track_ids(self):
        """Same trail fed twice should only count once."""
        acc = TrajectoryAccumulator("C001")
        rng = np.random.RandomState(42)
        trails = _make_trails(10, 300, 200, dx=50, dy=0, rng=rng)
        acc.add_trails(trails)
        acc.add_trails(trails)  # Same track IDs
        assert acc.count == 10

    def test_is_ready(self):
        """Should be ready when count >= threshold."""
        acc = TrajectoryAccumulator("C001")
        rng = np.random.RandomState(42)
        # Create 250 trails (> 200 default threshold)
        trails = _make_trails(250, 300, 200, dx=50, dy=0, rng=rng)
        acc.add_trails(trails)
        assert acc.is_ready

    def test_angle_computation(self):
        """Rightward motion (dx>0, dy=0) should give ~90° in image convention."""
        acc = TrajectoryAccumulator("C001")
        trail = deque([(100, 200), (110, 200), (120, 200), (130, 200), (140, 200)])
        acc.add_trails({0: trail})
        assert acc.count == 1
        angle = acc._summaries[0].angle_deg
        # dx=40, dy=0 → atan2(40, 0) = 90°
        assert abs(angle - 90.0) < 1.0

    def test_generate_rois_two_directions(self):
        """Two opposing streams should produce two ROI polygons."""
        acc = TrajectoryAccumulator("C001")
        rng = np.random.RandomState(42)

        # Eastbound vehicles: upper portion of image, moving right
        eb_trails = _make_trails(
            120, center_x=360, center_y=180, dx=100, dy=0, spread=30, rng=rng
        )
        # Westbound vehicles: lower portion of image, moving left
        wb_trails = _make_trails(
            120, center_x=360, center_y=320, dx=-100, dy=0, spread=30,
            rng=np.random.RandomState(99),
        )

        # Merge with non-overlapping track IDs
        all_trails = {}
        for tid, trail in eb_trails.items():
            all_trails[tid] = trail
        for tid, trail in wb_trails.items():
            all_trails[tid + 1000] = trail

        acc.add_trails(all_trails)
        assert acc.count == 240

        # Camera looking north (azimuth=0): EB road=90° maps to pixel 90° (right)
        calibration = {"azimuth_offset_deg": 0.0}
        roads = _make_roads_ew(bearing=90.0)

        result = acc.generate_rois(roads, calibration)
        assert result is not None
        assert result.source == "trajectory_cluster"
        assert len(result.rois) == 2

        # Check both directions are represented
        directions = {r.direction for r in result.rois}
        assert "EB" in directions
        assert "WB" in directions

        # Check road names
        assert all(r.road_name == "I-94" for r in result.rois)

        # Check polygons are valid
        for roi in result.rois:
            assert len(roi.polygon) >= 3
            for x, y in roi.polygon:
                assert 0 <= x <= 720
                assert 0 <= y <= 480

    def test_generate_rois_no_calibration(self):
        """Should compute azimuth inline when no calibration is provided."""
        acc = TrajectoryAccumulator("C001")
        rng = np.random.RandomState(42)

        # Create clear east-west flow (dx dominant, dy~0)
        eb_trails = _make_trails(
            120, center_x=360, center_y=180, dx=100, dy=0, spread=30, rng=rng
        )
        wb_trails = _make_trails(
            120, center_x=360, center_y=320, dx=-100, dy=0, spread=30,
            rng=np.random.RandomState(99),
        )

        all_trails = {}
        for tid, trail in eb_trails.items():
            all_trails[tid] = trail
        for tid, trail in wb_trails.items():
            all_trails[tid + 1000] = trail

        acc.add_trails(all_trails)

        # No calibration — azimuth computed inline
        roads = _make_roads_ew(bearing=90.0)
        result = acc.generate_rois(roads, calibration=None)
        assert result is not None
        assert len(result.rois) == 2

    def test_generate_rois_too_few_per_group(self):
        """Groups with too few members should be skipped."""
        acc = TrajectoryAccumulator("C001")
        rng = np.random.RandomState(42)

        # Only 15 vehicles in one direction (below min_group_size=20)
        trails = _make_trails(15, 360, 200, dx=100, dy=0, rng=rng)
        acc.add_trails(trails)

        # Need to pad to meet overall minimum
        padding = _make_trails(
            190, 360, 300, dx=-100, dy=0,
            rng=np.random.RandomState(99),
        )
        for tid, trail in padding.items():
            acc.add_trails({tid + 1000: trail})

        calibration = {"azimuth_offset_deg": 0.0}
        roads = _make_roads_ew(bearing=90.0)

        result = acc.generate_rois(roads, calibration)
        # Should only produce 1 ROI (the WB group with 190), not the EB group with 15
        assert result is not None
        assert len(result.rois) == 1
        assert result.rois[0].direction == "WB"

    def test_clear_resets_state(self):
        acc = TrajectoryAccumulator("C001")
        rng = np.random.RandomState(42)
        trails = _make_trails(50, 300, 200, dx=50, dy=0, rng=rng)
        acc.add_trails(trails)
        assert acc.count == 50

        acc.clear()
        assert acc.count == 0
        assert not acc._generation_attempted

    def test_dedup_targets(self):
        """Duplicate road entries (overlapping shapefile segments) should be merged."""
        acc = TrajectoryAccumulator("C001")
        rng = np.random.RandomState(42)

        trails = _make_trails(120, 360, 180, dx=100, dy=0, rng=rng)
        wb_trails = _make_trails(
            120, 360, 320, dx=-100, dy=0,
            rng=np.random.RandomState(99),
        )
        all_trails = dict(trails)
        for tid, trail in wb_trails.items():
            all_trails[tid + 1000] = trail
        acc.add_trails(all_trails)

        # Duplicate road entries (common with shapefiles)
        roads = _make_roads_ew() + _make_roads_ew()  # 4 entries, but only 2 unique

        calibration = {"azimuth_offset_deg": 0.0}
        result = acc.generate_rois(roads, calibration)
        assert result is not None
        # Should still produce exactly 2 ROIs, not 4
        assert len(result.rois) == 2


# ---------------------------------------------------------------------------
# Integration: TrajectoryClusterService
# ---------------------------------------------------------------------------

class TestTrajectoryClusterService:
    def test_get_or_create(self):
        svc = TrajectoryClusterService()
        acc = svc.get_or_create_accumulator("C001")
        assert acc.camera_id == "C001"
        # Same ID returns same instance
        assert svc.get_or_create_accumulator("C001") is acc

    def test_invalidate(self):
        svc = TrajectoryClusterService()
        svc.get_or_create_accumulator("C001")
        svc.invalidate("C001")
        # Should create a fresh accumulator
        acc2 = svc.get_or_create_accumulator("C001")
        assert acc2.count == 0

    def test_get_status_empty(self):
        svc = TrajectoryClusterService()
        status = svc.get_status("C999")
        assert status["count"] == 0
        assert not status["is_ready"]
