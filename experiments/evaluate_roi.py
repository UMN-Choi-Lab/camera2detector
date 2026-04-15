"""Evaluate ROI quality from trajectory clustering.

Loads collected trajectory data, runs the clustering algorithm,
and scores the resulting ROI polygons.

Usage:
    python experiments/evaluate_roi.py
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from shapely.geometry import MultiPoint, Point
from shapely.geometry import Polygon as ShapelyPolygon

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.trajectory_cluster import (
    TrajectorySummary,
    TrajectoryAccumulator,
    _build_direction_targets,
    _buffered_union_polygon,
    _circ_dist,
    _iqr_filter,
    _polygon_to_coords,
)
from backend.services.camera_calibration import (
    _circular_mean_axis,
    estimate_camera_azimuth,
)

# Load trajectory data
DATA_FILE = Path(__file__).parent / "C844_trajectories.json"
CAMERA_ID = "C844"
IMAGE_W, IMAGE_H = 720, 480

# C844 road info (from API)
ROADS = [
    {"route_label": "I 94", "cardinal": "EB", "bearing_deg": 122.5, "distance_m": 26.3, "bearing_to_road_deg": 0.7},
    {"route_label": "I 94", "cardinal": "WB", "bearing_deg": 122.5, "distance_m": 50.3, "bearing_to_road_deg": 0.7},
    # Exclude non-I-94 roads
]


def load_summaries() -> list[TrajectorySummary]:
    """Load trajectory summaries from collected data."""
    with open(DATA_FILE) as f:
        data = json.load(f)

    summaries = []
    for item in data["summaries"]:
        if isinstance(item, (list, tuple)):
            if len(item) >= 5:
                mean_cx, mean_cy, angle_deg, track_len, trail_pts = item[0], item[1], item[2], item[3], item[4]
                trail_points = tuple(tuple(p) for p in trail_pts) if trail_pts else ()
            else:
                mean_cx, mean_cy, angle_deg, track_len = item[0], item[1], item[2], item[3]
                trail_points = ()
        else:
            mean_cx, mean_cy, angle_deg = item["mean_cx"], item["mean_cy"], item["angle_deg"]
            trail_points = tuple(tuple(p) for p in item.get("trail_points", []))
        summaries.append(TrajectorySummary(mean_cx, mean_cy, angle_deg, trail_points))
    return summaries


def evaluate_rois(rois_dict: dict | None, summaries: list[TrajectorySummary]) -> dict:
    """Score ROI quality.

    Returns dict with all metric values.
    """
    if rois_dict is None or not rois_dict.get("rois"):
        return {
            "roi_quality": 0.0,
            "n_rois": 0,
            "total_vertices": 0,
            "overlap_area_pct": 0.0,
            "coverage_pct": 0.0,
            "separation_px": 0.0,
        }

    rois = rois_dict["rois"]
    n_rois = len(rois)

    # Build Shapely polygons
    polys = []
    for roi in rois:
        coords = roi["polygon"]
        if len(coords) >= 3:
            try:
                p = ShapelyPolygon(coords)
                if p.is_valid:
                    polys.append((p, roi))
            except Exception:
                pass

    if not polys:
        return {
            "roi_quality": 0.0,
            "n_rois": 0,
            "total_vertices": 0,
            "overlap_area_pct": 0.0,
            "coverage_pct": 0.0,
            "separation_px": 0.0,
        }

    # --- Metric 1: Number of ROIs (target: 2) ---
    roi_count_score = 100.0 if n_rois == 2 else max(0, 100 - 30 * abs(n_rois - 2))

    # --- Metric 2: Total vertices ---
    total_vertices = sum(len(roi["polygon"]) for roi in rois)
    vertex_score = 100.0
    if total_vertices > 30:
        vertex_score = max(0, 100 - 3 * (total_vertices - 30))
    elif total_vertices < 6:
        vertex_score = max(0, 100 - 20 * (6 - total_vertices))

    # --- Metric 3: Overlap between ROIs ---
    overlap_area = 0.0
    total_roi_area = sum(p.area for p, _ in polys)
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polys[i][0].intersects(polys[j][0]):
                overlap_area += polys[i][0].intersection(polys[j][0]).area
    overlap_pct = (overlap_area / max(total_roi_area, 1)) * 100
    overlap_score = max(0, 100 - 5 * overlap_pct)

    # --- Metric 4: Coverage — what fraction of trajectory points fall inside ROIs ---
    points_inside = 0
    for s in summaries:
        pt = Point(s.mean_cx, s.mean_cy)
        for p, _ in polys:
            if p.contains(pt):
                points_inside += 1
                break
    coverage_pct = (points_inside / max(len(summaries), 1)) * 100
    coverage_score = coverage_pct  # 0-100 directly

    # --- Metric 5: Separation — minimum distance between ROI polygons ---
    separation_px = 0.0
    if len(polys) >= 2:
        min_dist = float("inf")
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                d = polys[i][0].distance(polys[j][0])
                min_dist = min(min_dist, d)
        separation_px = min_dist if min_dist != float("inf") else 0.0
    # Separation score: some gap is good, too much means missing road area
    sep_score = min(100, separation_px * 5) if separation_px > 0 else 0
    if separation_px > 50:
        sep_score = max(0, 100 - (separation_px - 50) * 2)

    # --- Metric 6: Area efficiency — ROI area vs convex hull of all points ---
    all_pts = [(s.mean_cx, s.mean_cy) for s in summaries]
    if len(all_pts) >= 3:
        total_hull_area = MultiPoint(all_pts).convex_hull.area
        area_ratio = total_roi_area / max(total_hull_area, 1)
        # Ideal: ROIs cover the trajectory area without too much excess
        # Ratio close to 1.0 is perfect, >1.5 is too bloated, <0.3 is too small
        if area_ratio < 0.3:
            area_score = area_ratio / 0.3 * 100
        elif area_ratio <= 1.5:
            area_score = 100.0
        else:
            area_score = max(0, 100 - 30 * (area_ratio - 1.5))
    else:
        area_score = 0.0

    # --- Composite score ---
    roi_quality = (
        roi_count_score * 0.25 +
        coverage_score * 0.25 +
        overlap_score * 0.20 +
        vertex_score * 0.10 +
        sep_score * 0.10 +
        area_score * 0.10
    )

    return {
        "roi_quality": round(roi_quality, 2),
        "n_rois": n_rois,
        "total_vertices": total_vertices,
        "overlap_area_pct": round(overlap_pct, 2),
        "coverage_pct": round(coverage_pct, 2),
        "separation_px": round(separation_px, 2),
        "area_ratio": round(area_ratio if len(all_pts) >= 3 else 0, 3),
        # Sub-scores for debugging
        "_roi_count_score": round(roi_count_score, 1),
        "_coverage_score": round(coverage_score, 1),
        "_overlap_score": round(overlap_score, 1),
        "_vertex_score": round(vertex_score, 1),
        "_sep_score": round(sep_score, 1),
        "_area_score": round(area_score, 1),
    }


def run_current_algorithm(summaries: list[TrajectorySummary]) -> dict | None:
    """Run the current trajectory_cluster.py algorithm on collected data."""
    acc = TrajectoryAccumulator(CAMERA_ID)
    acc._summaries = list(summaries)
    acc._generation_attempted = False

    result = acc.generate_rois(
        roads=ROADS,
        calibration=None,  # Force inline azimuth computation
        image_width=IMAGE_W,
        image_height=IMAGE_H,
    )

    if result is None:
        return None
    return result.model_dump()


def main():
    summaries = load_summaries()
    print(f"Loaded {len(summaries)} trajectory summaries from {DATA_FILE.name}")

    result = run_current_algorithm(summaries)
    metrics = evaluate_rois(result, summaries)

    # Print METRIC lines for autoresearch.sh
    for k, v in metrics.items():
        if not k.startswith("_"):
            print(f"METRIC {k}={v}")

    # Print debug info
    print(f"\n--- Debug ---")
    if result:
        for roi in result.get("rois", []):
            print(f"  {roi['road_name']} {roi['direction']}: {len(roi['polygon'])} vertices")
    else:
        print("  No ROIs generated!")
    for k, v in metrics.items():
        if k.startswith("_"):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
