"""ROI generation from clustered vehicle trajectories.

Accumulates trajectory summaries (mean position + direction) from ByteTrack
across multiple 30-second windows.  When enough data is collected, groups
trajectories by road-direction using the camera's calibration azimuth (or an
inline estimate), removes spatial outliers via IQR, and computes convex hull
polygons as ROIs.
"""

import logging
import math
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import NamedTuple

import numpy as np
from shapely.geometry import MultiPoint
from shapely.geometry import Polygon as ShapelyPolygon

from backend.config import settings
from backend.models.schemas import CameraROIs, ROIPolygon
from backend.services.camera_calibration import (
    _circular_mean_axis,
    estimate_camera_azimuth,
)

logger = logging.getLogger(__name__)

# Reuse color palette from VLM ROI module
ROI_COLORS = [
    "#a855f7",  # purple
    "#3b82f6",  # blue
    "#ef4444",  # red
    "#f59e0b",  # amber
    "#10b981",  # emerald
    "#ec4899",  # pink
    "#06b6d4",  # cyan
    "#f97316",  # orange
]

_MIN_DISPLACEMENT = 3.0  # pixels — skip near-stationary vehicles


class TrajectorySummary(NamedTuple):
    mean_cx: float   # mean centroid x across the trail
    mean_cy: float   # mean centroid y across the trail
    angle_deg: float  # travel direction [0, 360), 0=image-up
    trail_points: tuple = ()  # sampled (cx, cy) points along the trail


def _circ_dist(a: float, b: float) -> float:
    """Circular angular distance in degrees, always in [0, 180]."""
    d = abs(a - b) % 360
    return min(d, 360 - d)


_CARDINAL_TO_BEARING = {"NB": 0, "EB": 90, "SB": 180, "WB": 270}

# Route priority: lower = more preferred (interstates > US > state)
def _route_priority(label: str) -> int:
    label = label.strip().upper()
    if label.startswith("I ") or label.startswith("I-"):
        return 0  # Interstate
    if label.startswith("US "):
        return 1  # US highway
    return 2  # State / other


def _build_direction_targets(roads: list[dict], azimuth: float) -> list[dict]:
    """Build direction targets, merging co-signed routes on the same physical road.

    Co-signed routes (e.g., US 12 / I-94 / US 52) share the same distance_m.
    We group by (distance_m, cardinal), then keep the highest-priority route name
    (interstate > US highway > state route) and average the bearings.
    """
    # Group roads by (rounded distance, cardinal) to detect co-signed routes
    from collections import defaultdict as _dd
    groups: dict[tuple[float, str], list[dict]] = _dd(list)
    for road in roads:
        cardinal = road.get("cardinal", "")
        if cardinal not in _CARDINAL_TO_BEARING:
            continue
        dist_key = round(road.get("distance_m", 0), 0)  # Round to 1m for grouping
        groups[(dist_key, cardinal)].append(road)

    targets = []
    seen = set()
    for (_dist, cardinal), members in groups.items():
        # Pick the most significant route name
        best = min(members, key=lambda r: _route_priority(r.get("route_label", "")))
        route_label = best.get("route_label", "Unknown")

        # Average the bearings for this group
        bearing = best.get("bearing_deg", 0)
        direction_bearing = _CARDINAL_TO_BEARING[cardinal]
        expected_pixel_angle = (direction_bearing - azimuth) % 360

        key = (route_label, cardinal)
        if key in seen:
            continue
        seen.add(key)

        targets.append({
            "route_label": route_label,
            "cardinal": cardinal,
            "expected_pixel_angle": expected_pixel_angle,
        })

    return targets


class TrajectoryAccumulator:
    """Collects trajectory summaries for one camera across 30s windows."""

    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self._summaries: list[TrajectorySummary] = []
        self._seen_track_ids: set[int] = set()
        self._generation_attempted = False
        self._windows_fed = 0

    @property
    def count(self) -> int:
        return len(self._summaries)

    @property
    def is_ready(self) -> bool:
        return self.count >= settings.cluster_min_trajectories

    def add_trails(self, trails: dict[int, deque]) -> int:
        """Extract trajectory summaries from current trail snapshot.

        Args:
            trails: track_id -> deque of (cx, cy) centroid positions.

        Returns:
            Number of new summaries added this call.
        """
        added = 0
        for tid, trail in trails.items():
            if tid in self._seen_track_ids:
                continue
            if len(trail) < 5:
                continue

            xs = [p[0] for p in trail]
            ys = [p[1] for p in trail]

            # Displacement from first to last point
            dx = trail[-1][0] - trail[0][0]
            dy = trail[-1][1] - trail[0][1]
            displacement = math.sqrt(dx * dx + dy * dy)
            if displacement < _MIN_DISPLACEMENT:
                continue

            angle = math.degrees(math.atan2(dx, -dy)) % 360
            # Sample up to 10 evenly-spaced points along the trail
            n = len(trail)
            max_pts = 10
            if n <= max_pts:
                sampled = tuple((float(p[0]), float(p[1])) for p in trail)
            else:
                indices = [int(i * (n - 1) / (max_pts - 1)) for i in range(max_pts)]
                sampled = tuple((float(trail[i][0]), float(trail[i][1])) for i in indices)
            self._summaries.append(TrajectorySummary(
                mean_cx=sum(xs) / len(xs),
                mean_cy=sum(ys) / len(ys),
                angle_deg=angle,
                trail_points=sampled,
            ))
            self._seen_track_ids.add(tid)
            added += 1

        if added > 0:
            self._windows_fed += 1

        return added

    def generate_rois(
        self,
        roads: list[dict],
        calibration: dict | None = None,
        image_width: int = 720,
        image_height: int = 480,
    ) -> CameraROIs | None:
        """Cluster trajectories by road-direction and compute ROI polygons.

        Args:
            roads: Nearby roads from road_geometry_service.get_camera_roads().
            calibration: Calibration dict with 'azimuth_offset_deg', or None.
            image_width: Image width in pixels.
            image_height: Image height in pixels.

        Returns:
            CameraROIs with convex hull polygons, or None if no valid groups.
        """
        if not self._summaries or not roads:
            return None

        self._generation_attempted = True

        # Step 1: Determine azimuth (pixel-to-geographic mapping)
        azimuth = self._resolve_azimuth(roads, calibration)
        if azimuth is None:
            logger.warning("Camera %s: could not determine azimuth for clustering",
                           self.camera_id)
            return None

        # Step 2: Build road-direction targets with expected pixel angles.
        # Merge co-signed routes (same distance = same physical road) and
        # keep the most significant route name (interstate > US > state).
        targets = _build_direction_targets(roads, azimuth)

        if not targets:
            logger.warning("Camera %s: no valid road-direction targets", self.camera_id)
            return None

        # Step 3: Assign each trajectory to nearest road-direction
        max_angle = settings.cluster_max_angle_deg
        groups: dict[tuple[str, str], list[TrajectorySummary]] = defaultdict(list)

        for s in self._summaries:
            best_target = None
            best_dist = max_angle
            for t in targets:
                dist = _circ_dist(s.angle_deg, t["expected_pixel_angle"])
                if dist < best_dist:
                    best_dist = dist
                    best_target = t
            if best_target is not None:
                key = (best_target["route_label"], best_target["cardinal"])
                groups[key].append(s)

        # Step 4: Per-group outlier removal
        filtered_groups: dict[tuple[str, str], list[tuple[float, float]]] = {}

        for (route_label, cardinal), members in groups.items():
            if len(members) < settings.cluster_min_group_size:
                logger.debug(
                    "Camera %s: skipping %s %s — only %d trajectories",
                    self.camera_id, route_label, cardinal, len(members),
                )
                continue

            filtered = _iqr_filter(members)
            if len(filtered) < 3:
                continue
            filtered_groups[(route_label, cardinal)] = filtered

        # Step 5: Compute decision boundary between groups, then
        # build concave hull polygons clipped to each side
        half_planes = _decision_boundary_split(
            filtered_groups, image_width, image_height
        )

        rois: list[ROIPolygon] = []
        color_idx = 0

        for key, points in filtered_groups.items():
            # Pre-clip trail points to this group's side of the boundary
            # so that trail points from boundary vehicles don't bleed over
            boundary = half_planes.get(key)
            if boundary is not None and not boundary.is_empty:
                from shapely.geometry import Point as ShapelyPoint
                points = [p for p in points if boundary.contains(ShapelyPoint(p))]
                if len(points) < 3:
                    continue

            # Build concave hull + buffer for this group
            poly = _concave_hull_polygon(
                points, image_width, image_height,
                ratio=0.5, buffer_px=25.0,
            )
            if poly is None:
                continue

            # Clip to this group's side of the decision boundary
            # Shrink boundary by a margin to ensure a gap between ROIs
            boundary = half_planes.get(key)
            if boundary is not None and not boundary.is_empty:
                margin = boundary.buffer(-10)  # 10px margin from boundary
                if not margin.is_empty:
                    poly = poly.intersection(margin)
                else:
                    poly = poly.intersection(boundary)
                if poly.geom_type == "MultiPolygon":
                    poly = max(poly.geoms, key=lambda g: g.area)
                if poly.is_empty or poly.geom_type != "Polygon":
                    continue

            polygon_coords = _polygon_to_coords(poly)
            if polygon_coords is None:
                continue

            route_label, cardinal = key
            rois.append(ROIPolygon(
                roi_id=str(uuid.uuid4())[:8],
                road_name=route_label,
                direction=cardinal,
                polygon=polygon_coords,
                color=ROI_COLORS[color_idx % len(ROI_COLORS)],
            ))
            color_idx += 1

        if not rois:
            logger.info("Camera %s: no valid ROI polygons from trajectory clustering",
                        self.camera_id)
            return None

        logger.info(
            "Camera %s: generated %d ROIs from %d trajectories (%d windows)",
            self.camera_id, len(rois), self.count, self._windows_fed,
        )

        return CameraROIs(
            camera_id=self.camera_id,
            image_width=image_width,
            image_height=image_height,
            rois=rois,
            generated_at=datetime.now(timezone.utc).isoformat(),
            source="trajectory_cluster",
        )

    def _resolve_azimuth(
        self, roads: list[dict], calibration: dict | None
    ) -> float | None:
        """Get azimuth from calibration, or compute inline from trajectory angles."""
        if calibration and "azimuth_offset_deg" in calibration:
            return calibration["azimuth_offset_deg"]

        # Compute inline — same algorithm as camera_calibration.py
        if len(self._summaries) < 30:
            return None

        angles = np.array([s.angle_deg for s in self._summaries])
        pixel_flow_axis = _circular_mean_axis(angles)

        primary_road = min(roads, key=lambda r: r.get("distance_m", 999))
        road_bearing = primary_road.get("bearing_deg", 0)
        bearing_to_road = primary_road.get("bearing_to_road_deg")

        return estimate_camera_azimuth(
            pixel_flow_axis, road_bearing, bearing_to_road
        )

    def clear(self) -> None:
        self._summaries.clear()
        self._seen_track_ids.clear()
        self._generation_attempted = False
        self._windows_fed = 0


def _iqr_filter(
    members: list[TrajectorySummary], factor: float = 1.5
) -> list[tuple[float, float]]:
    """Remove spatial outliers using IQR on (cx, cy) independently.

    Returns list of (cx, cy) tuples from trail points of members that
    passed the IQR filter. Uses trail_points if available, otherwise
    falls back to mean centroid.
    """
    cx = np.array([s.mean_cx for s in members])
    cy = np.array([s.mean_cy for s in members])

    q1x, q3x = np.percentile(cx, [25, 75])
    iqr_x = q3x - q1x
    mask_x = (cx >= q1x - factor * iqr_x) & (cx <= q3x + factor * iqr_x)

    q1y, q3y = np.percentile(cy, [25, 75])
    iqr_y = q3y - q1y
    mask_y = (cy >= q1y - factor * iqr_y) & (cy <= q3y + factor * iqr_y)

    mask = mask_x & mask_y

    # Collect all trail points from members that passed IQR filter
    result = []
    for i, passed in enumerate(mask):
        if not passed:
            continue
        s = members[i]
        if s.trail_points:
            result.extend(s.trail_points)
        else:
            result.append((s.mean_cx, s.mean_cy))
    return result


def _buffered_union_polygon(
    points: list[tuple[float, float]],
    image_width: int,
    image_height: int,
    buffer_px: float = 30.0,
) -> ShapelyPolygon | None:
    """Create a buffered union of trajectory points as a Shapely Polygon.

    Each point is buffered by buffer_px pixels, then all buffers are unioned
    into a single region that follows the natural road shape.

    Returns a Shapely Polygon, or None if degenerate.
    """
    mp = MultiPoint(points)
    buffered = mp.buffer(buffer_px, quad_segs=8)

    if buffered.is_empty or buffered.area < 1.0:
        return None

    # Simplify to reduce vertex count
    simplified = buffered.simplify(
        tolerance=settings.cluster_simplify_px, preserve_topology=True
    )

    # If MultiPolygon, keep only the largest component
    if simplified.geom_type == "MultiPolygon":
        simplified = max(simplified.geoms, key=lambda g: g.area)

    if simplified.is_empty or simplified.geom_type != "Polygon":
        return None

    return _clip_to_image(simplified, image_width, image_height)


def _concave_hull_polygon(
    points: list[tuple[float, float]],
    image_width: int,
    image_height: int,
    ratio: float = 0.3,
    buffer_px: float = 15.0,
) -> ShapelyPolygon | None:
    """Create a concave hull of trajectory points, then buffer slightly.

    Uses shapely.concave_hull for a tighter fit than convex hull,
    then applies a small buffer to ensure road width coverage.

    Args:
        points: List of (cx, cy) centroid positions.
        ratio: Concavity ratio [0=tightest, 1=convex hull].
        buffer_px: Buffer applied after hull to cover road width.

    Returns a Shapely Polygon, or None if degenerate.
    """
    import shapely

    if len(points) < 3:
        return None

    mp = MultiPoint(points)
    hull = shapely.concave_hull(mp, ratio=ratio)

    if hull.is_empty or hull.geom_type not in ("Polygon", "LineString"):
        return None

    # Buffer the hull to give road width
    buffered = hull.buffer(buffer_px, quad_segs=6)

    if buffered.is_empty or buffered.area < 1.0:
        return None

    # Simplify
    simplified = buffered.simplify(
        tolerance=settings.cluster_simplify_px, preserve_topology=True
    )

    if simplified.geom_type == "MultiPolygon":
        simplified = max(simplified.geoms, key=lambda g: g.area)

    if simplified.is_empty or simplified.geom_type != "Polygon":
        return None

    return _clip_to_image(simplified, image_width, image_height)


def _decision_boundary_split(
    groups: dict[tuple[str, str], list[tuple[float, float]]],
    image_width: int,
    image_height: int,
) -> dict[tuple[str, str], ShapelyPolygon]:
    """Split image space using perpendicular bisector between group centroids.

    For two direction groups, computes the midpoint between their centroids
    and draws a perpendicular bisector line across the image. Each group's
    polygon is clipped to its side of the boundary.

    Returns dict mapping group key -> half-plane polygon.
    """
    from shapely.geometry import LineString, box as shapely_box

    keys = list(groups.keys())
    if len(keys) < 2:
        # Single group: use entire image
        image_rect = shapely_box(0, 0, image_width, image_height)
        return {k: image_rect for k in keys}

    # Compute centroids of each group
    centroids = {}
    for key, pts in groups.items():
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        centroids[key] = (cx, cy)

    # For 2 groups: perpendicular bisector
    if len(keys) == 2:
        c1 = centroids[keys[0]]
        c2 = centroids[keys[1]]

        mid_x = (c1[0] + c2[0]) / 2
        mid_y = (c1[1] + c2[1]) / 2

        # Direction vector from c1 to c2
        dx = c2[0] - c1[0]
        dy = c2[1] - c1[1]

        # Perpendicular direction (rotate 90°)
        perp_dx, perp_dy = -dy, dx
        norm = math.sqrt(perp_dx**2 + perp_dy**2)
        if norm < 0.01:
            image_rect = shapely_box(0, 0, image_width, image_height)
            return {k: image_rect for k in keys}

        perp_dx /= norm
        perp_dy /= norm

        # Extend the bisector line well beyond image bounds
        ext = max(image_width, image_height) * 2
        line_start = (mid_x - perp_dx * ext, mid_y - perp_dy * ext)
        line_end = (mid_x + perp_dx * ext, mid_y + perp_dy * ext)

        # Create a splitting polygon: a wide band on one side of the line
        # Use the normal direction (c1→c2) to determine sides
        offset = ext
        # Side for keys[0] (c1 side): offset in -normal direction
        poly_side_0 = ShapelyPolygon([
            line_start,
            line_end,
            (line_end[0] - dx / norm * offset, line_end[1] - dy / norm * offset),
            (line_start[0] - dx / norm * offset, line_start[1] - dy / norm * offset),
        ])
        # Side for keys[1] (c2 side): offset in +normal direction
        poly_side_1 = ShapelyPolygon([
            line_start,
            line_end,
            (line_end[0] + dx / norm * offset, line_end[1] + dy / norm * offset),
            (line_start[0] + dx / norm * offset, line_start[1] + dy / norm * offset),
        ])

        image_rect = shapely_box(0, 0, image_width, image_height)
        return {
            keys[0]: image_rect.intersection(poly_side_0),
            keys[1]: image_rect.intersection(poly_side_1),
        }

    # For 3+ groups: fall back to full image (no boundary)
    image_rect = shapely_box(0, 0, image_width, image_height)
    return {k: image_rect for k in keys}


def _clip_to_image(
    poly: ShapelyPolygon, image_width: int, image_height: int
) -> ShapelyPolygon | None:
    """Clip polygon to image bounds."""
    from shapely.geometry import box as shapely_box
    image_rect = shapely_box(0, 0, image_width, image_height)
    clipped = poly.intersection(image_rect)

    if clipped.is_empty:
        return None
    if clipped.geom_type == "MultiPolygon":
        clipped = max(clipped.geoms, key=lambda g: g.area)
    if clipped.geom_type != "Polygon":
        return None
    return clipped


def _polygon_to_coords(
    poly: ShapelyPolygon,
) -> list[list[float]] | None:
    """Extract coordinate list from a Shapely Polygon.

    Returns list of [x, y] pairs (without closing duplicate), or None.
    """
    if poly is None or poly.is_empty or poly.geom_type != "Polygon":
        return None

    coords = [
        [round(float(x), 1), round(float(y), 1)]
        for x, y in poly.exterior.coords[:-1]
    ]

    return coords if len(coords) >= 3 else None


class TrajectoryClusterService:
    """Manages per-camera trajectory accumulation and ROI generation."""

    def __init__(self):
        self._accumulators: dict[str, TrajectoryAccumulator] = {}

    def get_or_create_accumulator(self, camera_id: str) -> TrajectoryAccumulator:
        if camera_id not in self._accumulators:
            self._accumulators[camera_id] = TrajectoryAccumulator(camera_id)
        return self._accumulators[camera_id]

    def try_generate(self, camera_id: str) -> CameraROIs | None:
        """Auto-generate ROIs if accumulator is ready. Saves to disk on success."""
        acc = self._accumulators.get(camera_id)
        if acc is None or not acc.is_ready:
            return None
        return self._do_generate(camera_id, acc)

    def force_generate(self, camera_id: str) -> CameraROIs | None:
        """Force ROI generation with lower threshold (for API endpoint)."""
        acc = self._accumulators.get(camera_id)
        if acc is None or acc.count < settings.cluster_min_force:
            return None
        return self._do_generate(camera_id, acc)

    def _do_generate(
        self, camera_id: str, acc: TrajectoryAccumulator
    ) -> CameraROIs | None:
        from backend.services.camera_calibration import calibration_service
        from backend.services.road_geometry import road_geometry_service
        from backend.services.vlm_roi import vlm_roi_service

        roads = road_geometry_service.get_camera_roads(camera_id)
        if not roads:
            logger.warning("No nearby roads for camera %s", camera_id)
            return None

        calibration = calibration_service.get_calibration(camera_id)
        result = acc.generate_rois(roads, calibration)

        if result is not None:
            vlm_roi_service.save_rois(result)
            logger.info(
                "Saved %d trajectory-clustered ROIs for camera %s",
                len(result.rois), camera_id,
            )
            # Clear accumulator after successful generation
            del self._accumulators[camera_id]

        return result

    def invalidate(self, camera_id: str) -> None:
        """Clear accumulator when camera movement is detected."""
        self._accumulators.pop(camera_id, None)
        logger.info("Invalidated trajectory accumulator for camera %s", camera_id)

    def get_status(self, camera_id: str) -> dict:
        """Return accumulation status for diagnostics."""
        acc = self._accumulators.get(camera_id)
        if acc is None:
            return {"camera_id": camera_id, "count": 0, "is_ready": False, "windows": 0}
        return {
            "camera_id": camera_id,
            "count": acc.count,
            "is_ready": acc.is_ready,
            "windows": acc._windows_fed,
            "generation_attempted": acc._generation_attempted,
        }


# Module-level singleton
trajectory_cluster_service = TrajectoryClusterService()
