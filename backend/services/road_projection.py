"""Geometric road projection: project road polylines into pixel-space ROI polygons.

Uses a simplified pinhole camera model with estimated parameters
(position, azimuth, height, tilt, FOV) to project world-space road
geometry from the MnDOT shapefile into image coordinates.
"""

import json
import logging
import math
import time
import uuid
from pathlib import Path

import numpy as np

from backend.config import settings

logger = logging.getLogger(__name__)


class CameraModel:
    """Simplified pinhole camera with known position and estimated orientation.

    Coordinate system:
    - World: UTM East-North-Up (EPSG:26915), meters
    - Camera: X=right, Y=down, Z=forward (standard CV convention)
    - Image: origin at top-left, x=right, y=down
    """

    def __init__(
        self,
        cam_east: float,
        cam_north: float,
        azimuth_deg: float,
        height_m: float,
        tilt_deg: float,
        fov_deg: float,
        image_width: int = 720,
        image_height: int = 480,
    ):
        self.cam_east = cam_east
        self.cam_north = cam_north
        self.height_m = height_m
        self.image_width = image_width
        self.image_height = image_height

        # Focal length in pixels from horizontal FOV
        self.fx = (image_width / 2) / math.tan(math.radians(fov_deg / 2))
        # Assume square pixels
        self.fy = self.fx

        # Principal point at image center
        self.cx = image_width / 2.0
        self.cy = image_height / 2.0

        # Build rotation matrix: world (ENU) → camera (X-right, Y-down, Z-forward)
        # Step 1: Rotate around Up axis by -azimuth (align North with camera forward)
        az = math.radians(azimuth_deg)
        # Step 2: Tilt camera downward by tilt_deg
        tilt = math.radians(tilt_deg)

        # Rotation: ENU → camera
        # First, azimuth rotation (around Up/Z axis):
        #   Forward direction in ENU: (sin(az), cos(az), 0) — azimuth from north
        #   Right direction in ENU:   (cos(az), -sin(az), 0)
        #   Up direction:             (0, 0, 1)
        #
        # Then tilt rotation (around camera's X/right axis):
        #   This tilts the forward vector downward

        # Azimuth rotation (world Z-up → camera looking along azimuth)
        R_az = np.array([
            [math.cos(az), -math.sin(az), 0],
            [math.sin(az),  math.cos(az), 0],
            [0,             0,            1],
        ])

        # After azimuth: x=East-rotated, y=North-rotated, z=Up
        # We want camera frame: X=right, Y=down, Z=forward
        # Map: camera_X = world_right (perpendicular to azimuth direction)
        #       camera_Z = world_forward (along azimuth)
        #       camera_Y = world_down

        # Convert from ENU to camera pre-tilt frame:
        # camera_X = East component (right when looking along azimuth)
        # camera_Z = North component (forward along azimuth)
        # camera_Y = -Up (down)
        R_enu_to_cam0 = np.array([
            [ math.cos(az), math.sin(az), 0],  # Right: perpendicular to azimuth
            [ 0,            0,           -1],  # Down: -Up
            [ math.sin(az), -math.cos(az), 0],  # Forward: along azimuth (sin, cos from N)
        ])

        # Wait — let me reconsider. Azimuth is measured clockwise from North.
        # Forward direction in ENU = (sin(az), cos(az), 0) [east, north, up]
        # Right direction in ENU = (cos(az), -sin(az), 0)
        # Down = (0, 0, -1)

        R_enu_to_cam0 = np.array([
            [ math.cos(az), -math.sin(az), 0],  # cam_X = right
            [ 0,             0,           -1],  # cam_Y = down
            [ math.sin(az),  math.cos(az), 0],  # cam_Z = forward
        ])

        # Tilt rotation around camera X axis (positive tilt = look down)
        R_tilt = np.array([
            [1, 0,               0],
            [0, math.cos(tilt), -math.sin(tilt)],
            [0, math.sin(tilt),  math.cos(tilt)],
        ])

        self._R = R_tilt @ R_enu_to_cam0

    def world_to_pixel(
        self, east: float, north: float, up: float = 0.0
    ) -> tuple[float, float] | None:
        """Project a world point (UTM ENU) to pixel coordinates.

        Args:
            east: UTM easting (meters)
            north: UTM northing (meters)
            up: Height above ground (meters), 0 for road surface

        Returns:
            (px_x, px_y) in image coordinates, or None if behind camera
            or outside the image bounds.
        """
        # Vector from camera to point in ENU
        d_east = east - self.cam_east
        d_north = north - self.cam_north
        d_up = up - self.height_m  # Road at ground level (up=0), camera at height_m

        enu = np.array([d_east, d_north, d_up])

        # Transform to camera coordinates
        cam = self._R @ enu

        # cam[2] is depth (Z forward). If behind camera, don't project.
        if cam[2] <= 0.1:
            return None

        # Pinhole projection
        px_x = self.fx * (cam[0] / cam[2]) + self.cx
        px_y = self.fy * (cam[1] / cam[2]) + self.cy

        # Only return if within image bounds (with small margin)
        margin = 50
        if (px_x < -margin or px_x > self.image_width + margin
                or px_y < -margin or px_y > self.image_height + margin):
            return None

        return (px_x, px_y)


def _road_perpendicular(utm_coords: list[list[float]], idx: int) -> tuple[float, float]:
    """Compute unit perpendicular vector to road at a given point index.

    Returns (perp_east, perp_north) pointing to the right of the road direction.
    """
    n = len(utm_coords)
    if n < 2:
        return (1.0, 0.0)

    # Use adjacent points for local direction
    i0 = max(0, idx - 1)
    i1 = min(n - 1, idx + 1)
    de = utm_coords[i1][0] - utm_coords[i0][0]
    dn = utm_coords[i1][1] - utm_coords[i0][1]
    length = math.sqrt(de * de + dn * dn)
    if length < 0.01:
        return (1.0, 0.0)

    # Perpendicular: rotate 90° clockwise (right-hand rule for road direction)
    perp_e = dn / length
    perp_n = -de / length
    return (perp_e, perp_n)


def _densify_and_filter(
    road_utm: list[list[float]], cam_east: float, cam_north: float,
    max_dist_m: float = 200.0, step_m: float = 5.0,
) -> list[list[float]]:
    """Interpolate road polyline at regular intervals and keep nearby segment.

    Shapefile vertices can be 100-200m apart, far too sparse for projection.
    This interpolates at step_m intervals, then keeps only points within
    max_dist_m of the camera.
    """
    # First find the segment of the polyline near the camera
    # by checking which consecutive pair of vertices brackets the camera
    densified = []
    for i in range(len(road_utm) - 1):
        e0, n0 = road_utm[i]
        e1, n1 = road_utm[i + 1]
        seg_len = math.sqrt((e1 - e0) ** 2 + (n1 - n0) ** 2)
        if seg_len < 0.1:
            continue

        # Check if either endpoint or midpoint is near enough to bother
        d0 = math.sqrt((e0 - cam_east) ** 2 + (n0 - cam_north) ** 2)
        d1 = math.sqrt((e1 - cam_east) ** 2 + (n1 - cam_north) ** 2)
        if d0 > max_dist_m + seg_len and d1 > max_dist_m + seg_len:
            continue  # Skip segments far from camera

        # Interpolate this segment
        n_steps = max(1, int(seg_len / step_m))
        for j in range(n_steps + 1):
            t = j / n_steps
            e = e0 + t * (e1 - e0)
            n = n0 + t * (n1 - n0)
            dist = math.sqrt((e - cam_east) ** 2 + (n - cam_north) ** 2)
            if dist <= max_dist_m:
                densified.append([e, n])

    return densified


# Keep old name as alias for tests
_filter_nearby_points = _densify_and_filter


def project_road_to_roi(
    cam: CameraModel,
    road_utm: list[list[float]],
    num_lanes: int = 4,
    lane_width_m: float = 3.7,
) -> list[list[float]] | None:
    """Project a road polyline into a pixel-space ROI polygon.

    Creates a buffered polygon by projecting left and right edges of the road
    (centerline ± half-width) and connecting them. Only uses the portion of
    the road within 150m of the camera.

    Args:
        cam: Camera model with projection parameters.
        road_utm: Road centerline as [[easting, northing], ...] in UTM.
        num_lanes: Number of lanes (determines road width).
        lane_width_m: Width per lane in meters.

    Returns:
        Polygon as [[px_x, px_y], ...] in image coordinates, or None if
        no points project successfully.
    """
    # Densify and filter to nearby segment only
    nearby = _densify_and_filter(road_utm, cam.cam_east, cam.cam_north)
    if len(nearby) < 2:
        return None

    half_width = (num_lanes * lane_width_m) / 2.0

    left_edge = []
    right_edge = []

    for i, (e, n) in enumerate(nearby):
        perp_e, perp_n = _road_perpendicular(nearby, i)

        # Left and right edge points
        left_e = e - perp_e * half_width
        left_n = n - perp_n * half_width
        right_e = e + perp_e * half_width
        right_n = n + perp_n * half_width

        left_px = cam.world_to_pixel(left_e, left_n, 0.0)
        right_px = cam.world_to_pixel(right_e, right_n, 0.0)

        if left_px is not None:
            left_edge.append(left_px)
        if right_px is not None:
            right_edge.append(right_px)

    if len(left_edge) < 2 or len(right_edge) < 2:
        return None

    # Form polygon: left edge forward, right edge reversed (closed loop)
    # Clamp to image bounds (points are already filtered to be near-visible)
    w, h = cam.image_width, cam.image_height
    polygon = []
    for x, y in left_edge:
        polygon.append([round(max(0, min(x, w - 1)), 1),
                        round(max(0, min(y, h - 1)), 1)])
    for x, y in reversed(right_edge):
        polygon.append([round(max(0, min(x, w - 1)), 1),
                        round(max(0, min(y, h - 1)), 1)])

    if len(polygon) < 3:
        return None

    return polygon


# Color palette for road ROIs
_ROI_COLORS = [
    "#a855f7",  # Purple
    "#3b82f6",  # Blue
    "#ef4444",  # Red
    "#22c55e",  # Green
    "#f59e0b",  # Amber
    "#06b6d4",  # Cyan
]


def generate_projected_rois(
    camera_id: str,
    cam_east: float,
    cam_north: float,
    azimuth_deg: float,
    roads_utm: list[dict],
    image_width: int = 720,
    image_height: int = 480,
    height_m: float | None = None,
    tilt_deg: float | None = None,
    fov_deg: float | None = None,
) -> dict | None:
    """Generate ROI polygons for all nearby roads by projecting their geometry.

    Args:
        camera_id: Camera identifier.
        cam_east: Camera easting in UTM (EPSG:26915).
        cam_north: Camera northing in UTM.
        azimuth_deg: Camera azimuth offset (from calibration).
        roads_utm: Roads with 'geometry_utm' field from road_geometry_service.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        height_m: Camera mount height (meters), or default from config.
        tilt_deg: Camera tilt angle (degrees), or default from config.
        fov_deg: Camera FOV (degrees), or default from config.

    Returns:
        CameraROIs-compatible dict, or None if no roads project successfully.
    """
    cam = CameraModel(
        cam_east=cam_east,
        cam_north=cam_north,
        azimuth_deg=azimuth_deg,
        height_m=height_m or settings.default_camera_height_m,
        tilt_deg=tilt_deg or settings.default_camera_tilt_deg,
        fov_deg=fov_deg or settings.default_camera_fov_deg,
        image_width=image_width,
        image_height=image_height,
    )

    rois = []
    for i, road in enumerate(roads_utm):
        utm_coords = road.get("geometry_utm", [])
        if len(utm_coords) < 2:
            continue

        polygon = project_road_to_roi(cam, utm_coords, num_lanes=4, lane_width_m=3.7)
        if polygon is None:
            continue

        route_label = road.get("route_label", "Unknown")
        cardinal = road.get("cardinal", "")
        color = _ROI_COLORS[i % len(_ROI_COLORS)]

        rois.append({
            "roi_id": str(uuid.uuid4())[:8],
            "road_name": route_label,
            "direction": cardinal,
            "polygon": polygon,
            "color": color,
        })

    if not rois:
        return None

    return {
        "camera_id": camera_id,
        "image_width": image_width,
        "image_height": image_height,
        "rois": rois,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": "projected",
    }


def save_projected_rois(rois_dict: dict, data_dir: str | None = None) -> Path:
    """Save projected ROIs to the standard data/rois/ directory."""
    dir_path = Path(data_dir or settings.roi_data_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / f"{rois_dict['camera_id']}.json"
    with open(path, "w") as f:
        json.dump(rois_dict, f, indent=2)
    logger.info("Saved projected ROIs for %s to %s", rois_dict["camera_id"], path)
    return path
