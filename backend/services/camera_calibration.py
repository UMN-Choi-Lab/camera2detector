"""Camera orientation estimation from traffic flow analysis.

Accumulates vehicle velocity vectors from ByteTrack, identifies the
dominant bidirectional flow axis using circular statistics, and matches
it to the nearest road bearing from the shapefile to estimate the
camera's azimuth (rotation from image-up to geographic north).
"""

import json
import logging
import math
import time
from pathlib import Path

import numpy as np

from backend.config import settings

logger = logging.getLogger(__name__)

# Minimum pixel displacement per frame to consider a vehicle "moving"
_MIN_DISPLACEMENT = 3.0


def _circular_mean_axis(angles_deg: np.ndarray) -> float:
    """Find the dominant axis of a bimodal circular distribution.

    Highway traffic produces two modes ~180° apart (opposing directions).
    We use the angle-doubling trick: map each angle θ → 2θ, compute the
    circular mean of the doubled angles, then halve the result.
    This folds opposing directions onto the same axis.

    Args:
        angles_deg: Array of angles in degrees [0, 360).

    Returns:
        Dominant axis angle in degrees [0, 360).
    """
    angles_rad = np.deg2rad(angles_deg)
    # Double angles to fold bimodal → unimodal
    doubled = 2.0 * angles_rad
    # Circular mean of doubled angles
    mean_sin = np.mean(np.sin(doubled))
    mean_cos = np.mean(np.cos(doubled))
    mean_doubled = math.atan2(mean_sin, mean_cos)
    # Halve to get back to original scale
    axis = math.degrees(mean_doubled / 2.0) % 360
    return axis


def estimate_camera_azimuth(
    pixel_flow_axis: float, road_bearing: float,
    bearing_cam_to_road: float | None = None,
) -> float:
    """Compute camera azimuth offset from pixel flow axis and road bearing.

    The azimuth offset is the geographic direction the camera looks toward
    (0° = north, 90° = east). The angle-doubling trick for bimodal flow
    leaves a 180° ambiguity; if bearing_cam_to_road is provided, we pick
    the solution that points the camera toward the road.

    Args:
        pixel_flow_axis: Dominant traffic flow direction in image coordinates
            (0° = image-up, 90° = image-right).
        road_bearing: Geographic bearing of the road (0° = north, 90° = east).
        bearing_cam_to_road: Geographic bearing from camera to the nearest
            point on the road. Used to resolve the 180° ambiguity.

    Returns:
        Azimuth offset in degrees [0, 360).
    """
    candidate1 = (road_bearing - pixel_flow_axis) % 360
    candidate2 = (candidate1 + 180) % 360

    if bearing_cam_to_road is None:
        return candidate1

    # Pick the candidate whose look direction is closer to the road
    def _angle_diff(a: float, b: float) -> float:
        d = abs(a - b) % 360
        return min(d, 360 - d)

    diff1 = _angle_diff(candidate1, bearing_cam_to_road)
    diff2 = _angle_diff(candidate2, bearing_cam_to_road)

    return candidate1 if diff1 <= diff2 else candidate2


class FlowAccumulator:
    """Collects vehicle velocity vectors and centroid positions."""

    def __init__(self, min_vehicles: int = 50):
        self._min_vehicles = min_vehicles
        self._angles: list[float] = []
        self._y_positions: list[float] = []  # Vehicle centroid y-positions

    def add_velocity(self, dx: float, dy: float, cy: float | None = None) -> None:
        """Add a vehicle velocity vector and optional centroid y-position.

        Args:
            dx: Horizontal displacement (positive = rightward).
            dy: Vertical displacement (positive = downward).
            cy: Centroid y-position in pixels (for tilt estimation).
        """
        displacement = math.sqrt(dx * dx + dy * dy)
        if displacement < _MIN_DISPLACEMENT:
            return  # Skip near-stationary detections

        # Convert to angle: 0° = image-up (north in image coords)
        # atan2(dx, -dy): dx is "east" component, -dy is "north" component
        angle = math.degrees(math.atan2(dx, -dy)) % 360
        self._angles.append(angle)

        if cy is not None:
            self._y_positions.append(cy)

    @property
    def count(self) -> int:
        return len(self._angles)

    @property
    def is_ready(self) -> bool:
        return self.count >= self._min_vehicles

    def get_dominant_axis(self) -> float | None:
        """Compute dominant bidirectional flow axis.

        Returns:
            Axis angle in degrees [0, 360), or None if not enough data.
        """
        if not self.is_ready:
            return None
        return _circular_mean_axis(np.array(self._angles))

    def get_median_y(self) -> float | None:
        """Get median y-position of tracked vehicles."""
        if not self._y_positions:
            return None
        return float(np.median(self._y_positions))

    def clear(self) -> None:
        self._angles.clear()
        self._y_positions.clear()


class CameraCalibrationService:
    """Manages per-camera orientation estimation and persistence."""

    def __init__(self, data_dir: str | None = None):
        self._data_dir = Path(data_dir or settings.calibration_data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        # Per-camera flow accumulators (active during calibration)
        self._accumulators: dict[str, FlowAccumulator] = {}
        # Cached calibration results
        self._calibrations: dict[str, dict] = {}

    def get_or_create_accumulator(self, camera_id: str) -> FlowAccumulator:
        if camera_id not in self._accumulators:
            self._accumulators[camera_id] = FlowAccumulator(
                min_vehicles=settings.calibration_min_vehicles
            )
        return self._accumulators[camera_id]

    def try_calibrate(
        self, camera_id: str, roads: list[dict]
    ) -> dict | None:
        """Attempt calibration if enough flow data has been accumulated.

        Args:
            camera_id: Camera identifier.
            roads: Nearby roads from road_geometry_service.get_camera_roads().

        Returns:
            Calibration result dict, or None if not ready / no roads.
        """
        acc = self._accumulators.get(camera_id)
        if acc is None or not acc.is_ready:
            return None

        if not roads:
            logger.warning("No nearby roads for camera %s, cannot calibrate", camera_id)
            return None

        pixel_flow_axis = acc.get_dominant_axis()
        if pixel_flow_axis is None:
            return None

        # Find best-matching road: use the closest road (smallest distance_m).
        primary_road = min(roads, key=lambda r: r.get("distance_m", 999))
        road_bearing = primary_road.get("bearing_deg", 0)
        bearing_to_road = primary_road.get("bearing_to_road_deg")

        azimuth_offset = estimate_camera_azimuth(
            pixel_flow_axis, road_bearing, bearing_to_road
        )

        # Auto-estimate tilt from median vehicle y-position
        # Solves: tan(tilt) = (h - ratio*d) / (ratio*h + d)
        # where ratio = (median_y - cy) / fy, h=camera height, d=road distance
        estimated_tilt = None
        median_y = acc.get_median_y()
        road_distance = primary_road.get("distance_m", 0)
        if median_y is not None and road_distance > 5:
            image_h = 480
            image_w = 720
            fov_h = settings.default_camera_fov_deg
            fy = (image_w / 2) / math.tan(math.radians(fov_h / 2))
            cy = image_h / 2
            h = settings.default_camera_height_m
            d = road_distance

            ratio = (median_y - cy) / fy
            denom = ratio * h + d
            if abs(denom) > 0.1:
                estimated_tilt = math.degrees(
                    math.atan((h - ratio * d) / denom)
                )
                logger.info(
                    "Camera %s auto-tilt: %.1f° (median_y=%.0f, road_dist=%.0f)",
                    camera_id, estimated_tilt, median_y, d,
                )

        # Confidence: based on concentration of the circular distribution
        # Higher concentration → more consistent flow → higher confidence
        angles_rad = np.deg2rad(acc._angles)
        doubled = 2.0 * angles_rad
        R = math.sqrt(
            np.mean(np.cos(doubled)) ** 2 + np.mean(np.sin(doubled)) ** 2
        )
        confidence = min(R, 1.0)  # R ∈ [0, 1], higher = more concentrated

        result = {
            "camera_id": camera_id,
            "azimuth_offset_deg": round(azimuth_offset, 2),
            "estimated_tilt_deg": round(estimated_tilt, 2) if estimated_tilt is not None else None,
            "confidence": round(confidence, 3),
            "n_vehicles": acc.count,
            "primary_road": primary_road.get("route_label", ""),
            "road_bearing_deg": round(road_bearing, 1),
            "road_distance_m": round(road_distance, 1),
            "pixel_flow_axis_deg": round(pixel_flow_axis, 1),
            "median_vehicle_y": round(median_y, 1) if median_y is not None else None,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

        # Save to disk
        path = self._data_dir / f"{camera_id}.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2)

        self._calibrations[camera_id] = result
        # Clear accumulator after successful calibration
        del self._accumulators[camera_id]

        logger.info(
            "Camera %s calibrated: azimuth=%.1f°, confidence=%.2f, "
            "road=%s (bearing=%.1f°), pixel_flow=%.1f°, n=%d",
            camera_id,
            azimuth_offset,
            confidence,
            primary_road.get("route_label", ""),
            road_bearing,
            pixel_flow_axis,
            acc.count,
        )

        return result

    def get_calibration(self, camera_id: str) -> dict | None:
        """Get cached or disk-loaded calibration for a camera."""
        if camera_id in self._calibrations:
            return self._calibrations[camera_id]

        path = self._data_dir / f"{camera_id}.json"
        if path.exists():
            with open(path) as f:
                result = json.load(f)
            self._calibrations[camera_id] = result
            return result

        return None

    def invalidate(self, camera_id: str) -> None:
        """Clear calibration when camera movement is detected."""
        self._calibrations.pop(camera_id, None)
        self._accumulators.pop(camera_id, None)
        logger.info("Invalidated calibration for camera %s", camera_id)


# Module-level singleton
calibration_service = CameraCalibrationService()
