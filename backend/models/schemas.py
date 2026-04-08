"""Pydantic response models."""

from pydantic import BaseModel


class CameraInfo(BaseModel):
    id: str
    name: str
    lat: float
    lon: float


class DetectorInfo(BaseModel):
    id: str
    label: str
    lat: float
    lon: float
    lane: str
    corridor: str
    distance_m: float


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    confidence: float
    road_name: str | None = None
    road_direction: str | None = None
    track_id: int | None = None


class RoadCount(BaseModel):
    road_name: str
    direction: str
    vehicle_count: int
    occupancy: float
    by_type: dict[str, int]


class CVResult(BaseModel):
    camera_id: str
    vehicle_count: int
    occupancy: float
    boxes: list[BoundingBox]
    road_counts: list[RoadCount] = []


class DetectorSample(BaseModel):
    detector_id: str
    volume: float | None = None
    occupancy: float | None = None
    speed: float | None = None


class StationAggregate(BaseModel):
    """Aggregated detector data for one station (one direction)."""
    station_label: str
    direction: str  # EB, WB, NB, SB
    volume: float | None = None  # Sum across lanes
    occupancy: float | None = None  # Average across lanes
    speed: float | None = None  # Average across lanes
    lane_count: int = 0
    detector_ids: list[str] = []


class DetectorEquivalent(BaseModel):
    """One virtual loop detector for one ROI in one 30s interval."""
    roi_id: str
    road_name: str
    direction: str
    volume: int
    occupancy: float
    speed: float | None = None
    by_type: dict[str, int] = {}


class IntervalResult(BaseModel):
    """Complete 30s aggregation result for one camera."""
    camera_id: str
    interval_start: str
    interval_end: str
    frame_count: int
    fps_actual: float
    detectors: list[DetectorEquivalent]
    total_volume: int
    total_occupancy: float


class SSEEvent(BaseModel):
    camera_id: str
    timestamp: str
    cv: CVResult
    detectors: list[DetectorSample]
    stations: list[StationAggregate] = []
    interval: IntervalResult | None = None


class CameraWithDetectors(BaseModel):
    camera: CameraInfo
    detectors: list[DetectorInfo]


class NearbyRoad(BaseModel):
    route_name: str
    route_label: str
    direction: str
    cardinal: str
    bearing_deg: float
    distance_m: float
    geometry_coords: list[list[float]]


class ROIPolygon(BaseModel):
    roi_id: str
    road_name: str
    direction: str
    polygon: list[list[float]]  # [[x, y], ...]
    color: str = "#a855f7"


class CameraROIs(BaseModel):
    camera_id: str
    image_width: int
    image_height: int
    rois: list[ROIPolygon]
    generated_at: str = ""
    source: str = "manual"  # "vlm" or "manual"
