"""YOLO vehicle detection + occupancy estimation + multi-frame tracking."""

import asyncio
import logging
import math
from collections import defaultdict
from io import BytesIO

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon

from backend.config import settings
from backend.models.schemas import BoundingBox, CVResult, RoadCount

logger = logging.getLogger(__name__)

# Cardinal direction labels by angle range
_CARDINAL_RANGES = [
    (337.5, 360, "NB"), (0, 22.5, "NB"),
    (22.5, 67.5, "NEB"), (67.5, 112.5, "EB"),
    (112.5, 157.5, "SEB"), (157.5, 202.5, "SB"),
    (202.5, 247.5, "SWB"), (247.5, 292.5, "WB"),
    (292.5, 337.5, "NWB"),
]


def _angle_to_cardinal(angle_deg: float) -> str:
    """Convert angle in degrees to cardinal direction."""
    angle_deg = angle_deg % 360
    for lo, hi, card in _CARDINAL_RANGES:
        if lo <= angle_deg < hi:
            return card
    return "NB"


def _match_direction_to_road(pixel_angle: float, roads: list[dict]) -> tuple[str, str] | None:
    """Match a pixel-space movement angle to the closest road's cardinal direction.

    Returns (route_label, cardinal) or None if no match.
    """
    if not roads:
        return None

    best_road = None
    best_diff = 999.0

    for road in roads:
        road_bearing = road.get("bearing_deg", 0)
        # Roads can go either way, so check both bearings
        for bearing in [road_bearing, (road_bearing + 180) % 360]:
            diff = abs(pixel_angle - bearing)
            if diff > 180:
                diff = 360 - diff
            if diff < best_diff:
                best_diff = diff
                cardinal = road.get("cardinal", "")
                # If the movement is closer to the reverse direction, flip cardinal
                if abs(pixel_angle - road_bearing) > 90 and abs(pixel_angle - road_bearing) < 270:
                    # Reverse cardinal
                    flip = {"NB": "SB", "SB": "NB", "EB": "WB", "WB": "EB"}
                    cardinal = flip.get(cardinal, cardinal)
                best_road = (road.get("route_label", ""), cardinal)

    # Only accept if within 45 degrees
    if best_diff <= 45:
        return best_road
    return None


def _build_roi_polygons(rois: list[dict]) -> list[tuple[ShapelyPolygon, dict]]:
    """Pre-build Shapely polygons from ROI dicts for point-in-polygon tests."""
    result = []
    for roi in rois:
        coords = roi.get("polygon", [])
        if len(coords) >= 3:
            try:
                poly = ShapelyPolygon(coords)
                if poly.is_valid:
                    result.append((poly, roi))
            except Exception:
                pass
    return result


def _find_roi_for_point(cx: float, cy: float, roi_polygons: list[tuple[ShapelyPolygon, dict]]) -> dict | None:
    """Find which ROI polygon contains the given point."""
    pt = ShapelyPoint(cx, cy)
    for poly, roi in roi_polygons:
        if poly.contains(pt):
            return roi
    return None


class CVPipeline:
    def __init__(self):
        self.model = None

    def load_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(settings.yolo_model)
            logger.info("YOLO model loaded: %s", settings.yolo_model)
        except Exception:
            logger.exception("Failed to load YOLO model")

    def analyze(self, image_bytes: bytes, camera_id: str) -> CVResult:
        """Run YOLO on an image and return vehicle count + occupancy."""
        if self.model is None:
            return CVResult(camera_id=camera_id, vehicle_count=0, occupancy=0.0, boxes=[])

        img = Image.open(BytesIO(image_bytes))
        img_w, img_h = img.size

        results = self.model(img, conf=settings.yolo_confidence, verbose=False)

        boxes: list[BoundingBox] = []
        total_box_area = 0.0

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in settings.vehicle_classes:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                label = r.names[cls_id]

                boxes.append(BoundingBox(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    label=label, confidence=conf,
                ))
                total_box_area += (x2 - x1) * (y2 - y1)

        # Occupancy: box area / road area (lower 60% of frame)
        road_area = img_w * img_h * 0.6
        occupancy = min(total_box_area / road_area, 1.0) if road_area > 0 else 0.0

        return CVResult(
            camera_id=camera_id,
            vehicle_count=len(boxes),
            occupancy=round(occupancy * 100, 1),
            boxes=boxes,
        )

    def _detect_boxes(self, image_bytes: bytes) -> tuple[list[dict], int, int]:
        """Run YOLO and return raw detection dicts + image dimensions."""
        if self.model is None:
            return [], 0, 0

        img = Image.open(BytesIO(image_bytes))
        img_w, img_h = img.size
        results = self.model(img, conf=settings.yolo_confidence, verbose=False)

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in settings.vehicle_classes:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                label = r.names[cls_id]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                detections.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cx": cx, "cy": cy,
                    "label": label, "confidence": conf,
                })

        return detections, img_w, img_h

    async def analyze_with_tracking(
        self,
        camera_id: str,
        fetch_image_fn,
        roads: list[dict] | None = None,
        rois: list[dict] | None = None,
    ) -> CVResult:
        """Aggregate CV metrics over a 30s window to match MnDOT detector intervals.

        Samples tracking_frames frames at tracking_interval_s intervals (~10 × 3s = 30s).
        Volume = unique tracked vehicles per ROI across all frames.
        Occupancy = average (box area in ROI / ROI area) across all frames.
        Visual overlay = last frame's bounding boxes.
        """
        if self.model is None:
            return CVResult(camera_id=camera_id, vehicle_count=0, occupancy=0.0, boxes=[])

        # Fetch N frames with interval between them
        frames_data = []
        img_w, img_h = 0, 0
        for i in range(settings.tracking_frames):
            try:
                image_bytes = await fetch_image_fn(camera_id)
                detections, iw, ih = self._detect_boxes(image_bytes)
                frames_data.append(detections)
                if iw and ih:
                    img_w, img_h = iw, ih
            except Exception:
                logger.debug("Failed to fetch frame %d for %s", i, camera_id)
                frames_data.append([])

            if i < settings.tracking_frames - 1:
                await asyncio.sleep(settings.tracking_interval_s)

        if not frames_data or not frames_data[-1]:
            return CVResult(camera_id=camera_id, vehicle_count=0, occupancy=0.0, boxes=[])

        # Track vehicles across frames using centroid matching
        tracks = self._track_centroids(frames_data)

        # Pre-build ROI polygons for point-in-polygon tests
        roi_polygons = _build_roi_polygons(rois or [])
        fallback_road_area = img_w * img_h * 0.6 if img_w and img_h else 1.0

        # Pre-compute ROI polygon areas
        roi_area_map: dict[str, float] = {}
        if roi_polygons:
            for poly, roi_dict in roi_polygons:
                rid = roi_dict.get("roi_id", "")
                roi_area_map[rid] = poly.area if poly.area > 0 else fallback_road_area
        total_road_area = sum(roi_area_map.values()) if roi_area_map else fallback_road_area

        # --- Volume: count unique tracked vehicles per ROI across ALL frames ---
        roi_track_seen: dict[str, set[int]] = defaultdict(set)
        if roi_polygons:
            all_track_histories = self._get_all_track_histories(frames_data, tracks)
            for track_id_key, history in all_track_histories.items():
                for _, cx, cy in history:
                    matched = _find_roi_for_point(cx, cy, roi_polygons)
                    if matched:
                        roi_track_seen[matched.get("roi_id", "")].add(track_id_key)
                        break  # count each track only once per ROI

        # --- Occupancy: average per-frame across the entire 30s window ---
        frame_roi_occupancies: dict[str, list[float]] = defaultdict(list)
        frame_total_occupancies: list[float] = []

        for frame_dets in frames_data:
            if not frame_dets:
                continue
            frame_roi_area: dict[str, float] = defaultdict(float)
            frame_total_box_area = 0.0

            for det in frame_dets:
                box_area = (det["x2"] - det["x1"]) * (det["y2"] - det["y1"])
                frame_total_box_area += box_area
                if roi_polygons:
                    matched = _find_roi_for_point(det["cx"], det["cy"], roi_polygons)
                    if matched:
                        frame_roi_area[matched.get("roi_id", "")] += box_area

            # Per-ROI occupancy for this frame
            if roi_polygons:
                for poly, roi_dict in roi_polygons:
                    rid = roi_dict.get("roi_id", "")
                    roi_area = roi_area_map.get(rid, fallback_road_area)
                    occ = min(frame_roi_area.get(rid, 0.0) / roi_area, 1.0) * 100
                    frame_roi_occupancies[rid].append(occ)

            # Total occupancy for this frame
            total_occ = min(frame_total_box_area / total_road_area, 1.0) * 100 if total_road_area > 0 else 0.0
            frame_total_occupancies.append(total_occ)

        avg_total_occupancy = (
            sum(frame_total_occupancies) / len(frame_total_occupancies)
            if frame_total_occupancies else 0.0
        )

        # --- Build visual overlay from last frame's detections ---
        last_frame = frames_data[-1]
        boxes: list[BoundingBox] = []
        road_vehicle_counts: dict[tuple[str, str], dict] = defaultdict(
            lambda: {"types": defaultdict(int)}
        )

        for det_idx, det in enumerate(last_frame):
            track_info = tracks.get(det_idx)
            road_name = None
            road_dir = None
            track_id = None

            if track_info:
                track_id = track_info["track_id"]

            # ROI-based assignment (preferred)
            if roi_polygons:
                matched_roi = _find_roi_for_point(det["cx"], det["cy"], roi_polygons)
                if matched_roi:
                    road_name = matched_roi.get("road_name")
                    road_dir = matched_roi.get("direction")

            # Fallback: bearing-based assignment when no ROIs
            if road_name is None and not roi_polygons and track_info:
                dx = track_info.get("dx", 0)
                dy = track_info.get("dy", 0)
                if abs(dx) > 2 or abs(dy) > 2:
                    pixel_angle = math.degrees(math.atan2(dx, -dy)) % 360
                    match = _match_direction_to_road(pixel_angle, roads or [])
                    if match:
                        road_name, road_dir = match

            if road_name is None and not roi_polygons and roads:
                if len(roads) == 1:
                    road_name = roads[0].get("route_label", "")
                    road_dir = roads[0].get("cardinal", "")

            boxes.append(BoundingBox(
                x1=det["x1"], y1=det["y1"], x2=det["x2"], y2=det["y2"],
                label=det["label"], confidence=det["confidence"],
                road_name=road_name, road_direction=road_dir,
                track_id=track_id,
            ))

            if road_name:
                key = (road_name, road_dir or "")
                road_vehicle_counts[key]["types"][det["label"]] += 1

        # --- Build per-road results with aggregated metrics ---
        road_counts = []
        for (rname, rdir), info in road_vehicle_counts.items():
            roi_id_for_road = None
            if roi_polygons:
                for poly, roi_dict in roi_polygons:
                    if roi_dict.get("road_name") == rname and roi_dict.get("direction") == rdir:
                        roi_id_for_road = roi_dict.get("roi_id", "")
                        break

            # Volume: unique tracks across all frames in 30s window
            volume = (
                len(roi_track_seen[roi_id_for_road])
                if roi_id_for_road and roi_id_for_road in roi_track_seen
                else sum(info["types"].values())
            )

            # Occupancy: averaged across all frames in 30s window
            occ = 0.0
            if roi_id_for_road and roi_id_for_road in frame_roi_occupancies:
                occs = frame_roi_occupancies[roi_id_for_road]
                occ = sum(occs) / len(occs) if occs else 0.0

            road_counts.append(RoadCount(
                road_name=rname,
                direction=rdir,
                vehicle_count=volume,
                occupancy=round(occ, 1),
                by_type=dict(info["types"]),
            ))

        total_volume = sum(rc.vehicle_count for rc in road_counts) if road_counts else len(boxes)

        return CVResult(
            camera_id=camera_id,
            vehicle_count=total_volume,
            occupancy=round(avg_total_occupancy, 1),
            boxes=boxes,
            road_counts=road_counts,
        )

    def _get_all_track_histories(self, frames_data: list[list[dict]],
                                   last_frame_tracks: dict[int, dict]) -> dict[int, list[tuple]]:
        """Get centroid history for ALL tracked vehicles across all frames.

        Returns dict of track_id -> [(frame_idx, cx, cy), ...].
        Used for counting unique vehicles that passed through ROIs.
        """
        # Re-run tracking to get full histories (not just last-frame mappings)
        if len(frames_data) < 2:
            # Single frame — each detection is its own track
            return {
                i: [(0, d["cx"], d["cy"])]
                for i, d in enumerate(frames_data[0]) if frames_data
            }

        next_track_id = 0
        active_tracks: dict[int, dict] = {}

        for frame_idx, dets in enumerate(frames_data):
            if not dets:
                continue
            if not active_tracks:
                for det in dets:
                    active_tracks[next_track_id] = {
                        **det, "history": [(frame_idx, det["cx"], det["cy"])]
                    }
                    next_track_id += 1
                continue

            track_ids = list(active_tracks.keys())
            track_centroids = np.array([[active_tracks[t]["cx"], active_tracks[t]["cy"]] for t in track_ids])
            det_centroids = np.array([[d["cx"], d["cy"]] for d in dets])
            cost = np.linalg.norm(track_centroids[:, None] - det_centroids[None, :], axis=2)
            row_ind, col_ind = linear_sum_assignment(cost)

            matched_det = set()
            new_active: dict[int, dict] = {}
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < 100:
                    tid = track_ids[r]
                    det = dets[c]
                    history = active_tracks[tid]["history"] + [(frame_idx, det["cx"], det["cy"])]
                    new_active[tid] = {**det, "history": history}
                    matched_det.add(c)
            for c, det in enumerate(dets):
                if c not in matched_det:
                    new_active[next_track_id] = {**det, "history": [(frame_idx, det["cx"], det["cy"])]}
                    next_track_id += 1
            active_tracks = new_active

        return {tid: t["history"] for tid, t in active_tracks.items()}

    def _track_centroids(self, frames_data: list[list[dict]]) -> dict[int, dict]:
        """Match detections across frames using centroid distance (Hungarian algorithm).

        Returns dict mapping last-frame detection index -> {track_id, dx, dy}
        where dx, dy is the average velocity in pixels/frame.
        """
        if len(frames_data) < 2:
            return {}

        # Build track assignments frame-by-frame
        # track_history[track_id] = list of (frame_idx, detection) tuples
        next_track_id = 0
        # Current active tracks: track_id -> last detection dict
        active_tracks: dict[int, dict] = {}

        # Initialize tracks from first non-empty frame
        for frame_idx, dets in enumerate(frames_data):
            if dets:
                for det in dets:
                    active_tracks[next_track_id] = {**det, "history": [(frame_idx, det["cx"], det["cy"])]}
                    next_track_id += 1
                start_frame = frame_idx + 1
                break
        else:
            return {}

        # Match subsequent frames
        for frame_idx in range(start_frame, len(frames_data)):
            dets = frames_data[frame_idx]
            if not dets or not active_tracks:
                continue

            track_ids = list(active_tracks.keys())
            track_centroids = np.array([[active_tracks[tid]["cx"], active_tracks[tid]["cy"]] for tid in track_ids])
            det_centroids = np.array([[d["cx"], d["cy"]] for d in dets])

            # Cost matrix: Euclidean distance
            cost = np.linalg.norm(track_centroids[:, None] - det_centroids[None, :], axis=2)

            row_ind, col_ind = linear_sum_assignment(cost)

            matched_det_indices = set()
            new_active: dict[int, dict] = {}

            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < 100:  # Max pixel distance threshold
                    tid = track_ids[r]
                    det = dets[c]
                    history = active_tracks[tid]["history"] + [(frame_idx, det["cx"], det["cy"])]
                    new_active[tid] = {**det, "history": history}
                    matched_det_indices.add(c)

            # Unmatched detections become new tracks
            for c, det in enumerate(dets):
                if c not in matched_det_indices:
                    new_active[next_track_id] = {**det, "history": [(frame_idx, det["cx"], det["cy"])]}
                    next_track_id += 1

            active_tracks = new_active

        # Map last-frame detections to track results
        last_frame = frames_data[-1]
        last_frame_idx = len(frames_data) - 1
        result: dict[int, dict] = {}

        for tid, track in active_tracks.items():
            history = track["history"]
            # Only include tracks visible in last frame
            if history[-1][0] != last_frame_idx:
                continue

            # Find matching detection in last frame by centroid
            last_cx, last_cy = history[-1][1], history[-1][2]
            for det_idx, det in enumerate(last_frame):
                if abs(det["cx"] - last_cx) < 1 and abs(det["cy"] - last_cy) < 1:
                    # Compute velocity
                    if len(history) >= 2:
                        dx = history[-1][1] - history[0][1]
                        dy = history[-1][2] - history[0][2]
                        n_frames = history[-1][0] - history[0][0]
                        if n_frames > 0:
                            dx /= n_frames
                            dy /= n_frames
                    else:
                        dx, dy = 0, 0

                    result[det_idx] = {"track_id": tid, "dx": dx, "dy": dy}
                    break

        return result


cv_pipeline = CVPipeline()
