"""Per-camera HLS stream reader + YOLO tracker + 30s window aggregator."""

import asyncio
import logging
import math
import queue
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

import cv2
import numpy as np
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon

from backend.config import settings
from backend.models.schemas import (
    BoundingBox,
    CVResult,
    DetectorEquivalent,
    IntervalResult,
    RoadCount,
)
from backend.services.camera_calibration import calibration_service
from backend.services.camera_watchdog import CameraWatchdog

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ROI helpers (reused from cv_pipeline, kept local to avoid circular imports)
# ---------------------------------------------------------------------------

def _build_roi_polygons(rois: list[dict]) -> list[tuple[ShapelyPolygon, dict]]:
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


def _find_roi_for_point(
    cx: float, cy: float, roi_polygons: list[tuple[ShapelyPolygon, dict]]
) -> dict | None:
    pt = ShapelyPoint(cx, cy)
    for poly, roi in roi_polygons:
        if poly.contains(pt):
            return roi
    return None


# ---------------------------------------------------------------------------
# Frame accumulator — computes loop-detector-equivalent metrics over 30s
# ---------------------------------------------------------------------------

class FrameAccumulator:
    """Accumulates per-frame tracking data and produces IntervalResult."""

    def __init__(self, camera_id: str, roi_polygons: list[tuple[ShapelyPolygon, dict]]):
        self.camera_id = camera_id
        self.roi_polygons = roi_polygons
        self._frame_count = 0
        self._start_time = time.monotonic()
        self._start_wall = datetime.now(timezone.utc)

        # Per-ROI accumulators
        self._roi_seen_tracks: dict[str, set[int]] = defaultdict(set)
        self._roi_occupied_frames: dict[str, int] = defaultdict(int)
        self._roi_type_counts: dict[str, dict[str, set[int]]] = defaultdict(
            lambda: defaultdict(set)
        )

        # Speed estimation: per-track first/last seen position and time within each ROI
        # roi_id -> track_id -> {"first": (time, cx, cy), "last": (time, cx, cy)}
        self._roi_track_positions: dict[str, dict[int, dict]] = defaultdict(dict)

        # Last frame data for visual overlay
        self.last_frame_detections: list[dict] = []

    def add_frame(self, detections: list[dict]):
        """Add one frame's worth of tracked detections."""
        self._frame_count += 1
        now = time.monotonic()
        self.last_frame_detections = detections

        roi_has_vehicle: dict[str, bool] = {}

        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue

            cx, cy = det["cx"], det["cy"]
            matched = _find_roi_for_point(cx, cy, self.roi_polygons)
            if matched:
                roi_id = matched.get("roi_id", "")
                self._roi_seen_tracks[roi_id].add(track_id)
                roi_has_vehicle[roi_id] = True
                self._roi_type_counts[roi_id][det["label"]].add(track_id)

                # Track position for speed estimation
                positions = self._roi_track_positions[roi_id]
                if track_id not in positions:
                    positions[track_id] = {
                        "first": (now, cx, cy),
                        "last": (now, cx, cy),
                    }
                else:
                    positions[track_id]["last"] = (now, cx, cy)

        for roi_id in roi_has_vehicle:
            self._roi_occupied_frames[roi_id] += 1

    def _compute_roi_speed(self, roi_id: str) -> float | None:
        """Compute average speed (mph) for vehicles in this ROI.

        Uses pixel displacement / time with configurable calibration.
        """
        positions = self._roi_track_positions.get(roi_id, {})
        if not positions:
            return None

        speeds = []
        ppm = settings.speed_calibration_ppm
        for track_id, pos in positions.items():
            t0, x0, y0 = pos["first"]
            t1, x1, y1 = pos["last"]
            dt = t1 - t0
            if dt < 0.3:  # Need at least 0.3s of observation
                continue
            px_dist = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            if px_dist < 5:  # Ignore near-stationary
                continue
            meters_per_sec = (px_dist / ppm) / dt
            mph = meters_per_sec * 2.237
            if 5 < mph < 120:  # Reject outliers
                speeds.append(mph)

        if not speeds:
            return None
        return round(sum(speeds) / len(speeds), 1)

    def finalize(self) -> IntervalResult:
        """Produce the 30s interval result."""
        elapsed = time.monotonic() - self._start_time
        fps_actual = self._frame_count / elapsed if elapsed > 0 else 0.0
        end_wall = datetime.now(timezone.utc)

        detector_eqs: list[DetectorEquivalent] = []
        for _poly, roi_dict in self.roi_polygons:
            roi_id = roi_dict.get("roi_id", "")
            volume = len(self._roi_seen_tracks.get(roi_id, set()))
            occupancy = (
                (self._roi_occupied_frames.get(roi_id, 0) / self._frame_count * 100)
                if self._frame_count > 0
                else 0.0
            )
            by_type = {
                label: len(track_ids)
                for label, track_ids in self._roi_type_counts.get(roi_id, {}).items()
            }
            speed = self._compute_roi_speed(roi_id)
            detector_eqs.append(
                DetectorEquivalent(
                    roi_id=roi_id,
                    road_name=roi_dict.get("road_name", ""),
                    direction=roi_dict.get("direction", ""),
                    volume=volume,
                    occupancy=round(occupancy, 1),
                    speed=speed,
                    by_type=by_type,
                )
            )

        total_volume = sum(d.volume for d in detector_eqs)
        total_occupancy = (
            sum(d.occupancy for d in detector_eqs) / max(len(detector_eqs), 1)
        )

        return IntervalResult(
            camera_id=self.camera_id,
            interval_start=self._start_wall.isoformat(),
            interval_end=end_wall.isoformat(),
            frame_count=self._frame_count,
            fps_actual=round(fps_actual, 1),
            detectors=detector_eqs,
            total_volume=total_volume,
            total_occupancy=round(total_occupancy, 1),
        )


# ---------------------------------------------------------------------------
# Frame reader — thread-safe HLS buffer drain (single-thread cap ownership)
# ---------------------------------------------------------------------------

class FrameReader:
    """Reads HLS frames in a dedicated thread into a jitter buffer (queue).

    Key invariant: ONLY the reader thread calls cap.read() and cap.release().
    HLS delivers frames in bursts (~30 per segment every ~3s). The queue
    absorbs the burst so the worker can consume at a steady rate.
    """

    def __init__(self, url: str, max_queue: int = 120):
        self._url = url
        self._cap: cv2.VideoCapture | None = None
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._should_stop = False
        self._thread: threading.Thread | None = None
        self._connected = False
        self._error: str | None = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        """Read frames from HLS with automatic reconnection."""
        import os
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
            'rw_timeout;5000000|timeout;5000000'  # 5s timeouts (µs)
        )

        while not self._should_stop:
            try:
                self._cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
                if not self._cap.isOpened():
                    self._error = f"Cannot open HLS: {self._url}"
                    time.sleep(2)
                    continue

                self._connected = True
                self._error = None
                read_interval = 1.0 / (self._cap.get(cv2.CAP_PROP_FPS) or 30.0)

                while not self._should_stop:
                    t0 = time.time()
                    ret, frame = self._cap.read()
                    if not ret:
                        break  # Inner loop — reconnect
                    try:
                        self._queue.put_nowait(frame)
                    except queue.Full:
                        try:
                            self._queue.get_nowait()
                        except queue.Empty:
                            pass
                        self._queue.put_nowait(frame)
                    elapsed = time.time() - t0
                    sleep_time = read_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            except Exception as e:
                self._error = str(e)
            finally:
                if self._cap is not None:
                    self._cap.release()
                    self._cap = None
                self._connected = False

            # Quick reconnect — don't wait 30s
            if not self._should_stop:
                logger.warning("HLS reconnecting for %s", self._url)
                time.sleep(1)

    def get_frame(self, timeout: float = 1.0) -> np.ndarray | None:
        """Get the next frame in order (blocks until available or timeout)."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def queued(self) -> int:
        return self._queue.qsize()

    def request_stop(self):
        self._should_stop = True

    def join(self, timeout: float = 3.0):
        if self._thread:
            self._thread.join(timeout=timeout)

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# ---------------------------------------------------------------------------
# Stream worker — one per camera
# ---------------------------------------------------------------------------

class StreamWorker:
    """Reads an HLS stream, runs YOLO tracking, and aggregates 30s intervals."""

    def __init__(self, camera_id: str, gpu_semaphore: asyncio.Semaphore):
        self.camera_id = camera_id
        self._gpu_semaphore = gpu_semaphore
        self._stop_event = asyncio.Event()

        # Published state (read by SSE router)
        self.latest_result: IntervalResult | None = None
        self.latest_boxes: list[BoundingBox] = []
        self.result_ready = asyncio.Event()

        # MJPEG streaming state
        self._mjpeg_subscribers = 0
        self.latest_frame_jpeg: bytes | None = None
        self._frame_seq = 0  # Incremented on each new frame processed
        self._frame_cond = asyncio.Condition()  # Notifies MJPEG consumers instantly

        # Per-frame tracking data for canvas overlay (HLS + overlay mode)
        self.latest_tracking_data: dict | None = None

        # Trajectory trails: track_id -> deque of (cx, cy) centroid positions
        # Also track last-seen frame number per track for cleanup
        self._track_trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=50))
        self._track_last_frame: dict[int, int] = {}
        self._trail_frame_counter = 0

        # Status
        self.connected = False
        self.fps = 0.0
        self.frames_processed = 0
        self.error: str | None = None

        # Camera movement detection
        self._watchdog = CameraWatchdog(
            reference_dir=settings.reference_frame_dir,
            ssim_threshold=settings.movement_ssim_threshold,
        )
        self.roi_invalidated = False

        # Per-camera YOLO model instance (shares GPU weights, owns tracker state)
        self._model = None

    def _load_model(self):
        from ultralytics import YOLO

        self._model = YOLO(settings.yolo_model)
        logger.info("Loaded YOLO model for camera %s", self.camera_id)

    async def run(self):
        """Main loop with reconnection."""
        self._load_model()
        retries = 0

        while not self._stop_event.is_set():
            try:
                await self._process_stream()
                retries = 0
            except Exception as e:
                retries += 1
                self.connected = False
                self.error = str(e)
                if retries >= settings.hls_reconnect_max_retries:
                    logger.error(
                        "HLS failed for %s after %d retries, giving up",
                        self.camera_id,
                        retries,
                    )
                    break
                delay = min(settings.hls_reconnect_delay_s * retries, 30.0)
                logger.warning(
                    "HLS error for %s (attempt %d/%d): %s — retrying in %.0fs",
                    self.camera_id,
                    retries,
                    settings.hls_reconnect_max_retries,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)

    async def _process_stream(self):
        """Open HLS, read frames, track, aggregate."""
        url = settings.hls_stream_url.format(camera_id=self.camera_id)
        logger.info("Opening HLS stream for %s: %s", self.camera_id, url)

        # FrameReader owns the VideoCapture entirely in its own thread
        reader = FrameReader(url)
        reader.start()

        # Wait for first frame (HLS needs time to download manifest + first segment)
        first_frame = None
        for _ in range(200):  # Up to 20s
            first_frame = await asyncio.to_thread(reader.get_frame, 0.1)
            if first_frame is not None:
                break
            if not reader.is_alive:
                raise ConnectionError(reader._error or "FrameReader died during connect")

        if first_frame is None:
            reader.request_stop()
            reader.join()
            raise ConnectionError("No frames received after 20s")

        self.connected = True
        self.error = None
        logger.info("HLS stream connected for %s", self.camera_id)

        # Load ROIs
        from backend.services.vlm_roi import vlm_roi_service

        roi_dicts = []
        camera_rois = vlm_roi_service.load_rois(self.camera_id)
        if camera_rois and camera_rois.rois:
            roi_dicts = [r.model_dump() for r in camera_rois.rois]
        roi_polygons = _build_roi_polygons(roi_dicts)

        accumulator = FrameAccumulator(self.camera_id, roi_polygons)

        # Align to next wall-clock 30s boundary
        now = time.time()
        window_end = (now // settings.aggregation_window_s + 1) * settings.aggregation_window_s

        # Run YOLO on every frame (no skipping — the output rate IS the tracking rate)
        frame_counter = 0
        last_detections: list[dict] = []
        output_interval = 1.0 / settings.hls_target_fps  # 15fps default

        last_frame = None  # Keep last frame for re-serving during queue gaps

        try:
            while not self._stop_event.is_set():
                t0 = time.monotonic()

                # Pop next frame from jitter buffer (short timeout)
                frame = await asyncio.to_thread(reader.get_frame, 0.1)
                if frame is None:
                    if not reader.is_alive:
                        raise ConnectionError(reader._error or "HLS stream disconnected")
                    # Queue empty — re-serve last frame to keep MJPEG alive
                    if last_frame is not None and self._mjpeg_subscribers > 0:
                        self._frame_seq += 1
                        async with self._frame_cond:
                            self._frame_cond.notify_all()
                    await asyncio.sleep(output_interval)
                    continue

                last_frame = frame
                frame_counter += 1
                self.frames_processed += 1

                # Run YOLO tracking on every frame
                async with self._gpu_semaphore:
                    last_detections = await asyncio.to_thread(
                        self._track_frame, frame
                    )

                accumulator.add_frame(last_detections)

                # Update trajectory trails
                self._trail_frame_counter += 1
                for det in last_detections:
                    tid = det.get("track_id")
                    if tid is not None:
                        self._track_trails[tid].append(
                            (int(det["cx"]), int(det["cy"]))
                        )
                        self._track_last_frame[tid] = self._trail_frame_counter
                # Purge stale trails
                stale = [
                    tid
                    for tid, last in self._track_last_frame.items()
                    if self._trail_frame_counter - last > 45
                ]
                for tid in stale:
                    self._track_trails.pop(tid, None)
                    self._track_last_frame.pop(tid, None)

                # Publish per-frame tracking data
                trails_snapshot = {
                    str(tid): list(pts)
                    for tid, pts in self._track_trails.items()
                }
                self.latest_tracking_data = {
                    "detections": [
                        {
                            "x1": d["x1"], "y1": d["y1"],
                            "x2": d["x2"], "y2": d["y2"],
                            "cx": d["cx"], "cy": d["cy"],
                            "label": d["label"],
                            "confidence": d["confidence"],
                            "track_id": d.get("track_id"),
                            "road_name": (
                                _find_roi_for_point(
                                    d["cx"], d["cy"], roi_polygons
                                )
                                or {}
                            ).get("road_name"),
                            "color": (
                                _find_roi_for_point(
                                    d["cx"], d["cy"], roi_polygons
                                )
                                or {}
                            ).get("color"),
                        }
                        for d in last_detections
                    ],
                    "trails": trails_snapshot,
                    "rois": roi_dicts,
                }

                # Annotate EVERY frame for smooth MJPEG (cheap: ~3ms)
                if self._mjpeg_subscribers > 0:
                    annotated = await asyncio.to_thread(
                        self._annotate_frame,
                        frame,
                        last_detections,
                        roi_dicts,
                        roi_polygons,
                        {int(k): v for k, v in trails_snapshot.items()},
                    )
                    self.latest_frame_jpeg = annotated

                self._frame_seq += 1
                async with self._frame_cond:
                    self._frame_cond.notify_all()

                # Check if 30s window is complete
                if time.time() >= window_end:
                    result = accumulator.finalize()

                    # Camera movement detection
                    if last_frame is not None:
                        movement = self._watchdog.check_movement(
                            self.camera_id, last_frame
                        )
                        result.camera_moved = movement["moved"]
                        result.ssim_score = movement["ssim"]
                        if movement["moved"]:
                            self.roi_invalidated = True
                            calibration_service.invalidate(self.camera_id)
                            logger.warning(
                                "Camera %s: movement detected (SSIM=%.3f), ROIs invalidated",
                                self.camera_id,
                                movement["ssim"],
                            )

                    # Feed velocity vectors to calibration flow accumulator
                    # Use trajectory trails (tracks ALL vehicles, not just ROI-matched)
                    flow_acc = calibration_service.get_or_create_accumulator(
                        self.camera_id
                    )
                    if not flow_acc.is_ready:
                        for tid, trail in self._track_trails.items():
                            if len(trail) >= 5:
                                x0, y0 = trail[0]
                                x1, y1 = trail[-1]
                                dx = x1 - x0
                                dy = y1 - y0
                                # Pass last centroid y for tilt estimation
                                flow_acc.add_velocity(dx, dy, cy=y1)

                    self.latest_result = result
                    self.latest_boxes = self._detections_to_boxes(
                        accumulator.last_frame_detections, roi_polygons
                    )
                    self.fps = result.fps_actual

                    self.result_ready.set()
                    self.result_ready.clear()

                    logger.info(
                        "Camera %s interval: %d frames, %.1f fps, vol=%d, occ=%.1f%%",
                        self.camera_id,
                        result.frame_count,
                        result.fps_actual,
                        result.total_volume,
                        result.total_occupancy,
                    )

                    fresh = vlm_roi_service.load_rois(self.camera_id)
                    if fresh and fresh.rois:
                        roi_dicts = [r.model_dump() for r in fresh.rois]
                        roi_polygons = _build_roi_polygons(roi_dicts)

                    accumulator = FrameAccumulator(self.camera_id, roi_polygons)
                    window_end += settings.aggregation_window_s

                # Pace output — only sleep when queue has enough buffer
                # If queue is low, process faster to prevent drain
                if reader.queued > 30:
                    elapsed = time.monotonic() - t0
                    sleep_time = output_interval - elapsed
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
        finally:
            reader.request_stop()  # Signal stop — reader thread releases cap itself
            reader.join(timeout=5.0)
            self.connected = False


    def _track_frame(self, frame: np.ndarray) -> list[dict]:
        """Run model.track() on a single frame (called in thread pool)."""
        results = self._model.track(
            frame,
            conf=settings.yolo_confidence,
            persist=True,
            tracker=settings.yolo_tracker,
            verbose=False,
            classes=settings.vehicle_classes,
        )

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                track_id = int(box.id[0]) if box.id is not None else None
                detections.append(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "cx": (x1 + x2) / 2,
                        "cy": (y1 + y2) / 2,
                        "label": r.names[cls_id],
                        "confidence": float(box.conf[0]),
                        "track_id": track_id,
                    }
                )
        return detections

    @staticmethod
    def _detections_to_boxes(
        detections: list[dict],
        roi_polygons: list[tuple[ShapelyPolygon, dict]],
    ) -> list[BoundingBox]:
        """Convert raw detection dicts to BoundingBox models with ROI assignment."""
        boxes = []
        for det in detections:
            road_name = None
            road_dir = None
            matched = _find_roi_for_point(det["cx"], det["cy"], roi_polygons)
            if matched:
                road_name = matched.get("road_name")
                road_dir = matched.get("direction")
            boxes.append(
                BoundingBox(
                    x1=det["x1"],
                    y1=det["y1"],
                    x2=det["x2"],
                    y2=det["y2"],
                    label=det["label"],
                    confidence=det["confidence"],
                    road_name=road_name,
                    road_direction=road_dir,
                    track_id=det.get("track_id"),
                )
            )
        return boxes

    # --- MJPEG support ---

    def subscribe_mjpeg(self):
        self._mjpeg_subscribers += 1

    def unsubscribe_mjpeg(self):
        self._mjpeg_subscribers = max(0, self._mjpeg_subscribers - 1)

    async def wait_for_frame(self, last_seq: int, timeout: float = 5.0) -> int:
        """Wait for a new annotated frame. Returns new sequence number, or last_seq on timeout."""
        if self._frame_seq > last_seq:
            return self._frame_seq
        try:
            async with self._frame_cond:
                await asyncio.wait_for(
                    self._frame_cond.wait_for(lambda: self._frame_seq > last_seq),
                    timeout=timeout,
                )
        except asyncio.TimeoutError:
            pass
        return self._frame_seq

    @staticmethod
    def _annotate_frame(
        frame: np.ndarray,
        detections: list[dict],
        roi_dicts: list[dict],
        roi_polygons: list[tuple[ShapelyPolygon, dict]],
        track_trails: dict[int, list[tuple[int, int]]] | None = None,
    ) -> bytes:
        """Draw ROIs, trajectories, bounding boxes, and track IDs. Return JPEG bytes."""
        annotated = frame.copy()

        # Draw ROI polygons — single overlay copy for all ROIs
        overlay = annotated.copy()
        for roi in roi_dicts:
            poly_pts = roi.get("polygon", [])
            if len(poly_pts) < 3:
                continue
            color_bgr = _hex_to_bgr(roi.get("color", "#a855f7"))
            pts = np.array(poly_pts, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], color_bgr)
        cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)

        for roi in roi_dicts:
            poly_pts = roi.get("polygon", [])
            if len(poly_pts) < 3:
                continue
            color_bgr = _hex_to_bgr(roi.get("color", "#a855f7"))
            pts = np.array(poly_pts, dtype=np.int32)
            cv2.polylines(annotated, [pts], isClosed=True, color=color_bgr, thickness=2)
            cx = int(sum(p[0] for p in poly_pts) / len(poly_pts))
            cy = int(sum(p[1] for p in poly_pts) / len(poly_pts))
            label = f"{roi.get('road_name', '')} {roi.get('direction', '')}"
            _draw_label(annotated, label, cx, cy, color_bgr, center=True)

        # Build track_id -> color map from current detections
        track_colors: dict[int, tuple] = {}
        for det in detections:
            tid = det.get("track_id")
            if tid is None:
                continue
            matched = _find_roi_for_point(det["cx"], det["cy"], roi_polygons)
            if matched:
                track_colors[tid] = _hex_to_bgr(matched.get("color", "#4ecca3"))
            else:
                track_colors[tid] = (163, 204, 78)

        # Draw trajectory trails with fading opacity
        if track_trails:
            _draw_trails(annotated, track_trails, track_colors, roi_polygons)

        # Draw bounding boxes with track IDs
        for det in detections:
            x1, y1 = int(det["x1"]), int(det["y1"])
            x2, y2 = int(det["x2"]), int(det["y2"])
            track_id = det.get("track_id")

            color_bgr = track_colors.get(track_id, (163, 204, 78))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)

            # Label: class + track ID
            parts = [det["label"]]
            if track_id is not None:
                parts.append(f"#{track_id}")
            label = " ".join(parts)
            _draw_label(annotated, label, x1, y1 - 4, color_bgr)

        # Encode to JPEG
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 65])
        return buf.tobytes()

    async def stop(self):
        """Signal the worker to stop."""
        self._stop_event.set()

    async def wait_for_result(self, timeout: float = 35.0) -> bool:
        """Wait for the next interval result. Returns True if result available."""
        try:
            await asyncio.wait_for(self.result_ready.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """Convert '#rrggbb' to (B, G, R) for OpenCV."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return (163, 204, 78)
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


def _draw_trails(
    img: np.ndarray,
    track_trails: dict[int, list[tuple[int, int]]],
    track_colors: dict[int, tuple],
    roi_polygons: list[tuple[ShapelyPolygon, dict]],
):
    """Draw trajectory trails directly on the image (no alpha copy for speed)."""
    for tid, points in track_trails.items():
        if len(points) < 2:
            continue

        color = track_colors.get(tid)
        if color is None:
            lx, ly = points[-1]
            matched = _find_roi_for_point(lx, ly, roi_polygons)
            color = _hex_to_bgr((matched or {}).get("color", "#4ecca3"))

        # Draw polyline in one call (much faster than per-segment)
        pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)

        # Dot at trail head
        cv2.circle(img, points[-1], 4, color, -1)


def _draw_label(
    img: np.ndarray, text: str, x: int, y: int, color: tuple, center: bool = False
):
    """Draw a text label with background on an image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    if center:
        x -= tw // 2
        y += th // 2
    # Background
    cv2.rectangle(img, (x - 1, y - th - 2), (x + tw + 1, y + 2), color, -1)
    # Text (black on colored bg)
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
