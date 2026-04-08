"""SSE streaming: push combined CV + detector data every 30s."""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from backend.services.mndot_client import mndot_client
from backend.services.cv_pipeline import cv_pipeline
from backend.services.metro_config import metro_config_service
from backend.services.road_geometry import road_geometry_service
from backend.services.vlm_roi import vlm_roi_service
from backend.services.clickhouse_client import clickhouse_client
from backend.services.stream_manager import stream_manager
from backend.config import settings
import re
from backend.models.schemas import CVResult, DetectorSample, SSEEvent, StationAggregate

logger = logging.getLogger(__name__)

router = APIRouter(tags=["sse"])


def _timestamp_to_30s_index(iso_timestamp: str) -> int:
    """Convert an ISO timestamp to the Mayfly 30-second array index (Central Time)."""
    dt = datetime.fromisoformat(iso_timestamp)
    ct = dt.astimezone(timezone(timedelta(hours=-6)))  # US Central
    return (ct.hour * 3600 + ct.minute * 60 + ct.second) // 30


def _current_30s_index() -> int:
    """Return the index into the Mayfly 30-second data array for current time (Central Time)."""
    ct = datetime.now(timezone(timedelta(hours=-6)))  # US Central (CST)
    return (ct.hour * 3600 + ct.minute * 60 + ct.second) // 30


def _parse_detector_direction(label: str) -> str:
    """Infer direction from MnDOT detector label.

    Labels like '94/PriorAWA1' → W→WB, '94/WPriorAE2' → E→EB.
    Looks for directional letter [NSEW] followed by optional letters/digits at end.
    """
    # Try pattern: direction letter E/W/N/S near end, possibly followed by A/D + lane digit
    m = re.search(r'A([NSEW])[AD]?\d*$', label)
    if m:
        d = m.group(1)
        return {"N": "NB", "S": "SB", "E": "EB", "W": "WB"}.get(d, "")
    # Fallback: last [NSEW] in the label
    matches = re.findall(r'[NSEW]', label.split('/')[-1] if '/' in label else label)
    if matches:
        d = matches[-1]
        return {"N": "NB", "S": "SB", "E": "EB", "W": "WB"}.get(d, "")
    return ""


def _group_detectors_by_station(
    detector_infos: list,
) -> dict[tuple[float, float], list]:
    """Group DetectorInfo objects by station (lat/lon)."""
    stations: dict[tuple[float, float], list] = {}
    for det in detector_infos:
        key = (det.lat, det.lon)
        stations.setdefault(key, []).append(det)
    return stations


async def _fetch_detector_samples(
    detector_ids: list[str],
    target_index: int | None = None,
) -> list[DetectorSample]:
    """Fetch detector data for a specific 30s slot.

    Fetches ALL matched detectors (no limit). If target_index is given,
    look for data at that slot. Otherwise use current time.
    """
    if not detector_ids:
        return []

    samples = []
    idx = target_index if target_index is not None else _current_30s_index()
    mayfly_ok = False

    def _latest_value(arr, idx, lookback=20):
        for i in range(idx, max(idx - lookback, -1), -1):
            if 0 <= i < len(arr) and arr[i] is not None:
                return arr[i]
        return None

    for det_id in detector_ids:
        try:
            counts = await mndot_client.fetch_mayfly_data(det_id, "counts")
            occ_arr = await mndot_client.fetch_mayfly_data(det_id, "occupancy")
            volume = _latest_value(counts, idx)
            occupancy = _latest_value(occ_arr, idx)
            speed = None
            try:
                spd_arr = await mndot_client.fetch_mayfly_data(det_id, "speed")
                speed = _latest_value(spd_arr, idx)
            except Exception:
                pass
            samples.append(DetectorSample(
                detector_id=det_id, volume=volume, occupancy=occupancy, speed=speed,
            ))
            mayfly_ok = True
        except Exception:
            logger.debug("Mayfly failed for detector %s", det_id)

    if mayfly_ok:
        return samples

    logger.debug("Mayfly unavailable, falling back to ClickHouse for %s", detector_ids)
    try:
        latest = await clickhouse_client.get_latest_samples(detector_ids)
    except Exception:
        logger.debug("ClickHouse fallback also failed for %s", detector_ids)
        return []

    return [
        DetectorSample(
            detector_id=det_id,
            volume=latest.get(det_id, {}).get("volume"),
            occupancy=latest.get(det_id, {}).get("occupancy"),
            speed=latest.get(det_id, {}).get("speed"),
        )
        for det_id in detector_ids
    ]


def _aggregate_stations(
    detector_infos: list,
    samples: list[DetectorSample],
) -> list[StationAggregate]:
    """Aggregate per-lane detector samples into per-station (per-direction) totals.

    Volume is summed across lanes. Occupancy and speed are averaged.
    'D' (demand) detectors are excluded to avoid double-counting.
    """
    sample_map = {s.detector_id: s for s in samples}
    stations = _group_detectors_by_station(detector_infos)

    aggregates = []
    for (lat, lon), dets in stations.items():
        # Exclude demand ('D') detectors (lane='0') — they often return null
        lane_dets = [d for d in dets if d.lane != "0"]
        if not lane_dets:
            continue

        # Parse direction from first detector's label
        direction = _parse_detector_direction(lane_dets[0].label)

        # Build station label from common prefix
        station_label = lane_dets[0].label
        # Strip lane suffix for display (e.g., "94/PriorAW1" → "94/PriorAW")
        station_label = re.sub(r'\d+$', '', station_label)

        volumes = []
        occupancies = []
        speeds = []
        det_ids = []

        for det in lane_dets:
            s = sample_map.get(det.id)
            if not s:
                continue
            det_ids.append(det.id)
            if s.volume is not None:
                volumes.append(s.volume)
            if s.occupancy is not None:
                occupancies.append(s.occupancy)
            if s.speed is not None:
                speeds.append(s.speed)

        aggregates.append(StationAggregate(
            station_label=station_label,
            direction=direction,
            volume=sum(volumes) if volumes else None,
            occupancy=sum(occupancies) / len(occupancies) if occupancies else None,
            speed=sum(speeds) / len(speeds) if speeds else None,
            lane_count=len(lane_dets),
            detector_ids=det_ids,
        ))

    return aggregates


# ---------------------------------------------------------------------------
# HLS-based SSE generator (new pipeline)
# ---------------------------------------------------------------------------

async def _generate_events_hls(camera_id: str):
    """Subscribe to a background stream worker and yield results as SSE events."""
    cam_det = metro_config_service.get_camera_with_detectors(camera_id)
    detector_ids = [d.id for d in cam_det.detectors] if cam_det else []
    detector_infos = cam_det.detectors if cam_det else []

    worker = await stream_manager.subscribe(camera_id)
    try:
        while True:
            got_result = await worker.wait_for_result(timeout=35.0)
            if not got_result:
                yield {"event": "keepalive", "data": "{}"}
                continue

            interval = worker.latest_result
            boxes = worker.latest_boxes

            # Build CVResult for backward-compatible visual overlay
            cv_result = CVResult(
                camera_id=camera_id,
                vehicle_count=interval.total_volume if interval else 0,
                occupancy=interval.total_occupancy if interval else 0.0,
                boxes=boxes,
                road_counts=[],
            )

            # Fetch detector data for the same 30s slot the CV window covered
            target_idx = None
            if interval:
                target_idx = _timestamp_to_30s_index(interval.interval_end)
            det_samples = await _fetch_detector_samples(detector_ids, target_index=target_idx)

            # Aggregate per-station (per-direction)
            stations = _aggregate_stations(detector_infos, det_samples)

            event = SSEEvent(
                camera_id=camera_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                cv=cv_result,
                detectors=det_samples,
                stations=stations,
                interval=interval,
            )

            yield {"event": "update", "data": event.model_dump_json()}
    except asyncio.CancelledError:
        pass
    finally:
        await stream_manager.unsubscribe(camera_id)


# ---------------------------------------------------------------------------
# Legacy JPEG-based SSE generator (fallback)
# ---------------------------------------------------------------------------

async def _fetch_image(camera_id: str) -> bytes:
    return await mndot_client.fetch_camera_image(camera_id)


async def _generate_events_legacy(camera_id: str):
    """Legacy generator using JPEG snapshots (when HLS is disabled)."""
    cam_det = metro_config_service.get_camera_with_detectors(camera_id)
    detector_ids = [d.id for d in cam_det.detectors] if cam_det else []
    roads = road_geometry_service.get_camera_roads(camera_id)

    roi_dicts = []
    camera_rois = vlm_roi_service.load_rois(camera_id)
    if camera_rois and camera_rois.rois:
        roi_dicts = [r.model_dump() for r in camera_rois.rois]

    while True:
        try:
            fresh_rois = vlm_roi_service.load_rois(camera_id)
            if fresh_rois and fresh_rois.rois:
                roi_dicts = [r.model_dump() for r in fresh_rois.rois]

            if roads or roi_dicts:
                cv_result = await cv_pipeline.analyze_with_tracking(
                    camera_id, _fetch_image, roads=roads, rois=roi_dicts,
                )
            else:
                image_bytes = await mndot_client.fetch_camera_image(camera_id)
                cv_result = cv_pipeline.analyze(image_bytes, camera_id)

            det_samples = await _fetch_detector_samples(detector_ids)

            event = SSEEvent(
                camera_id=camera_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                cv=cv_result,
                detectors=det_samples,
            )
            yield {"event": "update", "data": event.model_dump_json()}
        except Exception:
            logger.exception("SSE event generation failed for %s", camera_id)
            yield {
                "event": "error",
                "data": json.dumps({"error": "Failed to generate update"}),
            }

        await asyncio.sleep(settings.sse_interval_seconds)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.get("/sse/{camera_id}")
async def sse_stream(camera_id: str):
    """SSE endpoint that streams CV + detector data for a camera."""
    if settings.hls_enabled:
        return EventSourceResponse(_generate_events_hls(camera_id))
    return EventSourceResponse(_generate_events_legacy(camera_id))


# ---------------------------------------------------------------------------
# MJPEG live video stream with tracking overlay
# ---------------------------------------------------------------------------

async def _generate_mjpeg(camera_id: str):
    """Generator that yields annotated JPEG frames for MJPEG streaming."""
    worker = await stream_manager.subscribe(camera_id)
    worker.subscribe_mjpeg()
    last_seq = 0
    try:
        while True:
            new_seq = await worker.wait_for_frame(last_seq, timeout=5.0)
            if new_seq == last_seq:
                continue  # Timeout, no new frame
            last_seq = new_seq
            frame_bytes = worker.latest_frame_jpeg
            if frame_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n"
                    b"\r\n" + frame_bytes + b"\r\n"
                )
    except asyncio.CancelledError:
        pass
    finally:
        worker.unsubscribe_mjpeg()
        await stream_manager.unsubscribe(camera_id)


@router.get("/stream/{camera_id}")
async def mjpeg_stream(camera_id: str):
    """MJPEG endpoint: live video with tracking overlay."""
    from starlette.responses import StreamingResponse

    return StreamingResponse(
        _generate_mjpeg(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# Per-frame tracking data SSE (for HLS + canvas overlay mode)
# ---------------------------------------------------------------------------

async def _generate_tracking_events(camera_id: str):
    """Yield per-frame tracking data (boxes + trails) as SSE events at processing FPS."""
    worker = await stream_manager.subscribe(camera_id)
    last_seq = 0
    try:
        while True:
            new_seq = await worker.wait_for_frame(last_seq, timeout=5.0)
            if new_seq == last_seq:
                continue
            last_seq = new_seq
            data = worker.latest_tracking_data
            if data:
                yield {
                    "event": "tracking",
                    "data": json.dumps(data),
                }
    except asyncio.CancelledError:
        pass
    finally:
        await stream_manager.unsubscribe(camera_id)


@router.get("/tracking/{camera_id}")
async def tracking_stream(camera_id: str):
    """SSE endpoint: per-frame tracking data for canvas overlay on HLS video."""
    return EventSourceResponse(_generate_tracking_events(camera_id))


@router.get("/streams/status")
async def streams_status():
    """Return status of all active HLS stream workers."""
    return stream_manager.get_status()
