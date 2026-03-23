"""SSE streaming: push combined CV + detector data every 30s."""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from backend.services.mndot_client import mndot_client
from backend.services.cv_pipeline import cv_pipeline
from backend.services.metro_config import metro_config_service
from backend.services.road_geometry import road_geometry_service
from backend.services.vlm_roi import vlm_roi_service
from backend.services.clickhouse_client import clickhouse_client
from backend.config import settings
from backend.models.schemas import DetectorSample, SSEEvent

logger = logging.getLogger(__name__)

router = APIRouter(tags=["sse"])


def _current_30s_index() -> int:
    """Return the index into the Mayfly 30-second data array for current time (Central Time)."""
    ct = datetime.now(timezone(timedelta(hours=-6)))  # US Central (CST)
    return (ct.hour * 3600 + ct.minute * 60 + ct.second) // 30


async def _fetch_detector_samples(detector_ids: list[str]) -> list[DetectorSample]:
    """Fetch latest detector data. Tries Mayfly (real-time) first, falls back to ClickHouse."""
    ids = detector_ids[:5]
    if not ids:
        return []

    # Try Mayfly first (real-time)
    samples = []
    idx = _current_30s_index()
    mayfly_ok = False

    def _latest_value(arr, idx, lookback=20):
        """Get the most recent non-None value at or before idx."""
        for i in range(idx, max(idx - lookback, -1), -1):
            if 0 <= i < len(arr) and arr[i] is not None:
                return arr[i]
        return None

    for det_id in ids:
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
                detector_id=det_id,
                volume=volume,
                occupancy=occupancy,
                speed=speed,
            ))
            mayfly_ok = True
        except Exception:
            logger.debug("Mayfly failed for detector %s", det_id)

    if mayfly_ok:
        return samples

    # Fallback: ClickHouse (historical, daily batch)
    logger.debug("Mayfly unavailable, falling back to ClickHouse for %s", ids)
    try:
        latest = await clickhouse_client.get_latest_samples(ids)
    except Exception:
        logger.debug("ClickHouse fallback also failed for %s", ids)
        return []

    return [
        DetectorSample(
            detector_id=det_id,
            volume=latest.get(det_id, {}).get("volume"),
            occupancy=latest.get(det_id, {}).get("occupancy"),
            speed=latest.get(det_id, {}).get("speed"),
        )
        for det_id in ids
    ]


async def _fetch_image(camera_id: str) -> bytes:
    """Wrapper for mndot image fetch (used by tracking pipeline)."""
    return await mndot_client.fetch_camera_image(camera_id)


async def _generate_events(camera_id: str):
    """Generator that yields SSE events every ~30s (frame collection fills the window)."""
    cam_det = metro_config_service.get_camera_with_detectors(camera_id)
    detector_ids = [d.id for d in cam_det.detectors] if cam_det else []

    # Get nearby roads for this camera
    roads = road_geometry_service.get_camera_roads(camera_id)

    # Load ROIs if available
    roi_dicts = []
    camera_rois = vlm_roi_service.load_rois(camera_id)
    if camera_rois and camera_rois.rois:
        roi_dicts = [r.model_dump() for r in camera_rois.rois]

    while True:
        try:
            # Reload ROIs each cycle in case they were regenerated
            fresh_rois = vlm_roi_service.load_rois(camera_id)
            if fresh_rois and fresh_rois.rois:
                roi_dicts = [r.model_dump() for r in fresh_rois.rois]

            # Use tracking pipeline if roads or ROIs are available
            if roads or roi_dicts:
                cv_result = await cv_pipeline.analyze_with_tracking(
                    camera_id, _fetch_image, roads=roads, rois=roi_dicts,
                )
            else:
                image_bytes = await mndot_client.fetch_camera_image(camera_id)
                cv_result = cv_pipeline.analyze(image_bytes, camera_id)

            # Fetch detector data
            det_samples = await _fetch_detector_samples(detector_ids)

            event = SSEEvent(
                camera_id=camera_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                cv=cv_result,
                detectors=det_samples,
            )

            yield {
                "event": "update",
                "data": event.model_dump_json(),
            }
        except Exception:
            logger.exception("SSE event generation failed for %s", camera_id)
            yield {
                "event": "error",
                "data": json.dumps({"error": "Failed to generate update"}),
            }

        await asyncio.sleep(settings.sse_interval_seconds)


@router.get("/sse/{camera_id}")
async def sse_stream(camera_id: str):
    """SSE endpoint that streams CV + detector data for a camera."""
    return EventSourceResponse(_generate_events(camera_id))
