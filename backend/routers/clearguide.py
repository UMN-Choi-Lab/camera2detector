"""ClearGuide speed data comparison endpoints."""

import asyncio
import logging
import time

from fastapi import APIRouter, HTTPException

from backend.services.clearguide_client import clearguide_client
from backend.services.metro_config import metro_config_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["clearguide"])


@router.get("/camera/{camera_id}/clearguide/speed")
async def get_clearguide_speed(
    camera_id: str,
    hours: float = 1.0,
    link_id: int | None = None,
):
    """Get ClearGuide speed data near a camera for comparison with MnDOT."""
    if not clearguide_client.enabled:
        raise HTTPException(status_code=503, detail="ClearGuide not configured")

    # Get camera location
    cameras = metro_config_service.get_all_cameras()
    cam = next((c for c in cameras if c.id == camera_id), None)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Find ClearGuide link
    link_title = ""
    if link_id is None:
        link_info = await asyncio.to_thread(
            clearguide_client.find_link_near, cam.lat, cam.lon
        )
        if not link_info:
            return {
                "camera_id": camera_id,
                "link_id": None,
                "data": [],
                "error": "No ClearGuide link found nearby",
            }
        link_id = link_info["link_id"]
        link_title = link_info.get("link_title", "")

    # Fetch speed timeseries
    end_ts = int(time.time())
    start_ts = end_ts - int(hours * 3600)

    try:
        result = await asyncio.to_thread(
            clearguide_client.get_speed_timeseries, link_id, start_ts, end_ts
        )
    except Exception:
        logger.exception("ClearGuide speed fetch failed for link %s", link_id)
        raise HTTPException(status_code=502, detail="ClearGuide API error")

    # Parse timeseries response
    speed_data = _parse_speed_response(result)

    return {
        "camera_id": camera_id,
        "link_id": link_id,
        "link_title": link_title,
        "data": speed_data,
    }


def _parse_speed_response(result: dict) -> list[dict]:
    """Parse ClearGuide timeseries response into [{ts, speed}].

    ClearGuide response structure:
    { "series": { "all": { "avg_speed": { "data": [[ts, value], ...] } } } }
    """
    speed_data = []

    # Primary path: series.all.avg_speed.data
    series = result.get("series", {})
    if series:
        all_data = series.get("all", {})
        avg_speed = all_data.get("avg_speed", {})
        data_points = avg_speed.get("data", [])
        for item in data_points:
            if isinstance(item, list) and len(item) >= 2 and item[1] is not None:
                speed_data.append({"ts": item[0], "speed": item[1]})
        if speed_data:
            return speed_data

    # Fallback: flat results
    results = result.get("results", result)
    if isinstance(results, dict):
        ts_data = results.get("avg_speed", {})
        if isinstance(ts_data, list):
            for item in ts_data:
                if isinstance(item, list) and len(item) >= 2 and item[1] is not None:
                    speed_data.append({"ts": item[0], "speed": item[1]})
    elif isinstance(results, list):
        for item in results:
            if isinstance(item, dict):
                spd = item.get("avg_speed", item.get("speed"))
                if spd is not None:
                    speed_data.append({
                        "ts": item.get("timestamp", item.get("ts", "")),
                        "speed": spd,
                    })

    return speed_data
