"""Proxy endpoints for no-CORS MnDOT APIs."""

import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from backend.services.mndot_client import mndot_client
from backend.services.metro_config import metro_config_service
from backend.services.road_geometry import road_geometry_service
from backend.config import settings

router = APIRouter(tags=["proxy"])

# Simple in-memory cache for Mayfly data
_mayfly_cache: dict[str, tuple[float, dict]] = {}


@router.get("/cameras")
async def list_cameras(road: str | None = None):
    """Return cameras, optionally filtered by nearby road (e.g. ?road=I-94).

    Matches both hyphenated (I-94) and spaced (I 94) forms.
    """
    cameras = metro_config_service.get_all_cameras()
    if road:
        # Normalize: "I-94" → match both "I-94" and "I 94"
        variants = {road, road.replace("-", " "), road.replace(" ", "-")}
        cameras = [
            c for c in cameras
            if any(r.get("route_label") in variants
                   for r in road_geometry_service.get_camera_roads(c.id))
        ]
    return cameras


@router.get("/camera/{camera_id}/image")
async def proxy_camera_image(camera_id: str):
    """Proxy camera still image to avoid CORS issues."""
    try:
        image_bytes = await mndot_client.fetch_camera_image(camera_id)
        return Response(content=image_bytes, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch image: {e}")


@router.get("/camera/{camera_id}/detectors")
async def get_camera_detectors(camera_id: str):
    """Return detectors matched to this camera."""
    result = metro_config_service.get_camera_with_detectors(camera_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return result


@router.get("/detector/{detector_id}/{data_type}")
async def proxy_detector_data(detector_id: str, data_type: str = "v30"):
    """Proxy Mayfly detector data with caching."""
    if data_type not in ("v30", "o30", "c30", "s30"):
        raise HTTPException(status_code=400, detail="Invalid data type")

    cache_key = f"{detector_id}:{data_type}"
    now = time.time()

    if cache_key in _mayfly_cache:
        cached_time, cached_data = _mayfly_cache[cache_key]
        if now - cached_time < settings.mayfly_cache_seconds:
            return cached_data

    try:
        data = await mndot_client.fetch_mayfly_data(detector_id, data_type)
        _mayfly_cache[cache_key] = (now, data)
        return data
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch detector data: {e}")


@router.get("/camera/{camera_id}/roads")
async def get_camera_roads(camera_id: str):
    """Return nearby roads from shapefile for this camera."""
    roads = road_geometry_service.get_camera_roads(camera_id)
    if not roads:
        # Try live lookup if not in cache
        cam_det = metro_config_service.get_camera_with_detectors(camera_id)
        if cam_det is None:
            raise HTTPException(status_code=404, detail="Camera not found")
        roads = road_geometry_service.get_nearby_roads(cam_det.camera.lat, cam_det.camera.lon)
    return {"camera_id": camera_id, "roads": roads}
