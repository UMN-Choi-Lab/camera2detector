"""ROI management endpoints: generate, load, save, delete."""

import logging

from fastapi import APIRouter, HTTPException

from backend.models.schemas import CameraROIs, ROIPolygon
from backend.services.vlm_roi import vlm_roi_service
from backend.services.mndot_client import mndot_client
from backend.services.road_geometry import road_geometry_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["roi"])


@router.get("/camera/{camera_id}/rois")
async def get_rois(camera_id: str):
    """Load cached ROIs from JSON file."""
    rois = vlm_roi_service.load_rois(camera_id)
    if rois is None:
        return {"camera_id": camera_id, "rois": []}
    return rois.model_dump()


@router.post("/camera/{camera_id}/rois/generate")
async def generate_rois(camera_id: str):
    """Trigger VLM-based ROI generation for a camera (I-94 only)."""
    # Only generate for cameras near I-94
    nearby_roads = road_geometry_service.get_camera_roads(camera_id)
    if not any(r.get("route_label") in ("I-94", "I 94") for r in nearby_roads):
        raise HTTPException(
            status_code=400,
            detail=f"Camera {camera_id} is not near I-94. VLM ROI generation is currently limited to I-94 cameras.",
        )

    try:
        image_bytes = await mndot_client.fetch_camera_image(camera_id)
    except Exception:
        raise HTTPException(status_code=502, detail="Failed to fetch camera image")

    try:
        camera_rois = await vlm_roi_service.generate_rois(camera_id, image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("VLM ROI generation failed for %s", camera_id)
        raise HTTPException(status_code=500, detail="VLM generation failed")

    return camera_rois.model_dump()


@router.put("/camera/{camera_id}/rois")
async def save_rois(camera_id: str, payload: CameraROIs):
    """Save manually edited ROIs."""
    payload.camera_id = camera_id
    vlm_roi_service.save_rois(payload)
    return payload.model_dump()


@router.delete("/camera/{camera_id}/rois/{roi_id}")
async def delete_roi(camera_id: str, roi_id: str):
    """Delete a single ROI."""
    deleted = vlm_roi_service.delete_roi(camera_id, roi_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="ROI not found")
    return {"status": "deleted", "roi_id": roi_id}
