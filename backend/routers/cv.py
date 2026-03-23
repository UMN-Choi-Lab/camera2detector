"""Single-shot CV analysis endpoint."""

from fastapi import APIRouter, HTTPException

from backend.services.mndot_client import mndot_client
from backend.services.cv_pipeline import cv_pipeline
from backend.models.schemas import CVResult

router = APIRouter(tags=["cv"])


@router.get("/cv/{camera_id}", response_model=CVResult)
async def analyze_camera(camera_id: str):
    """Fetch camera image and run YOLO inference."""
    try:
        image_bytes = await mndot_client.fetch_camera_image(camera_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch image: {e}")

    result = cv_pipeline.analyze(image_bytes, camera_id)
    return result
