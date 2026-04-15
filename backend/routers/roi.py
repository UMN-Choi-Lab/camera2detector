"""ROI management endpoints: generate, load, save, delete, calibrate."""

import logging

from fastapi import APIRouter, HTTPException

from backend.config import settings
from backend.models.schemas import CameraROIs, ROIPolygon
from backend.services.camera_calibration import calibration_service
from backend.services.road_geometry import road_geometry_service
from backend.services.road_projection import generate_projected_rois, save_projected_rois
from backend.services.trajectory_cluster import trajectory_cluster_service
from backend.services.vlm_roi import vlm_roi_service
from backend.services.mndot_client import mndot_client

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


@router.get("/camera/{camera_id}/calibration")
async def get_calibration(camera_id: str):
    """Get current calibration data for a camera."""
    cal = calibration_service.get_calibration(camera_id)
    if cal is None:
        raise HTTPException(status_code=404, detail="No calibration data available")
    return cal


@router.post("/camera/{camera_id}/calibrate")
async def calibrate_camera(camera_id: str):
    """Trigger geometric road projection using current calibration data.

    Requires that the camera has accumulated enough flow data (via streaming)
    for orientation estimation. Returns projected ROI polygons.
    """
    # Check if calibration exists or can be computed
    cal = calibration_service.get_calibration(camera_id)
    roads = road_geometry_service.get_camera_roads(camera_id)

    if cal is None:
        # Try to compute from accumulated flow data
        cal = calibration_service.try_calibrate(camera_id, roads)
        if cal is None:
            raise HTTPException(
                status_code=425,  # Too Early
                detail="Not enough flow data yet. Stream the camera for ~5 minutes first.",
            )

    # Get road geometry in UTM for projection
    roads_utm = road_geometry_service.get_road_coords_utm(camera_id)
    if not roads_utm:
        raise HTTPException(
            status_code=404, detail=f"No nearby roads found for camera {camera_id}"
        )

    # Get camera UTM position
    cam_east, cam_north = _get_camera_utm(camera_id)
    if cam_east is None:
        raise HTTPException(status_code=404, detail="Camera location not found")

    result = generate_projected_rois(
        camera_id=camera_id,
        cam_east=cam_east,
        cam_north=cam_north,
        azimuth_deg=cal["azimuth_offset_deg"],
        roads_utm=roads_utm,
        tilt_deg=cal.get("estimated_tilt_deg"),
    )

    if result is None:
        raise HTTPException(
            status_code=422, detail="No roads could be projected into camera view"
        )

    save_projected_rois(result)
    return result


@router.post("/camera/{camera_id}/rois/cluster")
async def cluster_rois(camera_id: str):
    """Generate ROIs from accumulated vehicle trajectories.

    Requires that the camera has been streaming long enough to accumulate
    trajectory data (~5-10 minutes of traffic).
    """
    status = trajectory_cluster_service.get_status(camera_id)
    min_needed = settings.cluster_min_force

    if status["count"] < min_needed:
        raise HTTPException(
            status_code=425,
            detail=(
                f"Not enough trajectory data yet. Have {status['count']} trajectories, "
                f"need at least {min_needed}. Stream the camera for ~5 minutes first."
            ),
        )

    result = trajectory_cluster_service.force_generate(camera_id)
    if result is None:
        raise HTTPException(
            status_code=422,
            detail="No road-direction groups could form valid ROI polygons.",
        )

    return result.model_dump()


@router.get("/camera/{camera_id}/rois/cluster/status")
async def cluster_status(camera_id: str):
    """Check trajectory accumulation status for a camera."""
    return trajectory_cluster_service.get_status(camera_id)


def _get_camera_utm(camera_id: str) -> tuple[float | None, float | None]:
    """Look up camera location and convert to UTM."""
    from backend.services.metro_config import metro_config_service

    cam = metro_config_service.cameras.get(camera_id)
    if cam and cam.lat and cam.lon:
        e, n = road_geometry_service.latlon_to_utm(cam.lat, cam.lon)
        return e, n
    return None, None
