"""FastAPI application factory."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.services.metro_config import metro_config_service
from backend.services.mndot_client import mndot_client
from backend.services.cv_pipeline import cv_pipeline
from backend.services.road_geometry import road_geometry_service
from backend.services.stream_manager import stream_manager
from backend.routers import proxy, cv, sse, history, roi, clearguide
from backend.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init HTTP client, load metro config, load YOLO model, load road geometry."""
    # Ensure ROI data directory exists
    Path(settings.roi_data_dir).mkdir(parents=True, exist_ok=True)

    await mndot_client.start()
    await metro_config_service.load()
    cv_pipeline.load_model()

    # Load road geometry from shapefile
    try:
        road_geometry_service.load()
        # Precompute camera->roads mapping
        cameras = metro_config_service.get_all_cameras()
        if cameras:
            cam_dicts = [{"id": c.id, "lat": c.lat, "lon": c.lon} for c in cameras]
            road_geometry_service.precompute_camera_roads(cam_dicts)
    except Exception:
        logger.exception("Failed to load road geometry (non-fatal)")

    yield
    await stream_manager.shutdown()
    await mndot_client.close()


app = FastAPI(title="Camera2Detector", lifespan=lifespan)

app.include_router(proxy.router, prefix="/api")
app.include_router(cv.router, prefix="/api")
app.include_router(sse.router, prefix="/api")
app.include_router(history.router, prefix="/api")
app.include_router(roi.router, prefix="/api")
app.include_router(clearguide.router, prefix="/api")

@app.get("/roi-tool", include_in_schema=False)
async def roi_tool():
    return FileResponse("frontend/roi_tool.html")

app.mount("/roi-tests", StaticFiles(directory="data/roi_tests"), name="roi-tests")
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
