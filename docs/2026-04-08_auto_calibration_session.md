# Session Summary: Automated Camera Calibration (2026-04-08)

## Problem Statement

MnDOT RTMC operators frequently reposition ~2000 highway cameras. The existing ROI-based vehicle-to-road assignment breaks when cameras move. VLM-based auto ROI generation (GPT-5.3) was attempted previously but completely failed. Need a systematic, geometry-based approach.

## What Was Built

### Phase 0: Camera Movement Detection (`backend/services/camera_watchdog.py`)
- SSIM-based frame comparison against stored reference frames
- Threshold: 0.60 (camera pans drop to 0.2-0.4, normal variation stays >0.7)
- Integrated into `stream_worker.py` at 30s window boundary
- `IntervalResult` now carries `camera_moved` and `ssim_score` fields
- Reference frames stored in `data/reference_frames/`

### Phase 1A: Camera Orientation from Traffic Flow (`backend/services/camera_calibration.py`)
- `FlowAccumulator` collects ByteTrack velocity vectors from trajectory trails (all vehicles, not just ROI-matched)
- Angle-doubling trick for bimodal circular statistics → dominant flow axis
- Matches pixel flow axis to road bearing from shapefile → camera azimuth
- **180° ambiguity fix**: uses bearing-from-camera-to-road to pick correct direction
- **Auto-tilt estimation**: uses median vehicle y-position + road distance to solve for camera tilt angle
- Minimum 50 vehicles required, accumulates from trajectory trails
- Calibration saved to `data/calibrations/{camera_id}.json`

### Phase 1B: Geometric Road Projection (`backend/services/road_projection.py`)
- Simplified pinhole camera model: position (known), azimuth (from 1A), tilt (auto-estimated), height (default 9.1m), FOV (default 60°)
- Projects road polylines from MnDOT shapefile (UTM) into pixel-space ROI polygons
- **Road densification**: shapefile vertices are 100-230m apart; interpolates at 5m intervals within 200m of camera
- **Visibility filtering**: only keeps projected points within image bounds (±50px margin)
- Output: standard `CameraROIs` JSON format — zero changes to downstream pipeline

### Frontend Updates
- **Calibration overlay on camera view**: semi-transparent HUD showing azimuth, tilt, confidence, vehicle count, SSIM
- **"Calibrate" button** in ROI section: triggers flow analysis + road projection
- **Canvas overlay in MJPEG mode**: connected tracking SSE + requestAnimationFrame render loop for ROI/box/trail visualization
- **Red "CAMERA MOVED" badge** with pulse animation when movement detected
- Removed manual ROI editing UI (Edit ROIs, polygon drawing)

### Config & Schema Changes
- `config.py`: `movement_ssim_threshold`, `reference_frame_dir`, `default_camera_height_m/fov_deg/tilt_deg`, `calibration_min_vehicles`, `calibration_data_dir`
- `schemas.py`: `IntervalResult.camera_moved/ssim_score`, `CalibrationResult` model, `CameraROIs.source` extended
- `road_geometry.py`: added `bearing_to_road_deg` to road dicts, `latlon_to_utm()`, `get_road_coords_utm()`

### API Endpoints
- `GET /api/camera/{id}/calibration` — returns calibration data
- `POST /api/camera/{id}/calibrate` — runs calibration + generates projected ROIs

### Tests: 29 passing
- 9 watchdog tests (SSIM, thresholds, multi-camera)
- 12 calibration tests (circular stats, flow accumulator, azimuth estimation)
- 8 projection tests (pinhole model, perspective, polygon generation)

## Bugs Fixed During Session

1. **`metro_config_service.cameras` is a dict, not list** — `_get_camera_utm()` used `.get("id")` on Pydantic objects
2. **180° azimuth ambiguity** — angle-doubling gives axis but wrong direction half the time; fixed by checking bearing to road
3. **Chicken-and-egg: no flow data without ROIs** — flow vectors were extracted from ROI-matched tracks only; fixed to use all trajectory trails
4. **Shapefile vertex spacing too sparse** (100-230m) — added interpolation at 5m intervals within 200m radius
5. **Off-screen points clamped to (0,0)** — created degenerate polygons; fixed to discard instead of clamp
6. **Default tilt=25° way off** — MnDOT cameras are nearly horizontal; added auto-tilt from median vehicle y-position
7. **MJPEG mode had no canvas overlay** — ROIs only rendered in HLS mode; added tracking SSE + render loop to MJPEG
8. **Browser cache** — stale JS served after code changes; added `?v=N` cache busters

## Current Status

- System deployed at `localhost:30000` via Docker
- Camera C843 used as test case (US 12 / I-94 / US 52 interchange)
- Calibration produces azimuth and auto-tilt, projection generates ROIs
- **Projection accuracy still needs evaluation** — auto-tilt just implemented, awaiting user test
- Email drafted to MnDOT (Garrett) requesting camera mount height and PTZ/orientation data

## Known Limitations

- Camera model has 5 unknowns (azimuth, tilt, height, FOV, position) but only azimuth and tilt are estimated; height/FOV are defaults
- All nearby roads project to similar regions (US 12 / I-94 / US 52 share geometry at interchange)
- No elevation separation yet (Phase 3: depth estimation for bridges — deferred)
- Low-traffic cameras may not accumulate 50 vehicles quickly

## Files Created/Modified

### New Files
- `backend/services/camera_watchdog.py`
- `backend/services/camera_calibration.py`
- `backend/services/road_projection.py`
- `tests/__init__.py`
- `tests/test_camera_watchdog.py`
- `tests/test_camera_calibration.py`
- `tests/test_road_projection.py`

### Modified Files
- `backend/services/stream_worker.py` — watchdog + calibration integration
- `backend/services/road_geometry.py` — UTM methods, bearing_to_road_deg
- `backend/routers/roi.py` — calibration endpoints
- `backend/models/schemas.py` — new fields and models
- `backend/config.py` — calibration settings
- `frontend/index.html` — calibration overlay, simplified ROI section
- `frontend/css/style.css` — overlay and badge styles
- `frontend/js/app.js` — calibration UI logic
- `frontend/js/camera_panel.js` — MJPEG canvas overlay

## Next Steps

1. Evaluate auto-tilt projection accuracy on C843
2. Test on multiple cameras (easy: C805, C809; medium: C836, C862)
3. Get camera metadata from MnDOT (height, PTZ params, camera models)
4. Phase 2: Detector correlation optimizer (auto-tune ROI params)
5. Phase 3: Depth estimation for bridge/interchange separation
