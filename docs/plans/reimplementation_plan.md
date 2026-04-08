# REIMPLEMENTATION PLAN: Loop-Detector-Equivalent CV Pipeline

**Date**: 2026-04-07
**Test camera**: C843 (I-94 EB/WB, 2 ROIs, 720x480)
**Goal**: Produce 30-second volume, occupancy, and speed from HLS video streams — directly comparable to MnDOT loop detector output.

---

## Problem

The current CV pipeline fetches static JPEG snapshots every 3s (10 frames over 30s), then applies manual centroid tracking via Hungarian algorithm. This produces:

- **~27s of sleep** per cycle just waiting between frames
- **Unreliable tracking**: at 0.33 fps, vehicles move ~90m between frames at highway speed — centroid matching breaks down
- **Spatial occupancy** (box area / road area) instead of **temporal occupancy** (% of time sensor is occupied)
- **No speed estimation**
- **Duplicate tracking computation** (`_track_centroids` + `_get_all_track_histories` both redo Hungarian matching)

The result is not comparable to loop detector data.

## Target Output

Match what MnDOT loop detectors produce per 30-second interval per station:

| Metric | Loop Detector | CV Equivalent |
|--------|--------------|---------------|
| **Volume** | Magnetic pulse count | Unique track IDs crossing ROI |
| **Occupancy** | % time detector is actuated (0–100) | % of frames where ROI contains a vehicle |
| **Speed** | Length / actuation duration | Pixel displacement / time × calibration factor |

## Architecture

```
                   ┌──────────────────────────────────────────┐
                   │          StreamManager (singleton)        │
                   │  - subscribe(camera_id) → start worker   │
                   │  - unsubscribe(camera_id) → ref count    │
                   │  - gpu_semaphore(max_concurrent=4)       │
                   └──────┬───────────────────────┬───────────┘
                          │                       │
              ┌───────────▼──────────┐ ┌──────────▼───────────┐
              │  StreamWorker(C843)  │ │  StreamWorker(C862)  │  ...
              │                      │ │                      │
              │  HLS → cv2.read()   │ │  HLS → cv2.read()   │
              │  model.track()      │ │  model.track()      │
              │  FrameAccumulator   │ │  FrameAccumulator   │
              │     ↓ every 30s     │ │     ↓ every 30s     │
              │  IntervalResult     │ │  IntervalResult     │
              └───────────┬─────────┘ └──────────┬───────────┘
                          │                       │
              ┌───────────▼───────────────────────▼───────────┐
              │              SSE Router                        │
              │  subscribe → wait for result → yield event    │
              │  (multiple clients share one worker)          │
              └───────────────────────────────────────────────┘
```

### Key design decisions

1. **One YOLO model instance per camera** — `model.track(persist=True)` stores ByteTrack state on the model object. Separate instances prevent cross-camera tracker corruption. PyTorch shares GPU weight memory across instances.

2. **Background workers, not inline processing** — SSE endpoint subscribes to a worker and waits for results. No CV processing on the SSE path.

3. **Wall-clock-aligned 30s windows** — Align to :00 and :30 second boundaries to match MnDOT detector intervals exactly.

4. **Temporal occupancy** — Count frames where ROI has >= 1 vehicle. `occupied_frames / total_frames × 100`. At 5 FPS, each frame = 200ms time slice.

## Implementation Phases

### Phase 1: Core pipeline (stream → detect → track → aggregate)

#### 1.1 Add dependencies

**`requirements.txt`** — add:
```
opencv-python-headless>=4.9.0
```
(ultralytics bundles opencv, but explicit is better for the Docker build)

#### 1.2 New config settings

**`backend/config.py`** — add to `Settings`:

```python
# HLS streaming
hls_enabled: bool = True
hls_target_fps: float = 5.0            # Downsample from native ~15-30 FPS
hls_reconnect_delay_s: float = 5.0
hls_reconnect_max_retries: int = 10
aggregation_window_s: float = 30.0     # Match MnDOT detector interval

# YOLO tracking
yolo_tracker: str = "bytetrack.yaml"

# GPU management
max_concurrent_streams: int = 4
```

**Why 5 FPS**: YOLOv8n on RTX 6000 Ada handles ~200+ FPS at 720x480. At 5 FPS × 4 cameras = 20 FPS total, well within budget. 5 FPS gives 150 frames per 30s window — far better than current 10 frames. Highway vehicles at 60 mph move ~5.4m between frames (vs ~90m at 0.33 FPS), making ByteTrack reliable.

#### 1.3 New schemas

**`backend/models/schemas.py`** — add:

```python
class DetectorEquivalent(BaseModel):
    """One virtual loop detector for one ROI in one 30s interval."""
    roi_id: str
    road_name: str
    direction: str
    volume: int                       # Unique tracked vehicles through ROI
    occupancy: float                  # Time-based, 0–100%
    speed: float | None = None        # mph estimate (Phase 4)
    by_type: dict[str, int] = {}      # {car: 12, truck: 3}

class IntervalResult(BaseModel):
    """Complete 30s aggregation result for one camera."""
    camera_id: str
    interval_start: str               # ISO timestamp (wall-clock aligned)
    interval_end: str
    frame_count: int
    fps_actual: float
    detectors: list[DetectorEquivalent]
    total_volume: int
    total_occupancy: float
```

Extend `SSEEvent`:
```python
class SSEEvent(BaseModel):
    ...
    interval: IntervalResult | None = None   # NEW
```

#### 1.4 Stream worker

**New file: `backend/services/stream_worker.py`**

Core loop per camera:

```
open HLS stream via cv2.VideoCapture(url, CAP_FFMPEG)
create YOLO model instance (shares GPU weights)
create FrameAccumulator(rois)
loop:
    read frame (in thread pool — releases GIL during ffmpeg decode)
    skip if below target FPS interval
    run model.track(frame, persist=True) in thread pool (GPU semaphore)
    accumulator.add_frame(detections, timestamp)
    if wall-clock crosses 30s boundary:
        result = accumulator.finalize()
        publish result (asyncio.Event)
        reset accumulator
```

**Critical: `model.track(persist=True)`**
- ByteTrack maintains Kalman filter state across calls
- Returns `box.id` — persistent integer track ID per vehicle
- No manual Hungarian matching needed
- Handles occlusion, re-identification automatically

**Frame accumulator computes per ROI**:
- **Volume**: `len(seen_track_ids[roi_id])` — set of unique track IDs whose centroid fell inside the ROI polygon
- **Occupancy**: `occupied_frames[roi_id] / total_frames × 100` — binary per frame (any vehicle in ROI = occupied)
- **by_type**: count unique track IDs per vehicle class

#### 1.5 Stream manager

**New file: `backend/services/stream_manager.py`**

```python
class StreamManager:
    subscribe(camera_id)    # Start worker if not running, increment ref count
    unsubscribe(camera_id)  # Decrement ref count, stop worker if zero
    get_status()            # For /api/streams/status endpoint
    shutdown()              # Stop all workers (app lifespan)
```

- `asyncio.Semaphore(max_concurrent_streams)` gates GPU access
- Workers auto-stop when last SSE client disconnects

### Phase 2: Integration

#### 2.1 Rewrite SSE router

**`backend/routers/sse.py`** — replace inline CV processing:

```python
async def _generate_events(camera_id):
    worker = await stream_manager.subscribe(camera_id)
    try:
        while True:
            await worker.wait_for_result(timeout=35)  # asyncio.Event
            interval = worker.latest_result
            boxes = worker.latest_boxes          # Last frame's boxes for overlay
            det_samples = await _fetch_detector_samples(detector_ids)
            yield SSEEvent(cv=..., detectors=det_samples, interval=interval)
    finally:
        await stream_manager.unsubscribe(camera_id)
```

#### 2.2 App lifespan

**`backend/app.py`** — add shutdown:

```python
from backend.services.stream_manager import stream_manager

async def lifespan(app):
    ...existing startup...
    yield
    await stream_manager.shutdown()  # Stop all workers
    await mndot_client.close()
```

#### 2.3 Status endpoint

Add `GET /api/streams/status` — returns active workers, FPS, frame counts, last interval.

### Phase 3: Frontend updates

- Display `IntervalResult` detector-equivalent metrics alongside MnDOT detector data
- Show volume/occupancy/speed per ROI in the road breakdown panel
- Chart both CV-derived and detector-derived time series on the same axes

### Phase 4: Speed estimation (stretch)

- Track pixel displacement across frames for each vehicle
- Use ROI geometry (known lane width ~3.7m) to calibrate pixel→meter conversion
- Speed = distance traveled / time in ROI
- Validate against MnDOT speed data from same corridor

## HLS Stream Resilience

```
open stream
  ├─ success → read loop → on disconnect → wait 5s → reopen
  ├─ fail → retry with exponential backoff (5s, 10s, 15s... cap 30s)
  └─ 10 consecutive failures → fall back to JPEG snapshot mode (existing pipeline)
```

MnDOT streams can go offline for maintenance. Fallback ensures continuous data collection.

## Occupancy: Loop Detector vs CV

Physical loop detector (6 ft wire in pavement, 60 mph vehicle ≈ 88 ft/s):
```
time_on = vehicle_length / speed ≈ 15ft / 88fps ≈ 0.17s per vehicle
20 vehicles in 30s → occupancy = 20 × 0.17 / 30 = 11.3%
```

CV equivalent (5 FPS, ROI spans ~20ft of road):
```
frames_occupied = vehicles × (ROI_length / speed × FPS) ≈ 20 × (20/88 × 5) ≈ 23 frames
150 total frames → occupancy = 23 / 150 = 15.3%
```

CV occupancy will be **systematically higher** than loop detector occupancy because:
1. ROI is wider than a loop detector (~20ft vs 6ft)
2. Bounding boxes extend beyond vehicle edges

This is a known calibration offset — not a bug. The correlation should be linear and can be corrected with a scaling factor derived from C843 comparison data.

## GPU Budget

| Resource | Value |
|----------|-------|
| GPU | RTX 6000 Ada (48 GB VRAM) |
| YOLOv8n model | ~6 MB VRAM |
| Per-frame working memory | ~50 MB |
| ByteTrack state per camera | negligible (CPU-side Kalman filters) |
| 4 cameras × 5 FPS | 20 inferences/s → ~10% GPU utilization |
| Headroom | Can scale to 20+ concurrent cameras |

## File Changes

| File | Action | What |
|------|--------|------|
| `requirements.txt` | Modify | Add `opencv-python-headless` |
| `backend/config.py` | Modify | Add HLS/tracking/GPU settings |
| `backend/models/schemas.py` | Modify | Add `DetectorEquivalent`, `IntervalResult`; extend `SSEEvent` |
| `backend/services/stream_worker.py` | **New** | HLS reader + YOLO tracker + 30s aggregator |
| `backend/services/stream_manager.py` | **New** | Worker lifecycle + GPU semaphore |
| `backend/routers/sse.py` | Modify | Subscribe to stream_manager instead of inline CV |
| `backend/app.py` | Modify | Add stream_manager shutdown |
| `backend/services/cv_pipeline.py` | Keep | Preserved for single-shot `/api/cv` endpoint and fallback |
| `frontend/js/charts.js` | Modify | Chart IntervalResult data |
| `frontend/js/camera_panel.js` | Modify | Display detector-equivalent metrics |

## Test Plan (C843)

1. **HLS connectivity**: Verify `cv2.VideoCapture` opens C843's stream and reads frames
2. **Tracking sanity**: Run `model.track()` on 150 frames (30s), verify persistent track IDs
3. **Volume accuracy**: Compare unique track count per ROI against manual video count
4. **Occupancy correlation**: Plot CV occupancy vs MnDOT detector occupancy for same 30s intervals over 1 hour
5. **Reconnection**: Kill/restart stream mid-processing, verify auto-reconnect
6. **Multi-camera**: Run 4 cameras simultaneously, verify GPU stays under budget

## Implementation Order

```
Phase 1.2  config.py              (10 min)
Phase 1.3  schemas.py             (10 min)
Phase 1.4  stream_worker.py       (core — most work)
Phase 1.5  stream_manager.py
Phase 2.1  sse.py rewrite
Phase 2.2  app.py lifespan
─── Test with C843 at this point ───
Phase 2.3  status endpoint
Phase 3    frontend updates
Phase 4    speed estimation
```
