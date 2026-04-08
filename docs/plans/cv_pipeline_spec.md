# CV Pipeline Specification

**Date**: 2026-04-08
**Author**: Derived from iterative prototyping session

---

## Goal

Produce loop-detector-equivalent traffic metrics (volume, occupancy, speed) from MnDOT highway camera video streams using computer vision, and display them alongside real MnDOT detector data for validation.

## Functional Requirements

### F1: Video Processing Pipeline

- **Input**: MnDOT HLS video stream (`playlist.m3u8`) per camera
- **Processing**: YOLO object detection + ByteTrack multi-object tracking (`model.track(persist=True)`)
- **Output per 30s window** (aligned to wall-clock :00/:30 boundaries):
  - **Volume**: Count of unique tracked vehicles (by track ID) passing through each ROI
  - **Occupancy**: Fraction of frames where at least one vehicle is present in ROI (temporal, not spatial)
  - **Speed**: Average vehicle speed estimated from pixel displacement / time with configurable calibration (pixels-per-meter)
  - **By type**: Volume breakdown by vehicle class (car, truck, bus, motorcycle)
  - **Per direction**: Separate metrics for each ROI (e.g., I-94 EB, I-94 WB)

### F2: Real-Time Visualization

- **Video + annotations must be perfectly synchronized** — every displayed frame must show bounding boxes and trails that match that exact frame
- **Video must play at real-time speed** — no progressive lag, no slow-motion, no drift
- **Annotations on each frame**:
  - ROI polygons (colored by road/direction)
  - Bounding boxes with track IDs (e.g., `car #42`)
  - Trajectory trails: fading polylines showing recent centroid history per track, color-coded by ROI
- **Frame rate**: Target 10-15 fps for smooth appearance
- **Only active camera**: Processing runs only for the camera currently being viewed on the frontend

### F3: Detector Comparison

- **Both directions**: Fetch detector data from ALL nearby stations (not just first 5)
- **Station aggregation**: Sum volume across lanes per station, average occupancy/speed per station
- **Direction matching**: Parse direction from detector labels, match to CV ROI directions
- **Time sync**: Detector data must target the same 30s slot as the CV window (use interval end timestamp to compute Mayfly index)
- **Display**: Per-direction side-by-side comparison (CV EB vs Det EB, CV WB vs Det WB)

### F4: Dashboard

- **Stat cards**: CV Volume / Occupancy / Speed alongside Det Volume / Occupancy / Speed (6 cards)
- **Road breakdown**: Per-direction badges showing CV metrics + matched detector station metrics
- **Charts**: Time series with CV vs detector count, occupancy, and 3-way speed (CV / MnDOT / ClearGuide)
- **Countdown**: Progress bar showing time until next 30s update
- **Stream status**: FPS, frame count display

## Non-Functional Requirements

### NF1: Stability

- **No segfaults**: OpenCV VideoCapture must not be accessed from multiple threads simultaneously
- **Graceful reconnection**: HLS streams drop periodically (MnDOT infrastructure); auto-reconnect with exponential backoff
- **No container crashes**: Unhandled exceptions in async tasks must not kill the event loop

### NF2: Performance

- **GPU**: YOLOv8n on RTX 6000 Ada — ~5ms/frame at 720x480
- **Concurrent cameras**: Support up to 4 simultaneous streams (GPU semaphore)
- **Bandwidth**: MJPEG at ~40KB/frame × 15fps = ~600KB/s per viewer
- **Memory**: Minimal — YOLOv8n is 6MB, ByteTrack state is negligible

### NF3: Latency

- **Acceptable**: Fixed 2-6s offset from real-time (HLS transport latency)
- **Not acceptable**: Progressive lag that grows over time
- **Requirement**: The displayed video must stay at a constant offset from real-time, never falling further behind

## Architecture Constraints

### The HLS Problem

MnDOT streams use HLS (HTTP Live Streaming), which delivers video in **segments** (typically 2-6s chunks). This creates two fundamental challenges:

1. **Buffer accumulation**: If frames are read slower than they arrive, the buffer grows and lag accumulates. Solution: continuously drain the buffer, always process the latest frame.

2. **Sync for overlay**: Playing HLS natively in the browser (smooth 30fps) while overlaying server-computed annotations (boxes/trails) is **impossible to synchronize** because the browser's HLS player and the server's HLS reader connect independently to MnDOT's CDN with different latencies.

### Chosen Approach: Server-Rendered MJPEG

The only way to guarantee perfect video-annotation sync is to render annotations server-side onto each frame before sending to the browser.

```
MnDOT HLS → [Buffer Drain] → Latest Frame → YOLO Track → Annotate → JPEG → Browser <img>
```

**The buffer drain mechanism must**:
- Read frames from HLS as fast as they arrive (~30fps)
- Always keep only the latest frame
- Never access the VideoCapture object from multiple threads simultaneously
- Handle HLS segment gaps gracefully (no crash on read timeout)

### Rejected Approaches

| Approach | Why rejected |
|----------|-------------|
| HLS `<video>` + canvas overlay | Cannot sync — two independent HLS connections have different latency |
| FrameGrabber in separate thread + `cap.release()` from main | Segfault — OpenCV FFMPEG backend is not thread-safe for concurrent access |
| Sequential `cap.read()` with `BUFFERSIZE=1` | Plays at segment delivery rate (~2fps), not real-time |
| `cap.grab()` to drain buffer | `grab()` also blocks on HLS — can't distinguish buffered vs live |

### Recommended Implementation: Single-Thread Reader with Drain

Use a single thread that owns the VideoCapture exclusively:

```python
class FrameReader:
    """Runs in its own thread. Only this thread touches cap."""
    
    def __init__(self, url):
        self._cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self._latest_frame = None
        self._lock = threading.Lock()
        self._should_stop = False
    
    def run(self):
        """Main loop — continuously read, keep latest, release cap on exit."""
        try:
            while not self._should_stop:
                ret, frame = self._cap.read()
                if not ret:
                    break
                with self._lock:
                    self._latest_frame = frame
        finally:
            self._cap.release()  # ONLY this thread releases
    
    def get_frame(self):
        with self._lock:
            return self._latest_frame
    
    def request_stop(self):
        """Signal stop — do NOT touch cap from outside."""
        self._should_stop = True
```

Key invariant: **the thread that creates `cap` is the only thread that calls `cap.read()` and `cap.release()`**. The main async loop only reads `self._latest_frame` through the lock.

## Test Camera

- **C843**: I-94 at Prior Ave, 720x480
- **ROIs**: I-94 EB (purple) + I-94 WB (blue)
- **Detectors**: 
  - WB station (69m): 3207-3210 (lanes 1-4)
  - EB station (241m): 3166-3169 (lanes 1-4)

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `hls_target_fps` | 15.0 | Processing frame rate |
| `aggregation_window_s` | 30.0 | Window matching MnDOT detector intervals |
| `yolo_tracker` | bytetrack.yaml | Tracker for model.track() |
| `max_concurrent_streams` | 4 | GPU semaphore limit |
| `speed_calibration_ppm` | 7.0 | Pixels-per-meter (camera-dependent) |
| `hls_reconnect_delay_s` | 5.0 | Base delay between reconnection attempts |
| `hls_reconnect_max_retries` | 10 | Give up after N consecutive failures |

## Calibration Notes

| Metric | CV vs Detector | Reason | Correction |
|--------|---------------|--------|------------|
| Volume | Close (±30%) | ROI may not cover all lanes | Adjust ROI polygon |
| Occupancy | CV >> Det (5-6x) | ROI spans ~50ft, loop detector is 6ft | Linear scale factor ~0.15 |
| Speed | CV << Det (with default ppm) | Camera-dependent pixel scale | Set `speed_calibration_ppm` per camera |
