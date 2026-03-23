# Camera2Detector

**Real-time traffic camera validation of MnDOT highway detectors using computer vision**

Camera2Detector is a platform that uses YOLO-based computer vision on Minnesota Department of Transportation (MnDOT) highway camera feeds to generate independent traffic metrics (vehicle count, occupancy, speed) and correlates them in real time with MnDOT's ~7,000 loop/radar detectors. It enables transportation agencies to validate detector health, identify malfunctions, and supplement missing data with camera-derived measurements.

> **Developed by the [Choi Lab](https://choi-seongjin.umn.edu/) at the University of Minnesota Twin Cities, Department of Civil, Environmental, and Geo-Engineering (CEGE).**

---

## Motivation

MnDOT operates approximately **2,000 highway cameras** and **7,000 traffic detectors** across Minnesota. Detectors occasionally malfunction, drift, or go offline — but there is no independent ground truth to detect these failures in real time. Camera2Detector addresses this gap by:

1. Running object detection on existing camera feeds to produce an independent traffic measurement
2. Spatially matching cameras to nearby detectors using road geometry
3. Streaming side-by-side comparisons so operators can spot discrepancies instantly

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Road-Aware Vehicle Detection** | Vehicles are assigned to specific road segments (e.g., "I-94 WB", "I-35W NB") using shapefile geometry and tracking-based direction detection |
| **Multi-Frame Tracking** | 10 frames captured over 30 seconds; centroids matched across frames via the Hungarian algorithm to count unique vehicles and determine travel direction |
| **VLM-Based ROI Generation** | Vision-language models (GPT-5.3) automatically generate road-specific regions of interest for cameras viewing interchanges or multiple roads |
| **Real-Time SSE Streaming** | Server-Sent Events push CV + detector data every 30 seconds, matching MnDOT's native reporting interval |
| **Historical Comparison** | ClickHouse-backed overlay of same-time-yesterday data for trend analysis and anomaly detection |
| **Interactive Dashboard** | Dark-themed web UI with Leaflet map, Chart.js time series, and color-coded road overlays |
| **Per-Road Breakdown** | Count, occupancy, and vehicle type (car/motorcycle/bus/truck) breakdown per road visible from each camera |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser (Frontend)                        │
│  Leaflet Map  │  Camera Panel  │  Chart.js  │  ROI Editor   │
└──────┬──────────────┬──────────────┬──────────────┬─────────┘
       │              │              │              │
       ▼              ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Backend (:30000)                   │
│                                                             │
│  /api/cameras          – Camera listing & filtering         │
│  /api/camera/{id}/image – Proxy camera JPEG                 │
│  /api/cv/{id}          – Single-shot YOLO analysis          │
│  /api/sse/{id}         – 30s streaming (CV + detectors)     │
│  /api/camera/{id}/rois – ROI CRUD + VLM generation          │
│  /api/history/{id}     – ClickHouse time series             │
│  /api/detector/{id}    – Mayfly detector data proxy         │
└──────┬──────────────┬──────────────┬──────────────┬─────────┘
       │              │              │              │
       ▼              ▼              ▼              ▼
┌───────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────┐
│  MnDOT    │ │   YOLO     │ │ ClickHouse │ │  Road        │
│  APIs     │ │  (v8 nano) │ │  (sensors) │ │  Shapefile   │
│           │ │            │ │            │ │  (383 routes)│
│ • Images  │ │ • Detect   │ │ • raw_30s  │ │              │
│ • Config  │ │ • Track    │ │ • daily    │ │ • Geometry   │
│ • Mayfly  │ │ • Assign   │ │   metrics  │ │ • Bearings   │
└───────────┘ └────────────┘ └────────────┘ └──────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.12, FastAPI, Uvicorn, Pydantic |
| **Object Detection** | YOLOv8 nano (Ultralytics) |
| **Tracking** | Multi-frame centroid matching via SciPy (Hungarian algorithm) |
| **Road Geometry** | GeoPandas, Shapely, R-tree spatial index |
| **Time-Series DB** | ClickHouse (27.9B rows of 30-second detector data) |
| **VLM ROI Generation** | OpenAI API (GPT-5.3 with chain-of-thought prompting) |
| **Frontend** | Vanilla JS, Leaflet.js, Chart.js, Server-Sent Events |
| **Deployment** | Docker (Python 3.12-slim + GDAL + OpenCV) |

---

## Data Sources

### MnDOT Live APIs
- **Camera images**: JPEG snapshots from ~2,000 highway cameras statewide
- **Metro config**: XML manifest of all cameras and detectors with coordinates
- **Mayfly**: Real-time 30-second detector readings (volume, occupancy, speed) — 2,880 values per day per detector

### Road Geometry (Shapefile)
- **MnDOT Trunk Highways**: 383 route segments (79 in metro), EPSG:26915 → EPSG:4326
- Fields: route name, direction (D/I), cardinal (NB/SB/EB/WB)

### ClickHouse (Historical)
- **`sensors.raw_30s`**: 30-second detector readings (volume, occupancy, speed)
- **`sensors.detector_meta`**: 4,109 detector locations with route/direction/lane
- **`sensors.daily_metrics`**: Pre-aggregated daily statistics

---

## Getting Started

### Prerequisites

- Python 3.12+
- Docker (recommended)
- MnDOT data is publicly accessible (no API key needed for cameras/detectors)
- OpenAI API key (optional, only for VLM-based ROI generation)
- ClickHouse instance (optional, only for historical data overlay)
- Road shapefile: `data/shp/Trunk_Highways_in_Minnesota.shp` (download from [MnDOT GIS](https://gisdata.mn.gov/))

### Local Development

```bash
# Clone the repository
git clone https://github.com/UMN-Choi-Lab/camera2detector.git
cd camera2detector

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your settings (all optional except port)

# Place the shapefile
mkdir -p data/shp
# Copy Trunk_Highways_in_Minnesota.shp (and .shx, .dbf, .prj) to data/shp/

# Run with hot reload
C2D_RELOAD=1 python run.py
```

The dashboard will be available at `http://localhost:30000`.

### Docker

```bash
# Build
docker build -t camera2detector .

# Run (--network host required for ClickHouse access)
docker run -d \
  --name camera2detector \
  --network host \
  --env-file .env \
  -v $(pwd)/data/rois:/app/data/rois \
  -v $(pwd)/data/shp:/app/data/shp \
  camera2detector
```

The YOLO model (`yolov8n.pt`) is automatically downloaded during the Docker build.

---

## Configuration

All settings use the `C2D_` environment variable prefix (via Pydantic Settings):

| Variable | Default | Description |
|----------|---------|-------------|
| `C2D_PORT` | `30000` | Server port |
| `C2D_YOLO_MODEL` | `yolov8n.pt` | YOLO model file |
| `C2D_YOLO_CONFIDENCE` | `0.3` | Detection confidence threshold |
| `C2D_VEHICLE_CLASSES` | `[2,3,5,7]` | COCO classes: car, motorcycle, bus, truck |
| `C2D_TRACKING_FRAMES` | `10` | Frames per 30-second window |
| `C2D_TRACKING_INTERVAL_S` | `3.0` | Seconds between frames |
| `C2D_MAX_MATCH_DISTANCE_M` | `500` | Camera-to-detector matching radius (meters) |
| `C2D_ROAD_SEARCH_RADIUS_M` | `300` | Camera-to-road matching radius (meters) |
| `C2D_CLICKHOUSE_HOST` | `localhost` | ClickHouse server host |
| `C2D_CLICKHOUSE_PORT` | `8123` | ClickHouse HTTP port |
| `C2D_OPENAI_API_KEY` | *(empty)* | OpenAI key for VLM ROI generation |
| `C2D_OPENAI_MODEL` | `gpt-5.3-chat-latest` | VLM model for ROI generation |

---

## Project Structure

```
camera2detector/
├── backend/
│   ├── app.py                  # FastAPI app factory & lifespan
│   ├── config.py               # Pydantic settings (C2D_ prefix)
│   ├── models/
│   │   └── schemas.py          # 15 Pydantic models (CVResult, SSEEvent, ROI, etc.)
│   ├── routers/
│   │   ├── proxy.py            # MnDOT API proxy endpoints
│   │   ├── cv.py               # Single-shot YOLO inference
│   │   ├── sse.py              # SSE streaming with tracking pipeline
│   │   ├── history.py          # ClickHouse time-series API
│   │   ├── roi.py              # ROI CRUD + VLM generation
│   │   └── clearguide.py       # ClearGuide speed data proxy
│   └── services/
│       ├── mndot_client.py     # Async HTTP client for MnDOT APIs
│       ├── metro_config.py     # XML parsing + haversine spatial matching
│       ├── cv_pipeline.py      # YOLO detection + multi-frame tracking
│       ├── road_geometry.py    # Shapefile loading + spatial queries
│       ├── clickhouse_client.py # ClickHouse queries
│       ├── vlm_roi.py          # VLM ROI suggestor-reviewer loop
│       └── clearguide_client.py # ClearGuide OAuth2 + speed queries
├── frontend/
│   ├── index.html              # Main dashboard layout
│   ├── roi_tool.html           # Standalone ROI annotation tool
│   ├── css/style.css           # Dark theme (576 lines)
│   └── js/
│       ├── app.js              # Orchestrator: camera click → load → stream
│       ├── map.js              # Leaflet map with camera clusters + road lines
│       ├── camera_panel.js     # Image overlay with road-colored bounding boxes
│       ├── charts.js           # Chart.js time series (count, occupancy, speed)
│       ├── sse_client.js       # EventSource connection manager
│       ├── roi_editor.js       # Interactive polygon drawing on canvas
│       └── utils.js            # Formatting and normalization helpers
├── data/
│   ├── shp/                    # Road shapefile (not included — see setup)
│   └── rois/                   # Cached ROI JSON files per camera
├── Dockerfile
├── requirements.txt
├── run.py                      # Uvicorn entry point
└── .env.example                # Environment variable template
```

---

## API Reference

### Camera & Detector Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/cameras?road={name}` | List cameras, optionally filtered by road name |
| `GET` | `/api/camera/{id}/image` | Proxy camera JPEG from MnDOT |
| `GET` | `/api/camera/{id}/detectors` | Detectors within 500m of camera |
| `GET` | `/api/camera/{id}/roads` | Nearby road segments from shapefile |
| `GET` | `/api/detector/{id}/{type}` | Mayfly data (`counts`, `occupancy`, `speed`) |

### Computer Vision Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/cv/{camera_id}` | Single-frame YOLO detection → CVResult |
| `GET` | `/api/sse/{camera_id}` | SSE stream: CV + detector data every 30s |

### ROI Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/camera/{id}/rois` | Load cached ROIs |
| `POST` | `/api/camera/{id}/rois/generate` | Auto-generate via VLM |
| `PUT` | `/api/camera/{id}/rois` | Save manually edited ROIs |
| `DELETE` | `/api/camera/{id}/rois/{roi_id}` | Delete a single ROI |

### Historical Data (ClickHouse)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/history/detectors` | All detector metadata |
| `GET` | `/api/history/{detector_id}?type=v30&start=...&end=...` | Time-series data (raw or downsampled) |
| `GET` | `/api/history/{detector_id}/daily` | Daily aggregated metrics |

---

## How It Works

### 1. Vehicle Detection & Tracking

The CV pipeline captures **10 frames over 30 seconds** (one every 3s) from a camera and:

1. **Detects** vehicles in each frame using YOLOv8 nano (classes: car, motorcycle, bus, truck)
2. **Tracks** vehicles across frames by matching bounding box centroids using the **Hungarian algorithm** (SciPy's `linear_sum_assignment`)
3. **Counts** unique vehicles that appear across multiple frames (reducing double-counting from stationary vehicles)
4. **Computes occupancy** as the average ratio of bounding box area to effective road area across all frames

### 2. Road Assignment

Each detected vehicle is assigned to a specific road segment. Two methods are used:

- **ROI-based** (preferred): If the camera has defined Regions of Interest, vehicles are assigned via point-in-polygon tests on their centroid positions
- **Bearing-based** (fallback): The pixel velocity vector of tracked vehicles is converted to a compass direction and matched against the bearing of nearby road segments from the shapefile

### 3. VLM ROI Generation

For cameras viewing interchanges or multiple roads, a **suggestor-reviewer loop** automatically generates ROI polygons:

1. **Suggestor**: A vision-language model (GPT-5.3) receives the camera image + nearby road list and generates polygon coordinates using chain-of-thought reasoning
2. **Reviewer**: Evaluates polygons on 5 criteria (on pavement, angle matches road, covers lanes, proper placement, reasonable size) with a score out of 20
3. **Iteration**: If the score < 14/20, feedback is sent back to the suggestor for refinement (up to 3 iterations)

### 4. Detector Correlation

Camera-derived metrics are displayed alongside MnDOT detector readings for real-time comparison:

- **Spatial matching**: Detectors within 500m of the camera are matched using haversine distance from the metro config XML
- **Temporal alignment**: Both sources report at 30-second intervals, enabling direct comparison
- **Historical overlay**: ClickHouse provides same-time-yesterday data as dashed lines for trend analysis

---

## CV Pipeline Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | YOLOv8 nano | Fast inference for real-time streaming |
| Confidence | 0.3 | Balanced recall vs. precision for highway scenes |
| Vehicle classes | car (2), motorcycle (3), bus (5), truck (7) | COCO classes relevant to highway traffic |
| Tracking window | 10 frames × 3s = 30s | Matches MnDOT detector reporting interval |
| Occupancy formula | `Σ box_area / (image_area × 0.6)` | 0.6 factor approximates visible road surface |
| Matching threshold | 50px centroid distance | Balances re-identification vs. new vehicle detection |

---

## Acknowledgments

- **Minnesota Department of Transportation (MnDOT)** for publicly available camera feeds, detector data, and road geometry
- **Ultralytics** for the YOLOv8 object detection framework

---

## License

This project is developed for research purposes at the University of Minnesota. See [LICENSE](LICENSE) for details.

---

## Citation

If you use Camera2Detector in your research, please cite:

```bibtex
@software{camera2detector2026,
  author = {Choi, Seongjin},
  title = {Camera2Detector: Real-Time Traffic Camera Validation of Highway Detectors},
  year = {2026},
  institution = {University of Minnesota},
  url = {https://github.com/UMN-Choi-Lab/camera2detector}
}
```

---

## Contact

**Seongjin Choi, Ph.D.**
Assistant Professor, Department of Civil, Environmental, and Geo-Engineering
University of Minnesota Twin Cities
[choi-seongjin.umn.edu](https://choi-seongjin.umn.edu/)
