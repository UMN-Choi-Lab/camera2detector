# Autoresearch: Trajectory-Based ROI Clustering for I-94

## Objective
Optimize the polygon generation algorithm in `backend/services/trajectory_cluster.py` so that it produces clean, tight, non-overlapping ROI polygons for I-94 EB and I-94 WB from accumulated vehicle trajectory data. The test case is camera C844 (I-94 near US Bank Stadium, Minneapolis).

## Metrics
- **Primary**: `roi_quality` (composite score 0-100, higher is better)
  - Components: 2-roi penalty (-20 if not exactly 2 ROIs), overlap penalty (-area overlap), vertex count penalty (>20 vertices), coverage ratio (area covering actual trajectory points), separation clarity (gap between direction groups)
- **Secondary**: `n_rois`, `total_vertices`, `overlap_area_pct`, `coverage_pct`, `separation_px`

## Statistical Protocol
- Seeds: [1] (deterministic algorithm, no randomness)
- Report: single metric value
- Keep threshold: any improvement

## How to Run
`./experiments/autoresearch.sh` — outputs `METRIC name=number` lines.

## Files in Scope
- `backend/services/trajectory_cluster.py` — main algorithm (polygon generation, direction assignment, outlier removal)
- `experiments/evaluate_roi.py` — evaluation script (reads trajectory data, runs algorithm, scores output)
- `experiments/C844_trajectories.json` — collected trajectory data (158 vehicles, 5 min capture)

## Off Limits
- `backend/services/stream_worker.py` — integration code, not the algorithm
- `backend/routers/roi.py` — API layer
- `backend/config.py` — settings (can add new ones but don't change existing)
- All test files

## Constraints
- Only shapely, scipy, numpy (no scikit-learn, no new dependencies)
- Must produce valid CameraROIs JSON format
- Must work for any camera, not just C844
- ROI polygons: 3-20 vertices, non-overlapping
- Focus on I-94 only (filter non-I-94 roads)

## Current Phase
EXPLORE (run 1-N)

## What's Been Tried
(Updated as experiments accumulate)

## Key Insights
- C844 trajectory visualization shows two clear directional streams
- Upper-left→lower-right (~101 vehicles) vs lower-right→upper-left (~57 vehicles)
- Streams are spatially close but separable — convex hull too loose, causes overlap
- Buffered union (30px) produces blobby shapes
- Need decision boundary between direction groups
