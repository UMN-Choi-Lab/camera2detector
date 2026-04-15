# Autoresearch Dashboard: Trajectory ROI Clustering

**Phase:** EXPLORE → EXPLOIT | **Runs:** 11 | **Kept:** 4 | **Discarded:** 7 | **Crashed:** 0
**Baseline:** roi_quality: 88.92 score (#1)
**Best:** roi_quality: 96.79 score (#7, +8.9%)
**Consecutive discards:** 4

## Improvement Trajectory
```
#1  ████████████████████████████████████████░░░  88.92  baseline (buffered union)
#2  ██████████████████████████████████████████░  92.96  +4.5% concave hull + boundary
#3  ██████████████████████████████████████████░  93.76  +5.4% wider buffer
#4  ████████████████████████████████████████████ 96.15  +8.1% trail pts + margin
#7  ████████████████████████████████████████████ 96.79  +8.9% 10 trail pts  <- BEST
```

## Algorithm Summary (Best Configuration)
1. Accumulate TrajectorySummary(mean_cx, mean_cy, angle, trail_points[10]) per vehicle
2. Assign to road-direction by angular distance (merge co-signed routes)
3. IQR outlier removal on mean centroids
4. Pre-clip trail points by decision boundary (perpendicular bisector - 10px margin)
5. Concave hull (ratio=0.5) + 25px buffer → simplify → clip to image

## Validation
- C844: 2 ROIs (I 94 EB + WB), 87% coverage, 97.4% of actual I-94 traffic
- C843: 2 ROIs (I 94 EB + WB), 85% coverage, generalizes correctly
