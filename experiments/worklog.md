# Autoresearch Worklog: Trajectory ROI Clustering

## Session Info
- **Camera**: C844 (I-94, US Bank Stadium, Minneapolis)
- **Start**: 2026-04-15 ~03:50 UTC
- **Goal**: Optimize trajectory-based ROI polygon generation for I-94 EB/WB
- **Data**: 158 valid trajectories from 5-min nighttime capture (310 total tracks)

## Data Summary
- Direction A (axis ~101°): ~101 vehicles — upper-left to lower-right flow
- Direction B (axis ~281°): ~57 vehicles — lower-right to upper-left flow
- Two clear spatial streams visible in visualization
- Convex hulls overlap significantly in the center region

## Key Insights

1. **Decision boundary is the most impactful change** — perpendicular bisector between group centroids, with 10px margin, eliminates overlap while maintaining clean separation
2. **Concave hull (ratio=0.5) + 25px buffer** is the right polygon strategy — tighter than convex hull, but the buffer ensures road width coverage
3. **10 trail points per vehicle** is the sweet spot — fewer (5) misses road extent, more (50) overwhelms the hull algorithm
4. **Pre-clip trail points by decision boundary** prevents boundary-crossing trails from contaminating the other group
5. **85-87% coverage is correct** — the "missing" 13-15% are bridge/cross-traffic vehicles that correctly fall outside I-94 ROIs (97.4% of actual I-94 traffic is covered)
6. **Co-signed route merging** (I-94 > US 12 > MN 51) prevents trajectory splitting across duplicate road entries
7. **Algorithm generalizes** — tested on both C844 and C843 with good results

### Run 1: baseline — roi_quality=88.92 (KEEP) [EXPLORE]
- Buffered union 30px + polygon subtraction
- Coverage 82.3%, separation 6.7px

### Run 2: concave hull + decision boundary — roi_quality=92.96 (KEEP) [EXPLORE]
- Replaced buffered union with concave_hull(ratio=0.3) + 15px buffer
- Replaced polygon subtraction with perpendicular bisector decision boundary
- Separation jumped from 6.7 to 33px

### Run 3: wider buffer — roi_quality=93.76 (KEEP) [EXPLORE]
- ratio=0.5, buffer=25px — better coverage and area ratio

### Run 4: trail points + boundary margin — roi_quality=96.15 (KEEP) [EXPLORE]
- Store 5 sampled trail points per vehicle instead of just mean centroid
- Pre-clip trail points to decision boundary side
- 10px margin on boundary for clean gap

### Run 7: 10 trail points — roi_quality=96.79 (BEST) [EXPLORE]
- Increased trail sampling from 5 to 10 points per vehicle
- Coverage jumped to 87.2%

### Dead ends:
- IQR factor change: no effect (outliers are bridge traffic, filtered by direction)
- Oriented bounding rectangle: threshold never triggered
- All trail points (50/vehicle): too dense, oversimplifies
- Tighter simplify tolerance: cuts corners, reduces coverage
- Adaptive concave ratio: no benefit, data variability dominates

## Next Ideas
- Collect daytime data for comparison
- Test on more cameras (C626, C820, etc.)
- Investigate if longer collection duration (10+ min) improves stability
