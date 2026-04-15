# Ideas Backlog

## From Literature
- [ ] [literature] Use `shapely.concave_hull(ratio=0.3)` instead of buffered union — Shapely 2.0+ has built-in concave hull, tighter than convex, no extra deps
- [ ] [literature] Compute decision boundary as perpendicular bisector between direction cluster centroids, then clip polygons to their side — standard technique from lane detection papers (Ren 2014, IEEE)
- [ ] [literature] K-means clustering on (x, y, cos θ, sin θ) feature space with K=2 for two directions — used in trajectory-based lane detection
- [ ] [literature] Hausdorff distance for trajectory similarity — can group trails more accurately than mean centroids
- [ ] [literature] Use the full trail polyline (not just mean centroid) to generate denser point clouds for polygon estimation

## From Code Analysis
- [ ] [analysis] Replace IQR outlier removal with Mahalanobis distance — respects covariance structure of elongated road clusters
- [ ] [analysis] Use the individual trail points (not mean centroid) as the point cloud — gives 10x more points for polygon estimation
- [ ] [analysis] Compute oriented bounding rectangle (min_rotated_rectangle) as a simpler alternative — roads are roughly parallelogram-shaped in perspective
- [ ] [analysis] Filter by I-94 specifically: when user specifies a target road, only keep that road's direction targets
- [ ] [analysis] Adaptive buffer radius: larger near bottom of image (closer to camera), smaller near top (farther away) — perspective scaling

## From Visualization
- [ ] [visualization] The two direction streams have clear spatial separation (upper vs lower in image) — a simple y-threshold midline might work
- [ ] [visualization] Many vehicles cluster tightly along diagonal lines — concave hull or alpha shape would fit much better than convex hull
- [ ] [visualization] Trail paths show the actual road lanes — using all trail points (not just centroids) would give much richer data for polygon fitting
- [ ] [visualization] Direction A has 101 vehicles, B has 57 — skewed nighttime traffic, algorithm must handle imbalanced groups

## Intuition
- [ ] [intuition] Two-step approach: (1) compute decision boundary line between groups, (2) use concave hull on each side — clean separation + tight fit
- [ ] [intuition] Use the mean trail path (average all trails per direction) as the "road centerline", then buffer it with adaptive width
- [ ] [intuition] Score-based vertex reduction: start with detailed polygon, progressively simplify while monitoring point coverage
