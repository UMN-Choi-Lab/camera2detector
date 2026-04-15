"""Collect vehicle trajectories from a camera and visualize on a snapshot.

Usage:
    python experiments/visualize_trajectories.py C844 --duration 300

Connects to the tracking SSE, collects trajectory data, then plots
all vehicle centroids + direction arrows on a camera snapshot.
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import requests


def collect_trajectories(camera_id: str, duration_s: float, base_url: str) -> dict:
    """Collect trajectory data from the tracking SSE endpoint.

    Returns dict with:
        - trails: {track_id: [(cx, cy), ...]}
        - summaries: [(mean_cx, mean_cy, angle_deg, track_len)]
    """
    url = f"{base_url}/api/tracking/{camera_id}"
    print(f"Connecting to {url} for {duration_s}s...")

    trails: dict[int, list[tuple[float, float]]] = defaultdict(list)
    seen_in_frame: dict[int, list[dict]] = {}  # track_id -> list of detections

    start = time.time()
    frame_count = 0

    try:
        resp = requests.get(url, stream=True, timeout=10)
        resp.raise_for_status()

        buffer = ""
        for line in resp.iter_lines(decode_unicode=True):
            if time.time() - start > duration_s:
                break

            if line is None:
                continue

            if line.startswith("data:"):
                data_line = line[5:].strip()
                if not data_line:
                    continue

                try:
                    data = json.loads(data_line)
                except json.JSONDecodeError:
                    continue

                detections = data.get("detections", [])
                frame_count += 1

                for det in detections:
                    tid = det.get("track_id")
                    if tid is None:
                        continue
                    cx, cy = det["cx"], det["cy"]
                    trails[tid].append((cx, cy))

                elapsed = time.time() - start
                if frame_count % 50 == 0:
                    print(f"  {elapsed:.0f}s: {frame_count} frames, {len(trails)} tracks")

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Connection ended: {e}")

    # Compute summaries from trails
    summaries = []
    for tid, points in trails.items():
        if len(points) < 5:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        dx = points[-1][0] - points[0][0]
        dy = points[-1][1] - points[0][1]
        disp = math.sqrt(dx*dx + dy*dy)
        if disp < 3.0:
            continue
        angle = math.degrees(math.atan2(dx, -dy)) % 360
        # Sample up to 10 trail points
        n = len(points)
        max_pts = 10
        if n <= max_pts:
            sampled = [[float(p[0]), float(p[1])] for p in points]
        else:
            indices = [int(i * (n - 1) / (max_pts - 1)) for i in range(max_pts)]
            sampled = [[float(points[i][0]), float(points[i][1])] for i in indices]
        summaries.append((sum(xs)/len(xs), sum(ys)/len(ys), angle, len(points), sampled))

    print(f"\nCollected {frame_count} frames, {len(trails)} tracks, {len(summaries)} valid trajectories")
    return {"trails": dict(trails), "summaries": summaries}


def get_snapshot(camera_id: str) -> np.ndarray:
    """Download a camera snapshot."""
    url = f"https://video.dot.state.mn.us/video/image/metro/{camera_id}"
    print(f"Downloading snapshot from {url}...")
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def angle_to_color_hsv(angle_deg: float) -> tuple[int, int, int]:
    """Map angle [0, 360) to a BGR color via HSV colorwheel."""
    hue = int(angle_deg / 2)  # OpenCV hue is 0-179
    hsv = np.uint8([[[hue, 255, 220]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple(int(c) for c in bgr[0, 0])


def visualize(img: np.ndarray, data: dict, output_path: str):
    """Draw trajectories on the snapshot image."""
    trails = data["trails"]
    summaries = data["summaries"]

    # --- Panel 1: All trail paths ---
    panel1 = img.copy()
    overlay = panel1.copy()

    for tid, points in trails.items():
        if len(points) < 5:
            continue
        dx = points[-1][0] - points[0][0]
        dy = points[-1][1] - points[0][1]
        disp = math.sqrt(dx*dx + dy*dy)
        if disp < 3.0:
            continue
        angle = math.degrees(math.atan2(dx, -dy)) % 360
        color = angle_to_color_hsv(angle)

        pts = np.array(points, dtype=np.int32)
        cv2.polylines(overlay, [pts], False, color, 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.7, panel1, 0.3, 0, panel1)

    # Draw direction arrows at trail endpoints
    for tid, points in trails.items():
        if len(points) < 5:
            continue
        dx = points[-1][0] - points[0][0]
        dy = points[-1][1] - points[0][1]
        disp = math.sqrt(dx*dx + dy*dy)
        if disp < 3.0:
            continue
        angle = math.degrees(math.atan2(dx, -dy)) % 360
        color = angle_to_color_hsv(angle)

        # Arrow at the last point
        end = points[-1]
        norm = disp
        arrow_dx = int(dx / norm * 12)
        arrow_dy = int(dy / norm * 12)
        cv2.arrowedLine(
            panel1,
            (int(end[0]) - arrow_dx, int(end[1]) - arrow_dy),
            (int(end[0]), int(end[1])),
            color, 2, tipLength=0.4,
        )

    # Title
    cv2.putText(panel1, f"All Trails ({len(trails)} tracks)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # --- Panel 2: Trajectory centroids (mean positions) ---
    panel2 = img.copy()

    for s in summaries:
        mean_cx, mean_cy, angle, track_len = s[0], s[1], s[2], s[3]
        color = angle_to_color_hsv(angle)
        radius = max(3, min(8, track_len // 5))
        cv2.circle(panel2, (int(mean_cx), int(mean_cy)), radius, color, -1, cv2.LINE_AA)

        # Direction arrow from centroid
        arrow_len = 15
        dx = arrow_len * math.sin(math.radians(angle))
        dy = -arrow_len * math.cos(math.radians(angle))
        cv2.arrowedLine(
            panel2,
            (int(mean_cx), int(mean_cy)),
            (int(mean_cx + dx), int(mean_cy + dy)),
            color, 2, tipLength=0.4,
        )

    cv2.putText(panel2, f"Centroids ({len(summaries)} vehicles)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # --- Panel 3: Direction histogram / angle distribution ---
    panel3 = img.copy()

    # Separate by rough direction: split at 180 deg
    # Color by two main groups
    group_a = []  # angles roughly 0-180 (one direction)
    group_b = []  # angles roughly 180-360 (opposite)

    if summaries:
        angles = np.array([s[2] for s in summaries])
        # Use angle-doubling to find dominant axis
        doubled = 2.0 * np.deg2rad(angles)
        mean_doubled = math.atan2(np.mean(np.sin(doubled)), np.mean(np.cos(doubled)))
        axis = math.degrees(mean_doubled / 2.0) % 360

        for s in summaries:
            mean_cx, mean_cy, angle, track_len = s[0], s[1], s[2], s[3]
            dist_to_axis = min(abs(angle - axis) % 360, 360 - abs(angle - axis) % 360)
            if dist_to_axis < 90:
                group_a.append((mean_cx, mean_cy, angle, track_len))
                color = (255, 100, 50)   # Blue-ish = direction A
            else:
                group_b.append((mean_cx, mean_cy, angle, track_len))
                color = (50, 100, 255)   # Red-ish = direction B

            cv2.circle(panel3, (int(mean_cx), int(mean_cy)), 5, color, -1, cv2.LINE_AA)

        cv2.putText(panel3, f"Dir A: {len(group_a)} (axis={axis:.0f} deg)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 50), 2)
        cv2.putText(panel3, f"Dir B: {len(group_b)}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 100, 255), 2)

    # --- Panel 4: Current convex hull / ROI overlay ---
    panel4 = img.copy()

    if group_a:
        pts_a = np.array([(int(x), int(y)) for x, y, _, _ in group_a])
        if len(pts_a) >= 3:
            hull_a = cv2.convexHull(pts_a)
            cv2.drawContours(panel4, [hull_a], 0, (255, 100, 50), 2)
            overlay4 = panel4.copy()
            cv2.fillPoly(overlay4, [hull_a], (255, 100, 50))
            cv2.addWeighted(overlay4, 0.15, panel4, 0.85, 0, panel4)

    if group_b:
        pts_b = np.array([(int(x), int(y)) for x, y, _, _ in group_b])
        if len(pts_b) >= 3:
            hull_b = cv2.convexHull(pts_b)
            cv2.drawContours(panel4, [hull_b], 0, (50, 100, 255), 2)
            overlay4 = panel4.copy()
            cv2.fillPoly(overlay4, [hull_b], (50, 100, 255))
            cv2.addWeighted(overlay4, 0.15, panel4, 0.85, 0, panel4)

    cv2.putText(panel4, f"Convex Hulls", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # --- Combine into 2x2 grid ---
    top = np.hstack([panel1, panel2])
    bottom = np.hstack([panel3, panel4])
    combined = np.vstack([top, bottom])

    cv2.imwrite(output_path, combined)
    print(f"\nSaved visualization to {output_path}")
    print(f"  Image size: {combined.shape[1]}x{combined.shape[0]}")

    # Also save raw data
    data_path = output_path.replace(".jpg", ".json").replace(".png", ".json")
    with open(data_path, "w") as f:
        json.dump({
            "camera_id": args.camera_id,
            "duration_s": args.duration,
            "n_tracks": len(trails),
            "n_valid": len(summaries),
            "summaries": summaries,
        }, f, indent=2)
    print(f"  Raw data: {data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize vehicle trajectories")
    parser.add_argument("camera_id", help="Camera ID (e.g., C844)")
    parser.add_argument("--duration", type=int, default=300, help="Collection duration in seconds")
    parser.add_argument("--base-url", default="http://localhost:30000", help="API base URL")
    parser.add_argument("--output", default=None, help="Output image path")
    args = parser.parse_args()

    if args.output is None:
        Path("experiments").mkdir(exist_ok=True)
        args.output = f"experiments/{args.camera_id}_trajectories.jpg"

    # Collect data
    data = collect_trajectories(args.camera_id, args.duration, args.base_url)

    if not data["summaries"]:
        print("No valid trajectories collected. Is the camera streaming?")
        sys.exit(1)

    # Get snapshot
    snapshot = get_snapshot(args.camera_id)

    # Visualize
    visualize(snapshot, data, args.output)
