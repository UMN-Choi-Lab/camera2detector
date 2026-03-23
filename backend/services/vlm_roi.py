"""VLM-based ROI generation using OpenAI vision models.

Uses a suggestor-reviewer loop: the suggestor generates ROI polygons,
then a reviewer evaluates them on the annotated image. If rejected,
the suggestor retries with the reviewer's feedback. Repeats until
approved or max iterations reached.
"""

import base64
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from backend.config import settings
from backend.models.schemas import ROIPolygon, CameraROIs
from backend.services.road_geometry import road_geometry_service

logger = logging.getLogger(__name__)

ROI_COLORS = [
    "#a855f7",  # purple
    "#3b82f6",  # blue
    "#ef4444",  # red
    "#f59e0b",  # amber
    "#10b981",  # emerald
    "#ec4899",  # pink
    "#06b6d4",  # cyan
    "#f97316",  # orange
]

MAX_REVIEW_ITERATIONS = 3


def _is_reasoning_model(model: str) -> bool:
    return any(model.startswith(prefix) for prefix in ("gpt-5", "o1", "o3", "o4"))


class VLMROIService:
    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not settings.openai_api_key:
                raise ValueError("C2D_OPENAI_API_KEY not set")
            from openai import OpenAI
            self._client = OpenAI(api_key=settings.openai_api_key)
        return self._client

    def _roi_dir(self) -> Path:
        p = Path(settings.roi_data_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _roi_path(self, camera_id: str) -> Path:
        return self._roi_dir() / f"{camera_id}.json"

    def load_rois(self, camera_id: str) -> CameraROIs | None:
        path = self._roi_path(camera_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return CameraROIs(**data)
        except Exception:
            logger.exception("Failed to load ROIs for %s", camera_id)
            return None

    def save_rois(self, rois: CameraROIs) -> None:
        path = self._roi_path(rois.camera_id)
        path.write_text(json.dumps(rois.model_dump(), indent=2))
        logger.info("Saved %d ROIs for camera %s", len(rois.rois), rois.camera_id)

    def delete_roi(self, camera_id: str, roi_id: str) -> bool:
        existing = self.load_rois(camera_id)
        if not existing:
            return False
        existing.rois = [r for r in existing.rois if r.roi_id != roi_id]
        self.save_rois(existing)
        return True

    # ── Prompts ──────────────────────────────────────────────────────────

    def _build_suggestor_prompt(self, image_width: int, image_height: int,
                                nearby_roads: list[dict],
                                reviewer_feedback: str | None = None) -> str:
        focus_corridors = {"I-94", "I-494", "I-694", "I-35E", "I-35W"}
        road_descriptions = []
        for road in nearby_roads:
            label = road.get("route_label", "")
            cardinal = road.get("cardinal", "")
            if any(c in label for c in focus_corridors) or label.startswith("I-"):
                road_descriptions.append(f"{label} ({cardinal})" if cardinal else label)
        if not road_descriptions:
            for road in nearby_roads:
                label = road.get("route_label", "")
                cardinal = road.get("cardinal", "")
                road_descriptions.append(f"{label} ({cardinal})" if cardinal else label)
        road_list = ", ".join(dict.fromkeys(road_descriptions))

        feedback_section = ""
        if reviewer_feedback:
            feedback_section = f"""

IMPORTANT — PREVIOUS ATTEMPT WAS REJECTED. Reviewer feedback:
{reviewer_feedback}

You MUST fix all issues listed above. Pay careful attention to:
- Moving vertices that were off-road back onto pavement
- Adjusting the tilt angle to match the road
- Resizing zones that were too small or too large
- Repositioning zones that were in obstructed or washed-out areas
"""

        return f"""You are placing virtual loop-detector zones on a highway camera image for vehicle counting.

IMAGE: {image_width}x{image_height} px. Origin (0,0) = top-left. X right, Y down.
CAMERA: Near {road_list}. Mounted high, looking down at the highway.
{feedback_section}
TASK: For each visible direction of travel, draw ONE detection zone — a compact parallelogram strip that cuts across ALL lanes at one cross-section of the road, like a virtual inductive loop detector.

STEP 1 — ANALYZE THE ROAD:
- Find the MEDIAN (center divider) — barrier, yellow lines, or grass strip
- Find the OUTER EDGE of travel lanes — where pavement meets grass/dirt/shoulder
- Determine the road ANGLE through the frame

STEP 2 — CHOOSE WHERE TO PLACE EACH ZONE:
- Where individual vehicles are clearly visible (car/truck shapes, not just headlight blobs)
- NOT obstructed by signs, bridges, trees, or poles
- NOT in the distant section where vehicles are tiny — NOT at the very bottom where vehicles are cut off
- Typically in the middle third of the visible road length

STEP 3 — DRAW EACH ZONE as a narrow parallelogram:
- TWO LONG edges span across ALL lanes (median to outer edge), PERPENDICULAR to lane direction
- TWO SHORT edges run PARALLEL to the lane direction (~15-25% of visible road length)
- The parallelogram MUST be TILTED to match the road's actual angle in the image
- All 4 vertices MUST be on PAVED ROAD SURFACE — no grass, dirt, median barrier, shoulder, or signs

SIZE: each zone ~3-8% of image area. Both together ~6-14%.
VERTEX ORDER: 4 vertices, clockwise from bottom-left.

FEW-SHOT EXAMPLES (720x480 images — note tilted parallelograms matching road angle):

Example A — road from lower-left to upper-right:
  Direction 1: [[130,290], [248,328], [296,283], [193,245]]
  Direction 2: [[266,338], [418,372], [446,311], [309,291]]

Example B — road angled steeply lower-center to upper-right:
  Direction 1: [[226,170], [338,201], [456,161], [337,139]]
  Direction 2: [[412,206], [652,279], [703,232], [499,169]]

Example C — road from lower-left to center:
  Direction 1: [[142,259], [261,297], [318,271], [202,232]]
  Direction 2: [[290,310], [511,407], [562,349], [347,281]]

First, briefly describe:
1. Road angle through the frame
2. Median location and outer lane edges
3. Where you'll place each zone and why

Then output JSON:
```json
{{
  "rois": [
    {{
      "road_name": "I-94",
      "direction": "EB",
      "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    }}
  ]
}}
```

Rules:
- road_name: hyphenated (I-94, I-35W). direction: NB/SB/EB/WB.
- Exactly 4 integer vertices, clockwise from bottom-left
- Zone TILTED to match road angle, all vertices on pavement
- Each zone ~3-8% of image area"""

    def _build_reviewer_prompt(self, image_width: int, image_height: int,
                               rois_json: list[dict]) -> str:
        roi_desc = json.dumps(rois_json, indent=2)
        return f"""You are reviewing detection-zone ROI polygons drawn on a highway camera image.

IMAGE: {image_width}x{image_height} px. The colored polygons overlaid on the image are the proposed detection zones.

PROPOSED ROIs:
{roi_desc}

Score EACH zone on these criteria (each 0-2: 0=bad, 1=acceptable, 2=good):

1. ON PAVEMENT: Are vertices on paved road surface? (0=mostly off-road, 1=mostly on pavement but 1-2 vertices slightly off, 2=all on pavement)
2. ROAD ANGLE: Does the parallelogram tilt match the road's perspective angle? (0=totally wrong angle, 1=close but slightly off, 2=matches well)
3. LANE COVERAGE: Does the zone span across the travel lanes? (0=covers less than half the lanes, 1=covers most lanes, 2=spans full width)
4. PLACEMENT: Is the zone where vehicles are clearly visible? (0=in glare/behind sign/too far, 1=mostly clear with minor issue, 2=clear area)
5. SIZE: Is each zone ~3-8% of image area? (0=way too small or large, 1=slightly outside range, 2=in range)

Output your evaluation as JSON:
```json
{{
  "score": 0-10,
  "approved": true or false,
  "issues": [
    {{
      "roi_index": 0,
      "road_name": "I-94",
      "direction": "EB",
      "problems": ["vertex at [x,y] is on grass"],
      "suggestion": "move bottom-right vertex to approximately [new_x, new_y]"
    }}
  ],
  "summary": "Brief overall assessment"
}}
```

SCORING:
- Score is the SUM of all criteria scores across all zones (max 10 per zone, 20 for two zones)
- Set approved=true if score >= 14 (out of 20 for two zones) AND no zone has a 0 on any criterion
- Set approved=false if score < 14 OR any zone scores 0 on pavement or placement
- A zone that is "pretty good but not perfect" should still be approved — perfection is not required
- Only list issues for criteria scoring 0 or 1, with specific fix suggestions"""

    # ── VLM call helpers ─────────────────────────────────────────────────

    def _call_vlm(self, prompt: str, b64_image: str) -> str:
        client = self._get_client()
        model = settings.openai_model
        api_params = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_image}",
                        "detail": "high",
                    }},
                ],
            }],
        }
        if _is_reasoning_model(model):
            api_params["max_completion_tokens"] = 16000
        else:
            api_params["max_tokens"] = 2000
            api_params["temperature"] = 0.1
        response = client.chat.completions.create(**api_params)
        return response.choices[0].message.content

    def _draw_rois_on_image(self, img, rois: list[dict]) -> bytes:
        """Draw ROI polygons on image, return JPEG bytes for reviewer."""
        from PIL import ImageDraw, ImageFont
        annotated = img.copy()
        draw = ImageDraw.Draw(annotated, "RGBA")
        hex_colors = ["#a855f7", "#3b82f6", "#ef4444", "#f59e0b"]
        for i, roi in enumerate(rois):
            hex_c = hex_colors[i % len(hex_colors)]
            r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
            pts = [(p[0], p[1]) for p in roi["polygon"]]
            draw.polygon(pts, fill=(r, g, b, 60), outline=(r, g, b, 220))
            for j in range(len(pts)):
                draw.line([pts[j], pts[(j+1) % len(pts)]], fill=(r, g, b, 255), width=3)
            for pt in pts:
                draw.ellipse([pt[0]-4, pt[1]-4, pt[0]+4, pt[1]+4], fill=(r, g, b, 255))
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            label = f"{roi.get('road_name','')} {roi.get('direction','')}"
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            except (IOError, OSError):
                font = ImageFont.load_default()
            bbox = draw.textbbox((cx, cy), label, font=font)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
            draw.rectangle([cx-tw/2-2, cy-th/2-2, cx+tw/2+2, cy+th/2+2], fill=(0,0,0,180))
            draw.text((cx-tw/2, cy-th/2), label, fill=(255,255,255), font=font)
        buf = BytesIO()
        annotated.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def _parse_reviewer_response(self, raw_text: str) -> dict:
        text = raw_text.strip()
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        else:
            json_match = re.search(r'(\{\s*"approved"\s*:.*)', text, re.DOTALL)
            if json_match:
                text = json_match.group(1)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.error("Failed to parse reviewer JSON: %s", text[:500])
            return {"approved": False, "issues": [], "summary": "Parse error"}

    def _format_reviewer_feedback(self, review: dict) -> str:
        lines = []
        if review.get("summary"):
            lines.append(f"Overall: {review['summary']}")
        for issue in review.get("issues", []):
            road = issue.get("road_name", "?")
            direction = issue.get("direction", "?")
            lines.append(f"\n{road} {direction}:")
            for p in issue.get("problems", []):
                lines.append(f"  - {p}")
            if issue.get("suggestion"):
                lines.append(f"  FIX: {issue['suggestion']}")
        return "\n".join(lines)

    # ── Main generation with review loop ─────────────────────────────────

    async def generate_rois(self, camera_id: str, image_bytes: bytes) -> CameraROIs:
        """Generate ROIs using suggestor-reviewer loop."""
        from PIL import Image

        img = Image.open(BytesIO(image_bytes))
        img_w, img_h = img.size

        all_roads = road_geometry_service.get_camera_roads(camera_id)
        # Filter to I-94 only for focused prompting
        nearby_roads = [r for r in all_roads if r.get("route_label") in ("I-94", "I 94")]
        if not nearby_roads:
            nearby_roads = all_roads  # fallback if no I-94 match
        b64_original = base64.b64encode(image_bytes).decode("utf-8")

        reviewer_feedback = None
        best_rois_raw = []
        best_score = -1

        for iteration in range(1, MAX_REVIEW_ITERATIONS + 1):
            logger.info("ROI generation for %s — iteration %d/%d",
                       camera_id, iteration, MAX_REVIEW_ITERATIONS)

            # ── Suggestor ──
            prompt = self._build_suggestor_prompt(
                img_w, img_h, nearby_roads, reviewer_feedback)
            raw_suggest = self._call_vlm(prompt, b64_original)
            logger.info("Suggestor response for %s (iter %d): %s",
                       camera_id, iteration, raw_suggest[:200])

            rois_raw = self._parse_rois_raw(raw_suggest, img_w, img_h)
            if not rois_raw:
                logger.warning("No ROIs parsed for %s (iter %d)", camera_id, iteration)
                break

            # ── Reviewer ──
            annotated_bytes = self._draw_rois_on_image(img, rois_raw)
            b64_annotated = base64.b64encode(annotated_bytes).decode("utf-8")

            review_prompt = self._build_reviewer_prompt(
                img_w, img_h,
                [{"road_name": r["road_name"], "direction": r["direction"],
                  "polygon": r["polygon"]} for r in rois_raw])

            raw_review = self._call_vlm(review_prompt, b64_annotated)
            logger.info("Reviewer response for %s (iter %d): %s",
                       camera_id, iteration, raw_review[:200])

            review = self._parse_reviewer_response(raw_review)
            approved = review.get("approved", False)
            score = review.get("score", 0)

            # Track best by score
            if score > best_score:
                best_score = score
                best_rois_raw = rois_raw

            if approved:
                logger.info("ROIs APPROVED for %s (iter %d, score %d)",
                           camera_id, iteration, score)
                best_rois_raw = rois_raw
                break
            else:
                reviewer_feedback = self._format_reviewer_feedback(review)
                logger.info("ROIs REJECTED for %s (iter %d, score %d): %s",
                           camera_id, iteration, score, review.get("summary", ""))

        final_rois_raw = best_rois_raw

        # Convert raw dicts to ROIPolygon objects
        rois = []
        for i, roi in enumerate(final_rois_raw):
            color = ROI_COLORS[i % len(ROI_COLORS)]
            rois.append(ROIPolygon(
                roi_id=str(uuid.uuid4())[:8],
                road_name=roi["road_name"],
                direction=roi["direction"],
                polygon=roi["polygon"],
                color=color,
            ))

        camera_rois = CameraROIs(
            camera_id=camera_id,
            image_width=img_w,
            image_height=img_h,
            rois=rois,
            generated_at=datetime.now(timezone.utc).isoformat(),
            source="vlm",
        )

        self.save_rois(camera_rois)
        return camera_rois

    def _parse_rois_raw(self, raw_text: str, img_w: int, img_h: int) -> list[dict]:
        """Parse VLM response into raw dicts (used internally before converting to ROIPolygon)."""
        text = raw_text.strip()
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        else:
            json_match = re.search(r'(\{\s*"rois"\s*:.*)', text, re.DOTALL)
            if json_match:
                text = json_match.group(1)
            else:
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.error("Failed to parse VLM JSON: %s", text[:500])
            return []

        result = []
        for roi in data.get("rois", []):
            road_name = roi.get("road_name", "Unknown")
            direction = roi.get("direction", "")
            polygon = roi.get("polygon", [])
            road_name = re.sub(r'^(I|US|MN|TH)\s+', r'\1-', road_name)
            if len(polygon) < 3:
                continue
            clamped = []
            for pt in polygon:
                if len(pt) >= 2:
                    clamped.append([max(0, min(pt[0], img_w)),
                                    max(0, min(pt[1], img_h))])
            if len(clamped) < 3:
                continue
            result.append({
                "road_name": road_name,
                "direction": direction,
                "polygon": clamped,
            })
        return result

    def _parse_response(self, raw_text: str, img_w: int, img_h: int) -> list[ROIPolygon]:
        """Legacy parse method — kept for compatibility."""
        raw = self._parse_rois_raw(raw_text, img_w, img_h)
        result = []
        for i, roi in enumerate(raw):
            color = ROI_COLORS[i % len(ROI_COLORS)]
            result.append(ROIPolygon(
                roi_id=str(uuid.uuid4())[:8],
                road_name=roi["road_name"],
                direction=roi["direction"],
                polygon=roi["polygon"],
                color=color,
            ))
        return result


vlm_roi_service = VLMROIService()
