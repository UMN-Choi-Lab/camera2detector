"""Parse metro_config.xml.gz and perform spatial matching between cameras and detectors."""

import math
import xml.etree.ElementTree as ET
import logging

from backend.services.mndot_client import mndot_client
from backend.config import settings
from backend.models.schemas import CameraInfo, DetectorInfo, CameraWithDetectors

logger = logging.getLogger(__name__)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in meters between two lat/lon points."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class MetroConfigService:
    def __init__(self):
        self.cameras: dict[str, CameraInfo] = {}
        self.detectors: dict[str, DetectorInfo] = {}
        self.camera_detectors: dict[str, list[DetectorInfo]] = {}

    async def load(self):
        """Fetch and parse metro_config.xml.gz."""
        try:
            xml_bytes = await mndot_client.fetch_metro_config()
            self._parse(xml_bytes)
            logger.info(
                "Loaded %d cameras, %d detectors", len(self.cameras), len(self.detectors)
            )
        except Exception:
            logger.exception("Failed to load metro config")

    def _parse(self, xml_bytes: bytes):
        root = ET.fromstring(xml_bytes)

        cameras: dict[str, CameraInfo] = {}
        all_detectors: list[DetectorInfo] = []
        detector_map: dict[str, DetectorInfo] = {}
        for corridor in root.iter("corridor"):
            corridor_name = corridor.get("route", "") or corridor.get("name", "")
            for r_node in corridor.iter("r_node"):
                lat_s = r_node.get("lat")
                lon_s = r_node.get("lon")
                if not lat_s or not lon_s:
                    continue
                lat, lon = float(lat_s), float(lon_s)

                for det in r_node.iter("detector"):
                    det_name = det.get("name", "")
                    if not det_name:
                        continue
                    det_info = DetectorInfo(
                        id=det_name,
                        label=det.get("label", det_name),
                        lat=lat,
                        lon=lon,
                        lane=det.get("lane", ""),
                        corridor=corridor_name,
                        distance_m=0.0,
                    )
                    all_detectors.append(det_info)
                    detector_map[det_name] = det_info

        # Cameras
        for cam_elem in root.iter("camera"):
            cam_name = cam_elem.get("name", "")
            if not cam_name:
                continue
            lat_s = cam_elem.get("lat")
            lon_s = cam_elem.get("lon")
            if not lat_s or not lon_s:
                continue
            cameras[cam_name] = CameraInfo(
                id=cam_name,
                name=cam_elem.get("label", cam_name),
                lat=float(lat_s),
                lon=float(lon_s),
            )

        # Match cameras to nearby detectors
        camera_detectors: dict[str, list[DetectorInfo]] = {}
        for cam_id, cam in cameras.items():
            nearby = []
            for det in all_detectors:
                d = _haversine(cam.lat, cam.lon, det.lat, det.lon)
                if d <= settings.max_match_distance_m:
                    nearby.append(det.model_copy(update={"distance_m": round(d, 1)}))
            nearby.sort(key=lambda x: x.distance_m)
            camera_detectors[cam_id] = nearby

        self.cameras = cameras
        self.detectors = detector_map
        self.camera_detectors = camera_detectors

    def get_all_cameras(self) -> list[CameraInfo]:
        return list(self.cameras.values())

    def get_camera_with_detectors(self, camera_id: str) -> CameraWithDetectors | None:
        cam = self.cameras.get(camera_id)
        if cam is None:
            return None
        detectors = self.camera_detectors.get(camera_id, [])
        return CameraWithDetectors(camera=cam, detectors=detectors)


metro_config_service = MetroConfigService()
