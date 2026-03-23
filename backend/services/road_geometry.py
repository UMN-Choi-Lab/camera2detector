"""Road geometry service: load shapefile, spatial queries, bearing computation."""

import logging
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from shapely.ops import nearest_points

from backend.config import settings

logger = logging.getLogger(__name__)


class RoadGeometryService:
    def __init__(self):
        self.gdf: gpd.GeoDataFrame | None = None
        self._camera_roads_cache: dict[str, list[dict]] = {}

    def load(self, shapefile_path: str | None = None):
        """Load shapefile and reproject to EPSG:4326 (WGS84)."""
        path = shapefile_path or settings.shapefile_path
        if not Path(path).exists():
            logger.warning("Shapefile not found: %s", path)
            return

        logger.info("Loading road shapefile: %s", path)
        self.gdf = gpd.read_file(path)

        # Reproject from EPSG:26915 (UTM 15N) to EPSG:4326 (WGS84)
        if self.gdf.crs and self.gdf.crs.to_epsg() != 4326:
            self.gdf = self.gdf.to_crs(epsg=4326)

        # Build spatial index (automatically available via gdf.sindex)
        _ = self.gdf.sindex
        logger.info("Loaded %d road segments", len(self.gdf))

    def get_nearby_roads(self, lat: float, lon: float, radius_m: float | None = None) -> list[dict]:
        """Find road segments near a point. Returns list of road info dicts."""
        if self.gdf is None or self.gdf.empty:
            return []

        radius = radius_m or settings.road_search_radius_m
        # Convert radius from meters to approximate degrees (~111km per degree)
        radius_deg = radius / 111_000.0

        pt = Point(lon, lat)
        # Query spatial index with bounding box
        minx, miny, maxx, maxy = (
            lon - radius_deg, lat - radius_deg,
            lon + radius_deg, lat + radius_deg,
        )
        candidates_idx = list(self.gdf.sindex.intersection((minx, miny, maxx, maxy)))
        if not candidates_idx:
            return []

        candidates = self.gdf.iloc[candidates_idx].copy()

        # Compute actual distance in meters (haversine approximation via UTM projection)
        # Reproject point and candidates to UTM for accurate distance
        candidates_utm = candidates.to_crs(epsg=26915)
        pt_utm = gpd.GeoSeries([pt], crs="EPSG:4326").to_crs(epsg=26915).iloc[0]

        candidates_utm["_distance_m"] = candidates_utm.geometry.distance(pt_utm)
        nearby = candidates_utm[candidates_utm["_distance_m"] <= radius].copy()

        if nearby.empty:
            return []

        nearby = nearby.sort_values("_distance_m")

        # Also get the WGS84 geometries for GeoJSON output
        nearby_wgs = candidates.loc[nearby.index]

        results = []
        for idx, row in nearby.iterrows():
            wgs_geom = nearby_wgs.loc[idx, "geometry"]
            # Compute bearing from camera to nearest point on road
            nearest_pt_on_road = nearest_points(pt, wgs_geom)[1]
            bearing = self._compute_bearing(lat, lon, nearest_pt_on_road.y, nearest_pt_on_road.x)

            # Road segment bearing (direction of the road itself)
            road_bearing = self._road_bearing(wgs_geom)

            # Extract route info — handle both naming conventions
            route_name = row.get("ROUTE_NAME", "") or ""
            route_label = row.get("ROUTE_LABE", "") or row.get("ROUTE_LABEL", "") or ""
            traffic_dir = row.get("TRAFFIC_DI", "") or row.get("TRAFFIC_DIRECTION", "") or ""
            cardinal = row.get("ROUTE_CARD", "") or row.get("ROUTE_CARDINAL_DIRECTION", "") or ""

            # Convert geometry to coordinate list for GeoJSON
            coords = []
            if wgs_geom.geom_type == "LineString":
                coords = [[c[0], c[1]] for c in wgs_geom.coords]
            elif wgs_geom.geom_type == "MultiLineString":
                for line in wgs_geom.geoms:
                    coords.extend([[c[0], c[1]] for c in line.coords])

            results.append({
                "route_name": route_name,
                "route_label": route_label,
                "direction": traffic_dir,
                "cardinal": cardinal,
                "bearing_deg": round(road_bearing, 1),
                "distance_m": round(float(row["_distance_m"]), 1),
                "geometry_coords": coords,
            })

        # Deduplicate by route_label + cardinal (same road, same direction)
        seen = set()
        deduped = []
        for r in results:
            key = (r["route_label"], r["cardinal"])
            if key not in seen:
                seen.add(key)
                deduped.append(r)

        return deduped

    def precompute_camera_roads(self, cameras: list[dict]):
        """Precompute nearby roads for all cameras and cache."""
        if self.gdf is None:
            return

        logger.info("Precomputing road mappings for %d cameras", len(cameras))
        for cam in cameras:
            cam_id = cam.get("id", "")
            lat = cam.get("lat", 0)
            lon = cam.get("lon", 0)
            if lat and lon:
                self._camera_roads_cache[cam_id] = self.get_nearby_roads(lat, lon)

        logger.info("Cached road mappings for %d cameras", len(self._camera_roads_cache))

    def get_camera_roads(self, camera_id: str) -> list[dict]:
        """Get cached nearby roads for a camera."""
        return self._camera_roads_cache.get(camera_id, [])

    @staticmethod
    def _compute_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute bearing from point 1 to point 2 in degrees."""
        lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
        lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
        dlon = lon2_r - lon1_r

        x = math.sin(dlon) * math.cos(lat2_r)
        y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)

        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360

    @staticmethod
    def _road_bearing(geom) -> float:
        """Compute the overall bearing of a road segment."""
        if geom.geom_type == "LineString":
            coords = list(geom.coords)
        elif geom.geom_type == "MultiLineString":
            coords = []
            for line in geom.geoms:
                coords.extend(list(line.coords))
        else:
            return 0.0

        if len(coords) < 2:
            return 0.0

        # Use first and last points for overall bearing
        lon1, lat1 = coords[0][0], coords[0][1]
        lon2, lat2 = coords[-1][0], coords[-1][1]

        return RoadGeometryService._compute_bearing(lat1, lon1, lat2, lon2)


road_geometry_service = RoadGeometryService()
