"""ClearGuide (Iteris) API client for speed data comparison."""

import logging
import math
import os
import secrets
import time
import urllib.parse
from base64 import urlsafe_b64encode
from hashlib import sha256
from urllib.parse import urlencode, urljoin

import httpx

logger = logging.getLogger(__name__)

# Known I-94 seed link in ClearGuide
_I94_SEED_LINK_ID = -791212264  # I-94 (W) near downtown Minneapolis
_I94_NETWORK_ID = "S251R2"
_MAX_CORRIDOR_WALK = 300  # max links to walk per direction


def _haversine(lat1, lon1, lat2, lon2):
    """Haversine distance in meters between two lat/lon points."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class ClearGuideClient:
    """ClearGuide OAuth2 client for fetching speed data."""

    CLIENT_ID = "CaSCWF6viMWFg67UisSY99hHQFQpiEmQRxFc2NVv"
    AUTH_BASE = "https://auth.iteris.com/"
    API_BASE = "https://api.iteris-clearguide.com/v1"
    GEO_BASE = "https://geo.iteris-clearguide.com/v1"
    CUSTOMER_KEY = "mndot"
    DEPLOYMENT = "mndot"

    def __init__(self):
        self._id_token = ""
        self._refresh_token = ""
        self._token_expiry = 0.0
        # Corridor spatial index: list of {link_id, lat, lon, street_name, direction}
        self._corridor_links: list[dict] = []
        self._corridor_loaded = False

    @property
    def enabled(self) -> bool:
        return bool(os.getenv("ITERIS_USER") and os.getenv("ITERIS_PASS"))

    def _ensure_token(self):
        """Get or refresh OAuth2 token."""
        if self._id_token and time.time() < self._token_expiry - 60:
            return
        self._oauth_flow()

    def _oauth_flow(self):
        """Full OAuth2 PKCE flow."""
        username = os.getenv("ITERIS_USER", "")
        password = os.getenv("ITERIS_PASS", "")
        if not username or not password:
            raise RuntimeError("ITERIS_USER/ITERIS_PASS not configured")

        with httpx.Client(follow_redirects=True, timeout=30) as client:
            # Step 1: Login
            login_url = urljoin(self.AUTH_BASE, "api/login/")
            resp = client.post(login_url, data={"username": username, "password": password})
            if resp.status_code != 200:
                raise RuntimeError(f"ClearGuide login failed: {resp.status_code}")

            # Step 2: Get auth code with PKCE
            chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~"
            verifier = "".join(secrets.choice(chars) for _ in range(100))
            challenge = urlsafe_b64encode(
                sha256(verifier.encode("ascii")).digest()
            ).decode("ascii").strip("=")

            callback = f"https://{self.DEPLOYMENT}.iteris-clearguide.com/"
            auth_url = urljoin(self.AUTH_BASE, "o/authorize/") + "?" + urlencode({
                "response_type": "code",
                "response_mode": "query",
                "client_id": self.CLIENT_ID,
                "redirect_uri": callback,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            })
            resp = client.get(auth_url)
            url_str = str(resp.url)
            if "code=" not in url_str:
                raise RuntimeError(f"ClearGuide auth code failed: {url_str}")

            code = urllib.parse.parse_qs(url_str.split("?")[1])["code"][0]

            # Step 3: Exchange for tokens
            token_url = urljoin(self.AUTH_BASE, "o/token/")
            resp = client.post(token_url, data={
                "client_id": self.CLIENT_ID,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": callback,
                "grant_type": "authorization_code",
            })
            if resp.status_code != 200:
                raise RuntimeError(f"ClearGuide token exchange failed: {resp.status_code}")

            data = resp.json()
            self._id_token = data["id_token"]
            self._refresh_token = data.get("refresh_token", "")
            self._token_expiry = time.time() + data.get("expires_in", 3600)

        logger.info("ClearGuide OAuth2 tokens obtained")

    @property
    def _headers(self):
        return {
            "Authorization": f"Bearer {self._id_token}",
            "accept": "application/json",
            "origin": f"https://{self.DEPLOYMENT}.iteris-clearguide.com",
            "referer": f"https://{self.DEPLOYMENT}.iteris-clearguide.com/",
        }

    def _api_get(self, url: str, timeout: float = 30) -> httpx.Response:
        """GET with automatic token retry on 401."""
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, headers=self._headers)
            if resp.status_code == 401:
                self._id_token = ""
                self._ensure_token()
                resp = client.get(url, headers=self._headers)
            return resp

    # --- Corridor Building ---

    def _fetch_geo_link(self, link_id: int) -> dict | None:
        """Fetch a single link's geo metadata."""
        url = (
            f"{self.GEO_BASE}/links/?customer_key={self.CUSTOMER_KEY}"
            f"&network_id={_I94_NETWORK_ID}&link_id={link_id}"
        )
        try:
            resp = self._api_get(url)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                if results:
                    return results[0]
        except Exception:
            pass
        return None

    def _walk_corridor(self, seed_link_id: int) -> list[dict]:
        """Walk a corridor from a seed link, following upstream/downstream.

        Returns list of {link_id, lat, lon, street_name, direction}.
        """
        self._ensure_token()

        seed = self._fetch_geo_link(seed_link_id)
        if not seed:
            logger.error("Failed to fetch seed link %s", seed_link_id)
            return []

        street_name = seed.get("street_name", "")
        links = []

        def _extract(link_data: dict) -> dict:
            coords = link_data.get("middle_point", {}).get("coordinates", [0, 0])
            return {
                "link_id": link_data["link_id"],
                "lat": coords[1],
                "lon": coords[0],
                "street_name": link_data.get("street_name", ""),
                "direction": link_data.get("roadway_direction", ""),
            }

        # Add seed
        links.append(_extract(seed))

        # Walk downstream
        lid = seed.get("downstream_link", {}).get("link_id")
        visited = {seed_link_id}
        for _ in range(_MAX_CORRIDOR_WALK):
            if not lid or lid in visited:
                break
            visited.add(lid)
            link = self._fetch_geo_link(lid)
            if not link or link.get("street_name") != street_name:
                break
            links.append(_extract(link))
            lid = link.get("downstream_link", {}).get("link_id")

        # Walk upstream
        lid = seed.get("upstream_link", {}).get("link_id")
        for _ in range(_MAX_CORRIDOR_WALK):
            if not lid or lid in visited:
                break
            visited.add(lid)
            link = self._fetch_geo_link(lid)
            if not link or link.get("street_name") != street_name:
                break
            links.append(_extract(link))
            lid = link.get("upstream_link", {}).get("link_id")

        return links

    def build_corridor(self):
        """Build the I-94 corridor spatial index (both directions)."""
        if self._corridor_loaded:
            return

        logger.info("Building I-94 ClearGuide corridor index...")
        self._ensure_token()

        # Walk I-94 W from seed
        wb_links = self._walk_corridor(_I94_SEED_LINK_ID)
        logger.info("I-94 W: found %d links", len(wb_links))

        # Get opposite direction (I-94 E)
        seed = self._fetch_geo_link(_I94_SEED_LINK_ID)
        eb_links = []
        if seed:
            opp_id = seed.get("opposite_side_link", {}).get("link_id")
            if opp_id:
                eb_links = self._walk_corridor(opp_id)
                logger.info("I-94 E: found %d links", len(eb_links))

        self._corridor_links = wb_links + eb_links
        self._corridor_loaded = True
        logger.info("ClearGuide corridor index: %d total links", len(self._corridor_links))

    # --- Speed Data ---

    def get_speed_timeseries(
        self, link_id: int, start_ts: int, end_ts: int, granularity: str = "5min"
    ) -> dict:
        """Fetch speed time series for a ClearGuide link."""
        self._ensure_token()

        params = {
            "customer_key": self.CUSTOMER_KEY,
            "ltt": "true",
            "link_id": link_id,
            "metrics": "avg_speed",
            "s_timestamp": str(start_ts),
            "e_timestamp": str(end_ts),
            "granularity": granularity,
            "dow": "true",
            "tod": "true",
            "holidays": "true",
        }
        url = f"{self.API_BASE}/link/timeseries/?{urlencode(params)}"
        resp = self._api_get(url)
        resp.raise_for_status()
        return resp.json()

    def find_link_near(self, lat: float, lon: float) -> dict | None:
        """Find the nearest I-94 ClearGuide link to a lat/lon coordinate."""
        if not self._corridor_loaded:
            self.build_corridor()

        if not self._corridor_links:
            return None

        best = None
        best_dist = float("inf")
        for link in self._corridor_links:
            dist = _haversine(lat, lon, link["lat"], link["lon"])
            if dist < best_dist:
                best_dist = dist
                best = link

        # Only match within 2km
        if best and best_dist < 2000:
            return {
                "link_id": best["link_id"],
                "link_title": f"{best['street_name']} ({best['direction']})",
                "street_name": best["street_name"],
                "direction": best["direction"],
                "distance_m": round(best_dist),
            }

        return None


clearguide_client = ClearGuideClient()
