"""Async HTTP client for all MnDOT APIs."""

import gzip
from io import BytesIO

import httpx

from backend.config import settings


class MnDOTClient:
    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    async def start(self):
        self._client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)

    async def close(self):
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        assert self._client is not None, "Client not started"
        return self._client

    async def fetch_camera_image(self, camera_id: str) -> bytes:
        url = settings.camera_image_url.format(camera_id=camera_id)
        resp = await self.client.get(url)
        resp.raise_for_status()
        return resp.content

    async def fetch_metro_config(self) -> bytes:
        """Fetch and decompress metro_config.xml.gz, return raw XML bytes."""
        resp = await self.client.get(settings.metro_config_url)
        resp.raise_for_status()
        return gzip.decompress(resp.content)

    async def fetch_mayfly_data(self, detector_id: str, data_type: str = "counts", date: str | None = None) -> list:
        """Fetch 30-second detector data from Mayfly.

        data_type: counts, occupancy, speed
        date: yyyyMMdd format (defaults to today)
        Returns JSON array of 30-second values.
        """
        if date is None:
            from datetime import datetime, timezone, timedelta
            ct = datetime.now(timezone(timedelta(hours=-6)))  # US Central
            date = ct.strftime("%Y%m%d")
        url = f"{settings.mayfly_url}/{data_type}"
        params = {"date": date, "detector": detector_id}
        resp = await self.client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


mndot_client = MnDOTClient()
