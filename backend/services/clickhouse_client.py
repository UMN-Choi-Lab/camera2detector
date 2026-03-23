"""ClickHouse client for historical detector data queries."""

import asyncio
import logging
from datetime import date, datetime

from backend.config import settings

logger = logging.getLogger(__name__)


class ClickHouseClient:
    def __init__(self):
        self._client = None

    def _get_client(self):
        """Lazy-init clickhouse-connect client."""
        if self._client is None:
            import clickhouse_connect
            self._client = clickhouse_connect.get_client(
                host=settings.clickhouse_host,
                port=settings.clickhouse_port,
                database=settings.clickhouse_database,
            )
        return self._client

    async def query_timeseries(
        self, sensor_id: str, sensor_type: str, start: str, end: str,
    ) -> list[dict]:
        """Query raw_30s time series data for a sensor.

        Args:
            sensor_id: e.g. "1234"
            sensor_type: e.g. "v30", "o30", "c30", "s30"
            start: ISO date string "2026-01-01"
            end: ISO date string "2026-01-31"
        Returns:
            List of {"ts": ISO string, "value": float}
        """
        def _query():
            client = self._get_client()
            result = client.query(
                "SELECT ts, value FROM raw_30s "
                "WHERE sensor_id = %(sid)s AND sensor_type = %(stype)s "
                "AND ts >= %(start)s AND ts < %(end)s "
                "ORDER BY ts",
                parameters={
                    "sid": sensor_id,
                    "stype": sensor_type,
                    "start": start,
                    "end": end,
                },
            )
            return [
                {"ts": row[0].isoformat() if isinstance(row[0], datetime) else str(row[0]),
                 "value": float(row[1]) if row[1] is not None else None}
                for row in result.result_rows
            ]

        return await asyncio.to_thread(_query)

    async def get_detector_meta(self) -> list[dict]:
        """Get all detector metadata from ClickHouse."""
        def _query():
            client = self._get_client()
            result = client.query(
                "SELECT sensor_id, route, direction, lat, lon, lane "
                "FROM detector_meta ORDER BY sensor_id"
            )
            return [
                {
                    "sensor_id": str(row[0]),
                    "route": row[1] or "",
                    "direction": row[2] or "",
                    "lat": float(row[3]) if row[3] is not None else None,
                    "lon": float(row[4]) if row[4] is not None else None,
                    "lane": row[5] or "",
                }
                for row in result.result_rows
            ]

        return await asyncio.to_thread(_query)

    async def get_daily_metrics(self, sensor_id: str, day: str) -> list[dict]:
        """Get daily aggregated metrics for a sensor.

        Args:
            sensor_id: e.g. "1234"
            day: ISO date string "2026-01-15"
        Returns:
            List of {"metric": str, "sensor_type": str, "value": float}
        """
        def _query():
            client = self._get_client()
            result = client.query(
                "SELECT metric, sensor_type, value FROM daily_metrics "
                "WHERE sensor_id = %(sid)s AND day = %(day)s",
                parameters={"sid": sensor_id, "day": day},
            )
            return [
                {
                    "metric": row[0],
                    "sensor_type": row[1],
                    "value": float(row[2]) if row[2] is not None else None,
                }
                for row in result.result_rows
            ]

        return await asyncio.to_thread(_query)

    async def get_latest_samples(self, sensor_ids: list[str]) -> dict[str, dict]:
        """Get the most recent v30, o30, s30 values for a list of sensors.

        Queries the latest available data (matching current time-of-day on the most recent date).
        Returns dict: sensor_id -> {"volume": float|None, "occupancy": float|None, "speed": float|None}
        """
        if not sensor_ids:
            return {}

        def _query():
            client = self._get_client()
            placeholders = ", ".join(f"'{sid}'" for sid in sensor_ids)
            # Find the max date available, then query same time-of-day +/- 5 min
            result = client.query(
                f"WITH max_date AS ("
                f"  SELECT max(toDate(ts)) AS d FROM raw_30s WHERE sensor_id = '{sensor_ids[0]}'"
                f") "
                f"SELECT sensor_id, sensor_type, argMax(value, ts) AS latest_value "
                f"FROM raw_30s "
                f"WHERE sensor_id IN ({placeholders}) "
                f"AND sensor_type IN ('v30', 'c30', 's30') "
                f"AND toDate(ts) = (SELECT d FROM max_date) "
                f"AND abs(toSecond(ts) + toMinute(ts)*60 + toHour(ts)*3600 "
                f"       - (toSecond(now()) + toMinute(now())*60 + toHour(now())*3600)) < 300 "
                f"GROUP BY sensor_id, sensor_type"
            )
            samples: dict[str, dict] = {}
            type_map = {"v30": "volume", "c30": "occupancy", "s30": "speed"}
            for row in result.result_rows:
                sid = str(row[0])
                stype = row[1]
                val = float(row[2]) if row[2] is not None else None
                if sid not in samples:
                    samples[sid] = {"volume": None, "occupancy": None, "speed": None}
                key = type_map.get(stype)
                if key:
                    samples[sid][key] = val
            return samples

        return await asyncio.to_thread(_query)

    async def query_timeseries_sampled(
        self, sensor_id: str, sensor_type: str, start: str, end: str,
        interval_minutes: int = 5,
    ) -> list[dict]:
        """Query time series with downsampling for large date ranges.

        Groups by interval and returns averaged values.
        """
        def _query():
            client = self._get_client()
            result = client.query(
                "SELECT "
                "  toStartOfInterval(ts, INTERVAL %(imin)s MINUTE) AS bucket, "
                "  avg(value) AS avg_value "
                "FROM raw_30s "
                "WHERE sensor_id = %(sid)s AND sensor_type = %(stype)s "
                "AND ts >= %(start)s AND ts < %(end)s "
                "GROUP BY bucket ORDER BY bucket",
                parameters={
                    "sid": sensor_id,
                    "stype": sensor_type,
                    "start": start,
                    "end": end,
                    "imin": interval_minutes,
                },
            )
            return [
                {"ts": row[0].isoformat() if isinstance(row[0], datetime) else str(row[0]),
                 "value": round(float(row[1]), 2) if row[1] is not None else None}
                for row in result.result_rows
            ]

        return await asyncio.to_thread(_query)


clickhouse_client = ClickHouseClient()
