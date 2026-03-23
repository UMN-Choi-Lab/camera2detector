"""History API: ClickHouse-backed detector time series and metadata."""

import logging

from fastapi import APIRouter, HTTPException, Query

from backend.services.clickhouse_client import clickhouse_client

logger = logging.getLogger(__name__)

router = APIRouter(tags=["history"])


@router.get("/history/detectors")
async def list_detectors():
    """Return all detector metadata from ClickHouse."""
    try:
        return await clickhouse_client.get_detector_meta()
    except Exception as e:
        logger.exception("Failed to query ClickHouse detector metadata")
        raise HTTPException(status_code=502, detail=f"ClickHouse query failed: {e}")


@router.get("/history/{detector_id}")
async def get_timeseries(
    detector_id: str,
    type: str = Query("v30", regex="^(v30|o30|c30|s30)$"),
    start: str = Query(..., description="Start date ISO, e.g. 2026-01-01"),
    end: str = Query(..., description="End date ISO, e.g. 2026-01-31"),
    interval: int = Query(5, description="Downsample interval in minutes (0 for raw)"),
):
    """Return time-series data for a detector from ClickHouse."""
    try:
        if interval > 0:
            data = await clickhouse_client.query_timeseries_sampled(
                detector_id, type, start, end, interval_minutes=interval,
            )
        else:
            data = await clickhouse_client.query_timeseries(
                detector_id, type, start, end,
            )
        return {"detector_id": detector_id, "type": type, "data": data}
    except Exception as e:
        logger.exception("Failed to query ClickHouse timeseries")
        raise HTTPException(status_code=502, detail=f"ClickHouse query failed: {e}")


@router.get("/history/{detector_id}/daily")
async def get_daily_summary(
    detector_id: str,
    date: str = Query(..., description="Date ISO, e.g. 2026-01-15"),
):
    """Return daily aggregated metrics for a detector."""
    try:
        data = await clickhouse_client.get_daily_metrics(detector_id, date)
        return {"detector_id": detector_id, "date": date, "metrics": data}
    except Exception as e:
        logger.exception("Failed to query ClickHouse daily metrics")
        raise HTTPException(status_code=502, detail=f"ClickHouse query failed: {e}")
