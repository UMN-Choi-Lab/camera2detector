"""Manages background HLS stream workers with reference counting."""

import asyncio
import logging

from backend.config import settings
from backend.services.stream_worker import StreamWorker

logger = logging.getLogger(__name__)


class StreamManager:
    """Singleton that manages per-camera StreamWorker lifecycle.

    - subscribe(): starts a worker if none exists, increments ref count
    - unsubscribe(): decrements ref count, stops worker when zero
    - GPU semaphore limits concurrent inference across all cameras
    """

    def __init__(self):
        self._workers: dict[str, StreamWorker] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._ref_counts: dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._gpu_semaphore = asyncio.Semaphore(settings.max_concurrent_streams)

    async def subscribe(self, camera_id: str) -> StreamWorker:
        """Start or reuse a worker for the given camera. Returns the worker."""
        async with self._lock:
            if camera_id not in self._workers:
                worker = StreamWorker(camera_id, self._gpu_semaphore)
                self._workers[camera_id] = worker
                self._ref_counts[camera_id] = 0
                task = asyncio.create_task(
                    worker.run(), name=f"stream-{camera_id}"
                )
                self._tasks[camera_id] = task
                logger.info("Started stream worker for %s", camera_id)

            self._ref_counts[camera_id] += 1
            return self._workers[camera_id]

    async def unsubscribe(self, camera_id: str):
        """Decrement ref count; stop worker if no subscribers remain."""
        async with self._lock:
            if camera_id not in self._ref_counts:
                return
            self._ref_counts[camera_id] -= 1
            if self._ref_counts[camera_id] <= 0:
                await self._stop_worker(camera_id)

    async def _stop_worker(self, camera_id: str):
        """Stop and remove a worker (must be called under lock)."""
        worker = self._workers.pop(camera_id, None)
        task = self._tasks.pop(camera_id, None)
        self._ref_counts.pop(camera_id, None)

        if worker:
            await worker.stop()
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped stream worker for %s", camera_id)

    async def shutdown(self):
        """Stop all workers (called on app shutdown)."""
        async with self._lock:
            camera_ids = list(self._workers.keys())
            for cid in camera_ids:
                await self._stop_worker(cid)
        logger.info("All stream workers stopped")

    def get_status(self) -> dict:
        """Return status of all active workers."""
        return {
            cid: {
                "camera_id": cid,
                "active": True,
                "connected": w.connected,
                "fps": w.fps,
                "frames_processed": w.frames_processed,
                "subscribers": self._ref_counts.get(cid, 0),
                "last_interval": w.latest_result.model_dump() if w.latest_result else None,
                "error": w.error,
            }
            for cid, w in self._workers.items()
        }


stream_manager = StreamManager()
