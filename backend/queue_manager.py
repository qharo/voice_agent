import asyncio
from collections import deque
from metrics import queue_depth, active_sessions

MAX_CONCURRENT = 4  # max sessions processing at once


class QueueManager:
    def __init__(self, max_concurrent: int = MAX_CONCURRENT):
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._waiters: deque[asyncio.Future] = deque()

    @property
    def processing_count(self) -> int:
        return self.max_concurrent - self._semaphore._value

    @property
    def queue_length(self) -> int:
        return len(self._waiters)

    async def acquire(self, notify_position_cb=None) -> None:
        """
        Wait for a processing slot. 
        notify_position_cb(pos) is called when the user's queue position changes.
        """
        if self._semaphore._value > 0:
            # slot immediately available
            await self._semaphore.acquire()
            queue_depth.set(self.queue_length)
            return

        # No slot — join the queue
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self._waiters.append(fut)
        queue_depth.set(self.queue_length)

        # Notify initial position
        pos = list(self._waiters).index(fut) + 1
        if notify_position_cb:
            await notify_position_cb(pos)

        try:
            await fut  # wait until we're at the front
            await self._semaphore.acquire()
        except asyncio.CancelledError:
            self._waiters.discard(fut) if hasattr(self._waiters, 'discard') else None
            if fut in self._waiters:
                self._waiters.remove(fut)
            queue_depth.set(self.queue_length)
            raise
        finally:
            queue_depth.set(self.queue_length)

    def release(self) -> None:
        """Release a processing slot and wake the next waiter."""
        self._semaphore.release()
        # Wake the next waiter in line and notify all others of new position
        if self._waiters:
            next_fut = self._waiters.popleft()
            if not next_fut.done():
                next_fut.set_result(None)
        queue_depth.set(self.queue_length)


# Global instance
queue_manager = QueueManager(max_concurrent=MAX_CONCURRENT)
