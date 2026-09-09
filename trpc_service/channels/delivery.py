"""Small, safe primitives for external IM delivery.

The Worker execution is deliberately outside this module.  A delivery retry
only repeats an SDK write that the facade proves did not send any bytes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import AsyncIterator


class ChannelSendError(Exception):
    """A fixed SDK-facade send failure.

    ``sent`` is intentionally the only detail.  It prevents retrying a write
    that might already be visible to an IM user and never carries SDK text.
    """

    def __init__(self, *, sent: bool) -> None:
        super().__init__("channel send failed")
        self.sent = sent


async def send_with_retry(send: Callable[[], Awaitable[object]]) -> bool:
    """Try an unsent SDK write up to three times, without swallowing cancel.

    Returns whether a write completed.  A ``sent=True`` error is terminal,
    because a second send could duplicate a user-visible reply.
    """
    for attempt in range(3):
        try:
            await send()
            return True
        except asyncio.CancelledError:
            raise
        except ChannelSendError as exc:
            if exc.sent or attempt == 2:
                return False
            await asyncio.sleep(0.1 * (attempt + 1))
    return False


class ChannelExecutionStream(AsyncIterator[object]):
    """Internal public-event stream carrying one already-created task identity.

    The identity becomes available only after admission and task creation;
    external services use it after their *SDK* terminal write for audit.  It
    is intentionally not part of the Console event schema.
    """

    def __init__(self, iterator: AsyncIterator[object]) -> None:
        self._iterator = iterator
        self._nested_iterator: AsyncIterator[object] | None = None
        self.task: object | None = None
        self.delivery_events: str = "all"

    def __aiter__(self) -> "ChannelExecutionStream":
        return self

    async def __anext__(self) -> object:
        return await anext(self._iterator)

    def set_nested_iterator(self, iterator: AsyncIterator[object] | None) -> None:
        """Register the active Worker stream so consumer cancellation closes it now."""
        self._nested_iterator = iterator

    async def aclose(self) -> None:
        nested = self._nested_iterator
        self._nested_iterator = None
        if nested is not None:
            close_nested = getattr(nested, "aclose", None)
            if close_nested is not None:
                await close_nested()
        close = getattr(self._iterator, "aclose", None)
        if close is not None:
            await close()


__all__ = ["ChannelExecutionStream", "ChannelSendError", "send_with_retry"]
