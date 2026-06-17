"""A tiny thread-safe synchronous event bus.

This is used for *intra-process* notifications that are not tied to Qt
(so the engine can be exercised in headless tests). The GUI does not rely
on the bus for high-rate data -- it polls :class:`~neurobci.core.app_state.
AppState` snapshots via a timer instead -- but the bus is handy for
discrete events (mode changed, stream lost, recording started...).

Handlers are called synchronously on the publisher's thread, so they must
be cheap and must not raise (exceptions are caught and logged).
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Any, Callable

logger = logging.getLogger(__name__)

Handler = Callable[[Any], None]

# Well-known event topics (use constants to avoid typos).
EVT_MODE_CHANGED = "mode_changed"
EVT_CONNECTION_CHANGED = "connection_changed"
EVT_STREAM_INFO = "stream_info"
EVT_ERROR = "error"
EVT_RECORDING_CHANGED = "recording_changed"


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Handler]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, topic: str, handler: Handler) -> None:
        with self._lock:
            self._subs[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Handler) -> None:
        with self._lock:
            if handler in self._subs.get(topic, []):
                self._subs[topic].remove(handler)

    def publish(self, topic: str, payload: Any = None) -> None:
        with self._lock:
            handlers = list(self._subs.get(topic, []))
        for h in handlers:
            try:
                h(payload)
            except Exception:  # noqa: BLE001 - never let a subscriber break the bus
                logger.exception("Event handler for %r failed.", topic)
