"""Background acquisition loop.

A single daemon thread owns the *only* write access to the ring buffer.
It pulls from the active source as fast as ``pull_interval`` allows,
writes samples, and keeps :class:`AppState` up to date (effective sample
rate, last timestamp, dropped-sample estimate, connection health).

Robustness is a first-class concern: a read raising an exception must not
silently kill acquisition. It is logged, the connection is marked
``LOST``/``STALE``, and the loop keeps trying.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from neurobci.acquisition.base import EEGSource
from neurobci.core.app_state import AppState, ConnectionStatus
from neurobci.core.events import EVT_CONNECTION_CHANGED, EVT_ERROR, EventBus
from neurobci.core.ring_buffer import RingBuffer

logger = logging.getLogger(__name__)


class AcquisitionThread:
    def __init__(
        self,
        source: EEGSource,
        ring_buffer: RingBuffer,
        app_state: AppState,
        event_bus: EventBus,
        pull_interval_s: float = 0.02,
        stale_after_s: float = 1.0,
    ) -> None:
        self._source = source
        self._buffer = ring_buffer
        self._state = app_state
        self._bus = event_bus
        self._pull_interval = pull_interval_s
        self._stale_after = stale_after_s

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        # Optional session recorder; written to from this thread only.
        self._recorder = None
        self._recorder_lock = threading.Lock()

        # Effective-rate measurement window.
        self._rate_count = 0
        self._rate_t0 = 0.0
        self._last_data_wall = 0.0
        self._last_ts = float("nan")
        self._simulated = source.info.source_kind == "simulated"

    def set_recorder(self, recorder) -> None:
        """Attach (or detach with ``None``) a session recorder."""
        with self._recorder_lock:
            self._recorder = recorder

    # ----- lifecycle ----------------------------------------------------- #

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._source.start()
        now = time.time()
        self._rate_t0 = now
        self._last_data_wall = now
        self._thread = threading.Thread(
            target=self._run, name="acquisition", daemon=True
        )
        self._thread.start()
        logger.info("Acquisition thread started.")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            self._source.stop()
        except Exception:  # noqa: BLE001
            logger.debug("Source stop() raised.", exc_info=True)
        logger.info("Acquisition thread stopped.")

    # ----- main loop ----------------------------------------------------- #

    def _run(self) -> None:
        sfreq = self._source.info.sfreq
        while not self._stop.is_set():
            loop_start = time.perf_counter()
            try:
                data, ts = self._source.read()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Source read failed.")
                self._bus.publish(EVT_ERROR, str(exc))
                self._set_connection(ConnectionStatus.LOST)
                self._sleep_remaining(loop_start)
                continue

            now = time.time()
            if data.shape[0] > 0:
                self._handle_data(data, ts, now, sfreq)
            else:
                self._maybe_mark_stale(now)

            self._sleep_remaining(loop_start)

    def _handle_data(self, data, ts, now, sfreq) -> None:
        self._buffer.append(data, ts)

        # Persist to the active recording, if any. A recorder failure must
        # never stop acquisition.
        with self._recorder_lock:
            recorder = self._recorder
        if recorder is not None:
            try:
                recorder.write(data, ts)
            except Exception:  # noqa: BLE001
                logger.exception("Recorder write failed; detaching recorder.")
                self.set_recorder(None)

        # Drop-sample estimate from timestamp gaps (LSL only; sim is contiguous).
        dropped = 0
        if not self._simulated and np.isfinite(self._last_ts) and ts.size:
            gap = ts[0] - self._last_ts
            expected = 1.0 / sfreq
            if gap > expected * 1.5:
                dropped = int(round(gap * sfreq)) - 1
        if ts.size:
            self._last_ts = float(ts[-1])

        # Effective sample-rate over a ~1 s window.
        self._rate_count += data.shape[0]
        elapsed = now - self._rate_t0
        measured = self._rate_count / elapsed if elapsed > 0 else 0.0
        if elapsed >= 1.0:
            self._rate_count = 0
            self._rate_t0 = now

        self._last_data_wall = now
        prev = self._state.connection
        new_status = (
            ConnectionStatus.SIMULATED if self._simulated else ConnectionStatus.CONNECTED
        )
        self._state.update(
            samples_received=self._buffer.total_written,
            measured_sfreq=measured,
            last_timestamp=self._last_ts,
            dropped_samples=self._state.snapshot().dropped_samples + max(dropped, 0),
            connection=new_status,
        )
        if prev != new_status:
            self._bus.publish(EVT_CONNECTION_CHANGED, new_status)

    def _maybe_mark_stale(self, now: float) -> None:
        if self._simulated:
            return
        if now - self._last_data_wall > self._stale_after:
            cur = self._state.connection
            if cur == ConnectionStatus.CONNECTED:
                self._set_connection(ConnectionStatus.STALE)

    def _set_connection(self, status: ConnectionStatus) -> None:
        if self._state.connection != status:
            self._state.update(connection=status)
            self._bus.publish(EVT_CONNECTION_CHANGED, status)

    def _sleep_remaining(self, loop_start: float) -> None:
        dt = self._pull_interval - (time.perf_counter() - loop_start)
        if dt > 0:
            self._stop.wait(dt)
