"""High-level acquisition facade.

Wires a *source* (chosen from configuration) to a ring buffer, app state
and the acquisition thread. This is the single object the GUI (or a
headless script) talks to in order to start/stop data flow and to read
recent windows for display or processing.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from neurobci.acquisition.acquisition_thread import AcquisitionThread
from neurobci.acquisition.base import EEGSource
from neurobci.acquisition.lsl_source import LSLSource
from neurobci.acquisition.simulated import SimulatedSource
from neurobci.config.schema import AppConfig
from neurobci.core.app_state import AppState, ConnectionStatus, OperatingMode
from neurobci.core.events import (
    EVT_MODE_CHANGED,
    EVT_RECORDING_CHANGED,
    EVT_STREAM_INFO,
    EventBus,
)
from neurobci.core.ring_buffer import RingBuffer
from neurobci.core.stream_info import StreamInfo
from neurobci.recording.writer import SessionRecorder

logger = logging.getLogger(__name__)


def build_source(config: AppConfig) -> EEGSource:
    """Factory: create the source described by the configuration."""

    kind = config.acquisition.source_type.lower()
    if kind == "simulated":
        return SimulatedSource(
            channels=config.channels,
            sim=config.simulation,
            sfreq=config.acquisition.expected_sfreq,
            realtime=True,
        )
    if kind == "lsl":
        return LSLSource(acq=config.acquisition, channels=config.channels)
    raise ValueError(f"Unknown / not-yet-implemented source type: {kind!r}")


_MODE_FOR_SOURCE = {
    "simulated": OperatingMode.SIMULATION,
    "lsl": OperatingMode.LIVE,
    "replay": OperatingMode.REPLAY,
}


class AcquisitionEngine:
    def __init__(
        self,
        config: AppConfig,
        app_state: AppState | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.config = config
        self.state = app_state or AppState()
        self.bus = event_bus or EventBus()
        self._source: EEGSource | None = None
        self._buffer: RingBuffer | None = None
        self._thread: AcquisitionThread | None = None
        self._recorder: SessionRecorder | None = None

    @property
    def buffer(self) -> RingBuffer | None:
        return self._buffer

    @property
    def stream_info(self) -> StreamInfo | None:
        return self._source.info if self._source else None

    @property
    def running(self) -> bool:
        return self._thread is not None

    def start(self) -> None:
        if self.running:
            return
        self._source = build_source(self.config)
        # Start the source first so LSL can populate real stream metadata.
        self._source.start()
        info = self._source.info

        capacity = max(int(self.config.acquisition.buffer_seconds * info.sfreq), 1)
        self._buffer = RingBuffer(capacity=capacity, n_channels=info.n_channels)

        mode = _MODE_FOR_SOURCE.get(
            self.config.acquisition.source_type.lower(), OperatingMode.IDLE
        )
        self.state.update(
            mode=mode,
            stream_info=info,
            connection=ConnectionStatus.CONNECTING,
            samples_received=0,
            dropped_samples=0,
        )
        self.bus.publish(EVT_MODE_CHANGED, mode)
        self.bus.publish(EVT_STREAM_INFO, info)

        self._thread = AcquisitionThread(
            source=self._source,
            ring_buffer=self._buffer,
            app_state=self.state,
            event_bus=self.bus,
            pull_interval_s=self.config.acquisition.pull_interval_s,
        )
        self._thread.start()
        logger.info("Acquisition engine running in %s mode.", mode.value)

    def stop(self) -> None:
        if self._recorder is not None:
            self.stop_recording()
        if self._thread:
            self._thread.stop()
        self._thread = None
        self._source = None
        self.state.update(
            mode=OperatingMode.IDLE, connection=ConnectionStatus.DISCONNECTED
        )
        self.bus.publish(EVT_MODE_CHANGED, OperatingMode.IDLE)

    # ----- recording ----------------------------------------------------- #

    @property
    def recorder(self) -> SessionRecorder | None:
        return self._recorder

    @property
    def is_recording(self) -> bool:
        return self._recorder is not None

    def start_recording(
        self, participant_id: str | None = None, notes: str | None = None
    ) -> SessionRecorder:
        """Begin writing the live stream to a new session directory."""
        if not self.running or self._thread is None or self.stream_info is None:
            raise RuntimeError("Cannot record: acquisition is not running.")
        if self._recorder is not None:
            return self._recorder
        rec_cfg = self.config.recording
        recorder = SessionRecorder(
            output_root=rec_cfg.directory,
            stream_info=self.stream_info,
            config=self.config,
            participant_id=participant_id or rec_cfg.participant_id,
            notes=notes if notes is not None else rec_cfg.notes,
        )
        recorder.start()
        self._thread.set_recorder(recorder)
        self._recorder = recorder
        self.state.update(recording=True)
        self.bus.publish(EVT_RECORDING_CHANGED, True)
        return recorder

    def stop_recording(self):
        """Finalise the active recording and return its stats (or None)."""
        if self._recorder is None:
            return None
        if self._thread is not None:
            self._thread.set_recorder(None)
        stats = self._recorder.stop()
        self._recorder = None
        self.state.update(recording=False)
        self.bus.publish(EVT_RECORDING_CHANGED, False)
        return stats

    def push_marker(self, label: str) -> None:
        """Record an event marker against the live recording (if any)."""
        if self._recorder is None:
            return
        snap = self.state.snapshot()
        ts = snap.last_timestamp
        timestamp = ts if (ts and np.isfinite(ts)) else time.time()
        self._recorder.push_marker(label, timestamp=timestamp)

    # ----- read access for consumers (GUI, quality, processing) --------- #

    def latest(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        if self._buffer is None:
            return (np.empty((0, 0)), np.empty(0))
        return self._buffer.latest(n_samples)

    def latest_seconds(self, seconds: float) -> tuple[np.ndarray, np.ndarray]:
        info = self.stream_info
        if info is None:
            return (np.empty((0, 0)), np.empty(0))
        return self.latest(int(seconds * info.sfreq))
