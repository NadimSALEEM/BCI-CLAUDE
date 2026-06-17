"""High-level acquisition facade.

Wires a *source* (chosen from configuration) to a ring buffer, app state
and the acquisition thread. This is the single object the GUI (or a
headless script) talks to in order to start/stop data flow and to read
recent windows for display or processing.
"""

from __future__ import annotations

import logging
import threading
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
from neurobci.preprocessing.pipeline import Pipeline
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
    if kind == "replay":
        from neurobci.acquisition.replay_source import ReplaySource
        if not config.acquisition.replay_path:
            raise ValueError("Replay source requires acquisition.replay_path.")
        return ReplaySource(
            session=config.acquisition.replay_path,
            speed=config.acquisition.replay_speed,
            loop=config.acquisition.replay_loop,
        )
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
        self._processed_buffer: RingBuffer | None = None
        self._pipeline: Pipeline | None = None
        # Guards the live pipeline: the acquisition thread reads it every
        # chunk while the UI may rebuild/edit it. Held only briefly.
        self._pipeline_lock = threading.Lock()
        self._thread: AcquisitionThread | None = None
        self._recorder: SessionRecorder | None = None

    @property
    def buffer(self) -> RingBuffer | None:
        return self._buffer

    @property
    def processed_buffer(self) -> RingBuffer | None:
        """Ring buffer of preprocessed samples (causal pipeline applied)."""
        return self._processed_buffer

    @property
    def pipeline(self) -> Pipeline | None:
        """The live preprocessing pipeline shared with every consumer."""
        return self._pipeline

    @property
    def pipeline_lock(self) -> threading.Lock:
        """Acquire before editing :attr:`pipeline` from another thread."""
        return self._pipeline_lock

    @property
    def stream_info(self) -> StreamInfo | None:
        return self._source.info if self._source else None

    @property
    def source(self) -> EEGSource | None:
        """The active source (e.g. to drive replay transport controls)."""
        return self._source

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
        # A parallel buffer holding the same samples after the causal
        # preprocessing pipeline, so every downstream consumer (quality,
        # spectral, BCI) can work on cleaned data, not raw.
        self._processed_buffer = RingBuffer(capacity=capacity, n_channels=info.n_channels)
        self._pipeline = Pipeline.from_config(
            self.config.preprocessing, info.sfreq, info.channel_kinds, info.channel_names
        )

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
            processed_buffer=self._processed_buffer,
            pipeline=self._pipeline,
            pipeline_lock=self._pipeline_lock,
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
        with self._pipeline_lock:
            self._pipeline = None
        self._processed_buffer = None
        self.state.update(
            mode=OperatingMode.IDLE, connection=ConnectionStatus.DISCONNECTED
        )
        self.bus.publish(EVT_MODE_CHANGED, OperatingMode.IDLE)

    # ----- preprocessing ------------------------------------------------- #

    def rebuild_preprocessing(self) -> Pipeline | None:
        """Rebuild the live pipeline from ``config.preprocessing``.

        Used after the configuration's stage list is replaced wholesale
        (e.g. a profile reset). In-place edits should instead mutate
        :attr:`pipeline` directly while holding :attr:`pipeline_lock`.
        """
        info = self.stream_info
        if info is None:
            return None
        with self._pipeline_lock:
            self._pipeline = Pipeline.from_config(
                self.config.preprocessing, info.sfreq,
                info.channel_kinds, info.channel_names,
            )
            if self._thread is not None:
                self._thread.set_pipeline(self._pipeline)
        return self._pipeline

    def calibrate_artifacts(self, seconds: float = 10.0) -> list[str]:
        """Fit the pipeline's calibrated stages (ICA / ASR / bad-channel) on a
        recent window of *raw* data. Returns a per-stage summary.

        Calibrate on a stretch you believe is relatively clean (e.g. resting,
        eyes open). Fitting runs off the acquisition thread; the freshly-fitted
        pipeline is swapped in under the lock.
        """
        if self._pipeline is None:
            return ["No pipeline (acquisition not running)."]
        info = self.stream_info
        if info is None:
            return ["No stream."]
        data, _ = self.latest(int(seconds * info.sfreq))
        if data.shape[0] < int(0.5 * info.sfreq):
            return ["Not enough data buffered yet to calibrate."]
        with self._pipeline_lock:
            summaries = self._pipeline.fit(data)
            self._pipeline.reset()
        return summaries or ["Nothing to calibrate (no ICA/ASR/bad-channel stage enabled)."]

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

    def latest_processed(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Most recent preprocessed samples (falls back to raw if absent)."""
        if self._processed_buffer is None:
            return self.latest(n_samples)
        return self._processed_buffer.latest(n_samples)

    def latest_processed_seconds(self, seconds: float) -> tuple[np.ndarray, np.ndarray]:
        info = self.stream_info
        if info is None:
            return (np.empty((0, 0)), np.empty(0))
        return self.latest_processed(int(seconds * info.sfreq))
