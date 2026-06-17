"""Application-wide observable state.

A single :class:`AppState` instance is shared between the engine (writer,
mostly the acquisition thread) and the GUI (reader, via a timer). All
access goes through a lock so reads are consistent. The GUI never mutates
acquisition state directly; it asks the engine, which updates here.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum

from neurobci.core.stream_info import StreamInfo


class OperatingMode(str, Enum):
    IDLE = "idle"
    LIVE = "live"               # LSL acquisition
    SIMULATION = "simulation"
    REPLAY = "replay"
    OFFLINE = "offline"


class ConnectionStatus(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"     # receiving stable data
    STALE = "stale"             # connected but data delayed/incomplete
    LOST = "lost"               # was connected, stream dropped
    SIMULATED = "simulated"     # synthetic data (clearly flagged)
    REPLAYED = "replayed"


@dataclass
class _StateData:
    mode: OperatingMode = OperatingMode.IDLE
    connection: ConnectionStatus = ConnectionStatus.DISCONNECTED
    recording: bool = False
    stream_info: StreamInfo | None = None
    samples_received: int = 0
    measured_sfreq: float = 0.0
    last_timestamp: float = 0.0
    dropped_samples: int = 0
    status_message: str = ""
    # Phase-1 placeholders kept visible in the UI status line; later phases
    # fill these in:
    paradigm: str = "none"
    model_name: str = "none"
    last_prediction: str = "-"
    last_confidence: float = 0.0
    emergency_stop: bool = False


class AppState:
    """Thread-safe wrapper around :class:`_StateData`."""

    def __init__(self) -> None:
        self._d = _StateData()
        self._lock = threading.Lock()

    def snapshot(self) -> _StateData:
        """Return a shallow copy safe to read without the lock."""
        with self._lock:
            return _StateData(**vars(self._d))

    def update(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                if not hasattr(self._d, key):
                    raise AttributeError(f"Unknown state field: {key}")
                setattr(self._d, key, value)

    # Convenience typed getters used in hot paths (avoid full snapshot copy).
    @property
    def connection(self) -> ConnectionStatus:
        with self._lock:
            return self._d.connection

    @property
    def mode(self) -> OperatingMode:
        with self._lock:
            return self._d.mode

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._d.recording
