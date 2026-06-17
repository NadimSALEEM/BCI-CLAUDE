"""Replay a recorded session as an :class:`EEGSource`.

This is the third source backend (after simulated and LSL); because it
implements the same contract, *every* downstream module -- buffering,
visualisation, quality, preprocessing, spectral analysis, calibration --
works on replayed data unchanged.

Transport controls (play/pause, speed, seek, loop, restart) make it useful
both as an interactive tool and for deterministic regression testing
(``realtime=False`` emits fixed blocks with no wall-clock dependence).
Markers from the original session are re-surfaced via :meth:`poll_markers`.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from neurobci.acquisition.base import EEGSource
from neurobci.core.stream_info import StreamInfo
from neurobci.recording.exporter import LoadedSession, load_session


class ReplaySource(EEGSource):
    def __init__(
        self,
        session: LoadedSession | str,
        speed: float = 1.0,
        loop: bool = False,
        realtime: bool = True,
        block_s: float = 0.05,
    ) -> None:
        if not isinstance(session, LoadedSession):
            session = load_session(session)
        self._session = session
        self._data = np.ascontiguousarray(session.data, dtype=np.float32)
        self._sfreq = float(session.sfreq)
        self._n = self._data.shape[0]
        self._info = StreamInfo(
            name=session.meta.get("stream_name", "replay"),
            sfreq=self._sfreq,
            channel_names=session.channel_names,
            channel_kinds=session.channel_kinds,
            source_kind="replay",
            units=session.meta.get("units", "uV"),
        )
        # Markers sorted by sample index.
        self._markers = sorted(
            ({"sample": int(m.get("sample", 0)), "label": str(m.get("label", ""))}
             for m in session.markers),
            key=lambda m: m["sample"],
        )

        self._speed = max(float(speed), 0.01)
        self._loop = bool(loop)
        self._realtime = realtime
        self._block = max(int(block_s * self._sfreq), 1)

        self._lock = threading.Lock()
        self._pos = 0                 # next session sample to emit
        self._global = 0             # total emitted (for timestamps)
        self._accum = 0.0            # fractional sample accumulator (realtime)
        self._paused = False
        self._finished = False
        self._started = False
        self._t0 = 0.0
        self._last = 0.0
        self._pending: list[dict] = []
        self._injector = None        # optional ArtifactInjector overlay

    # ----- EEGSource interface ------------------------------------------ #

    @property
    def info(self) -> StreamInfo:
        return self._info

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._t0 = time.time()
        self._last = self._t0

    def stop(self) -> None:
        self._started = False

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._started or self._finished or self._paused:
            return self._empty()
        if self._realtime:
            now = time.time()
            with self._lock:
                dt = now - self._last
                self._last = now
                self._accum += dt * self._speed * self._sfreq
                n = int(self._accum)
                self._accum -= n
        else:
            n = self._block
        if n <= 0:
            return self._empty()
        return self._emit(min(n, int(self._sfreq)))  # cap a single read at 1 s

    # ----- emission ------------------------------------------------------ #

    def _emit(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            chunks, marks = [], []
            remaining = n
            while remaining > 0 and not self._finished:
                end = min(self._pos + remaining, self._n)
                chunks.append(self._data[self._pos:end])
                self._collect_markers(self._pos, end, marks)
                taken = end - self._pos
                self._pos = end
                remaining -= taken
                if self._pos >= self._n:
                    if self._loop:
                        self._pos = 0
                    else:
                        self._finished = True
            if not chunks:
                return self._empty()
            data = np.concatenate(chunks, axis=0)
            g0 = self._global
            self._global += data.shape[0]
            self._pending.extend(marks)
            injector = self._injector
        if injector is not None:
            data = injector.inject(data, g0)
        ts = self._t0 + (g0 + np.arange(data.shape[0])) / self._sfreq
        return data, ts.astype(np.float64)

    def set_injector(self, injector) -> None:
        """Attach (or detach with ``None``) an artifact overlay."""
        with self._lock:
            self._injector = injector

    def _collect_markers(self, start: int, end: int, out: list) -> None:
        for m in self._markers:
            if start <= m["sample"] < end:
                out.append({"label": m["label"], "sample": m["sample"],
                            "t": time.time()})

    def poll_markers(self) -> list[dict]:
        """Return and clear markers crossed since the last poll."""
        with self._lock:
            out, self._pending = self._pending, []
            return out

    def _empty(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.empty((0, self._info.n_channels), dtype=np.float32),
                np.empty(0, dtype=np.float64))

    # ----- transport ----------------------------------------------------- #

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False
        self._last = time.time()

    @property
    def paused(self) -> bool:
        return self._paused

    def restart(self) -> None:
        with self._lock:
            self._pos = 0
            self._accum = 0.0
            self._finished = False
            self._last = time.time()

    def seek(self, seconds: float) -> None:
        with self._lock:
            self._pos = int(np.clip(seconds * self._sfreq, 0, self._n - 1))
            self._finished = False
            self._last = time.time()

    def set_speed(self, speed: float) -> None:
        self._speed = max(float(speed), 0.01)

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def position_s(self) -> float:
        return self._pos / self._sfreq

    @property
    def duration_s(self) -> float:
        return self._n / self._sfreq

    # ----- convenience for tests ---------------------------------------- #

    def read_all(self) -> tuple[np.ndarray, list[dict]]:
        """Deterministically read the whole (non-looping) session."""
        self.start()
        out = []
        while not self._finished:
            d, _ = self._emit(self._block)
            if d.shape[0]:
                out.append(d)
        return (np.concatenate(out, axis=0) if out else self._empty()[0],
                self.poll_markers())
