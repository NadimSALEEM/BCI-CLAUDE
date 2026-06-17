"""Crash-safe streaming session recorder.

A *session* is a directory holding raw samples, per-sample timestamps,
markers, and a self-describing metadata document that captures everything
needed to reproduce the recording (full config snapshot, channel layout,
software version, random seed, participant pseudonym, notes).

Design choices:

* **Streaming append** of raw ``float32`` and ``float64`` timestamps means
  a crash loses at most the last unflushed chunk -- the session remains
  readable (``n_samples`` is recoverable from file size).
* **No personally identifying data** is stored: only a caller-supplied
  pseudonym.
* The recorder is written to by the single acquisition thread (EEG chunks)
  and, occasionally, by the UI thread (markers); a lock guards both.

Files inside ``<dir>/<participant>_<timestamp>/``::

    metadata.json     # partial at start, finalised at stop
    eeg.f32           # (n_samples, n_channels) C-order float32, microvolts
    timestamps.f64    # (n_samples,) float64 LSL/source timestamps
    markers.jsonl     # one {"t","label","sample"} object per line
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from neurobci.config.schema import AppConfig
from neurobci.core.stream_info import StreamInfo
from neurobci.version import __version__

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
EEG_FILE = "eeg.f32"
TS_FILE = "timestamps.f64"
MARKERS_FILE = "markers.jsonl"
META_FILE = "metadata.json"


def _sanitize(text: str) -> str:
    safe = "".join(c for c in text if c.isalnum() or c in "-_")
    return safe or "anon"


@dataclass
class RecorderStats:
    n_samples: int
    duration_s: float
    n_markers: int
    bytes_written: int


class SessionRecorder:
    def __init__(
        self,
        output_root: Path | str,
        stream_info: StreamInfo,
        config: AppConfig,
        participant_id: str = "anon",
        notes: str = "",
    ) -> None:
        self._info = stream_info
        self._config = config
        self._participant = _sanitize(participant_id)
        self._notes = notes
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._dir = Path(output_root) / f"{self._participant}_{ts}"

        self._lock = threading.Lock()
        self._eeg_f = None
        self._ts_f = None
        self._markers_f = None
        self._n_samples = 0
        self._n_markers = 0
        self._first_ts: float | None = None
        self._last_ts: float | None = None
        self._started = False
        self._stopped = False

    # ----- properties ---------------------------------------------------- #

    @property
    def path(self) -> Path:
        return self._dir

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def duration_s(self) -> float:
        if self._info.sfreq > 0:
            return self._n_samples / self._info.sfreq
        return 0.0

    def stats(self) -> RecorderStats:
        with self._lock:
            nbytes = self._n_samples * self._info.n_channels * 4
            return RecorderStats(
                n_samples=self._n_samples,
                duration_s=self.duration_s,
                n_markers=self._n_markers,
                bytes_written=nbytes,
            )

    # ----- lifecycle ----------------------------------------------------- #

    def start(self) -> Path:
        if self._started:
            return self._dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._eeg_f = open(self._dir / EEG_FILE, "wb")
        self._ts_f = open(self._dir / TS_FILE, "wb")
        self._markers_f = open(self._dir / MARKERS_FILE, "w", encoding="utf-8")
        self._started = True
        self._write_metadata(final=False)
        logger.info("Recording started: %s", self._dir)
        return self._dir

    def write(self, samples: np.ndarray, timestamps: np.ndarray) -> None:
        """Append a chunk of samples. Called by the acquisition thread."""
        if not self._started or self._stopped:
            return
        if samples.ndim != 2 or samples.shape[1] != self._info.n_channels:
            raise ValueError(
                f"expected (m, {self._info.n_channels}), got {samples.shape}"
            )
        m = samples.shape[0]
        if m == 0:
            return
        with self._lock:
            self._eeg_f.write(np.ascontiguousarray(samples, dtype=np.float32).tobytes())
            self._ts_f.write(np.ascontiguousarray(timestamps, dtype=np.float64).tobytes())
            self._eeg_f.flush()
            self._ts_f.flush()
            if self._first_ts is None and timestamps.size:
                self._first_ts = float(timestamps[0])
            if timestamps.size:
                self._last_ts = float(timestamps[-1])
            self._n_samples += m

    def push_marker(
        self, label: str, timestamp: float | None = None, sample: int | None = None
    ) -> None:
        """Record an event marker. Safe to call from any thread."""
        if not self._started or self._stopped:
            return
        with self._lock:
            t = time.time() if timestamp is None else float(timestamp)
            s = self._n_samples if sample is None else int(sample)
            self._markers_f.write(
                json.dumps({"t": t, "label": str(label), "sample": s}) + "\n"
            )
            self._markers_f.flush()
            self._n_markers += 1

    def stop(self) -> RecorderStats:
        if not self._started or self._stopped:
            return self.stats()
        with self._lock:
            for f in (self._eeg_f, self._ts_f, self._markers_f):
                try:
                    f.flush()
                    f.close()
                except Exception:  # noqa: BLE001
                    logger.debug("Error closing a session file.", exc_info=True)
            self._stopped = True
        self._write_metadata(final=True)
        stats = self.stats()
        logger.info(
            "Recording stopped: %s (%d samples, %.1fs, %d markers)",
            self._dir, stats.n_samples, stats.duration_s, stats.n_markers,
        )
        return stats

    # ----- metadata ------------------------------------------------------ #

    def _write_metadata(self, final: bool) -> None:
        meta = {
            "format_version": FORMAT_VERSION,
            "software_version": __version__,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "participant_id": self._participant,
            "notes": self._notes,
            "stream_name": self._info.name,
            "source_kind": self._info.source_kind,
            "units": self._info.units,
            "sfreq_nominal": self._info.sfreq,
            "n_channels": self._info.n_channels,
            "channel_names": self._info.channel_names,
            "channel_kinds": self._info.channel_kinds,
            "dtype": "float32",
            "files": {
                "eeg": EEG_FILE, "timestamps": TS_FILE, "markers": MARKERS_FILE,
            },
            "config": self._config.to_dict(),
            "finalised": final,
        }
        if final:
            meta.update(
                {
                    "ended_utc": datetime.now(timezone.utc).isoformat(),
                    "n_samples": self._n_samples,
                    "n_markers": self._n_markers,
                    "duration_s": self.duration_s,
                    "first_timestamp": self._first_ts,
                    "last_timestamp": self._last_ts,
                }
            )
        (self._dir / META_FILE).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
