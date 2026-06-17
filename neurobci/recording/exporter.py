"""Load recorded sessions and export them to scientific formats.

Reading back is robust to an unfinalised (crashed) recording: if the
metadata lacks ``n_samples`` it is recovered from the raw file size. MNE
export sets correct channel types (EEG vs EOG), converts microvolts to
volts, and attaches markers as annotations -- so sessions drop straight
into standard MNE-Python / BIDS workflows.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from neurobci.core.stream_info import KIND_EOG
from neurobci.recording.writer import (
    EEG_FILE,
    MARKERS_FILE,
    META_FILE,
    TS_FILE,
)

logger = logging.getLogger(__name__)


@dataclass
class LoadedSession:
    path: Path
    meta: dict
    data: np.ndarray            # (n_samples, n_channels) float32, microvolts
    timestamps: np.ndarray      # (n_samples,) float64
    markers: list[dict] = field(default_factory=list)

    @property
    def channel_names(self) -> list[str]:
        return self.meta["channel_names"]

    @property
    def channel_kinds(self) -> list[str]:
        return self.meta["channel_kinds"]

    @property
    def sfreq(self) -> float:
        return float(self.meta["sfreq_nominal"])

    @property
    def n_samples(self) -> int:
        return self.data.shape[0]


def load_session(path: Path | str) -> LoadedSession:
    path = Path(path)
    meta = json.loads((path / META_FILE).read_text(encoding="utf-8"))
    n_channels = int(meta["n_channels"])

    raw = np.fromfile(path / EEG_FILE, dtype=np.float32)
    # Recover sample count from file size (crash-safe), trim any partial row.
    n_samples = raw.size // n_channels
    data = raw[: n_samples * n_channels].reshape(n_samples, n_channels)

    ts = np.fromfile(path / TS_FILE, dtype=np.float64)
    ts = ts[:n_samples]
    if ts.size < n_samples:  # pad if timestamps lagged the EEG file
        ts = np.concatenate([ts, np.full(n_samples - ts.size, np.nan)])

    markers: list[dict] = []
    mfile = path / MARKERS_FILE
    if mfile.exists():
        for line in mfile.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    markers.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed marker line.")

    logger.info("Loaded session %s: %d samples, %d markers.",
                path.name, n_samples, len(markers))
    return LoadedSession(path=path, meta=meta, data=data, timestamps=ts, markers=markers)


def to_mne_raw(session: LoadedSession):
    """Build an :class:`mne.io.RawArray` (lazy MNE import)."""

    import mne

    kinds = session.channel_kinds
    ch_types = ["eog" if k == KIND_EOG else "eeg" for k in kinds]
    info = mne.create_info(
        ch_names=list(session.channel_names),
        sfreq=session.sfreq,
        ch_types=ch_types,
    )
    # MNE expects volts; recordings are microvolts -> scale.
    data_v = (session.data.T.astype(np.float64)) * 1e-6
    raw = mne.io.RawArray(data_v, info, verbose="ERROR")

    if session.markers and session.timestamps.size:
        ref = session.timestamps[0]
        if not np.isfinite(ref):
            ref = 0.0
        onsets, descs = [], []
        for mk in session.markers:
            t = mk.get("t")
            if t is None or not np.isfinite(t):
                # Fall back to sample-based onset.
                onset = mk.get("sample", 0) / session.sfreq
            else:
                onset = max(t - ref, 0.0)
            onsets.append(onset)
            descs.append(str(mk.get("label", "")))
        raw.set_annotations(
            mne.Annotations(onset=onsets, duration=[0.0] * len(onsets),
                            description=descs)
        )
    return raw


def export_fif(session: LoadedSession, out_path: Path | str) -> Path:
    """Export to MNE ``.fif`` (Raw). ``out_path`` should end in ``_raw.fif``."""
    out_path = Path(out_path)
    raw = to_mne_raw(session)
    raw.save(out_path, overwrite=True, verbose="ERROR")
    return out_path


def export_npz(session: LoadedSession, out_path: Path | str) -> Path:
    """Export to a portable ``.npz`` (data, timestamps, names, kinds, sfreq)."""
    out_path = Path(out_path)
    np.savez_compressed(
        out_path,
        data=session.data,
        timestamps=session.timestamps,
        channel_names=np.array(session.channel_names),
        channel_kinds=np.array(session.channel_kinds),
        sfreq=session.sfreq,
    )
    return out_path
