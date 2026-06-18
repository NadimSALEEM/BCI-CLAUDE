"""Import externally-recorded sessions (XDF, MNE FIF) as :class:`LoadedSession`.

The native recorder writes a directory (``eeg.f32`` + ``timestamps.f64`` +
``markers.jsonl`` + ``metadata.json``), read by
:func:`neurobci.recording.exporter.load_session`. Real-world EEG is just as
often shipped as a single ``.xdf`` (LabRecorder / LSL) or ``.fif``
(MNE-Python / BIDS) file.

By decoding those into the very same :class:`LoadedSession` the rest of the
platform consumes, an imported file becomes a first-class replay source: it
can drive every workspace and, in particular, be re-published as a virtual
LSL stream exactly like a native recording.

Heavy optional dependencies (``pyxdf``, ``mne``) are imported lazily so the
core install stays slim and an absent reader fails with a clear message only
when that format is actually opened.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from neurobci.core.electrodes import classify_channel
from neurobci.recording.exporter import LoadedSession, load_session

logger = logging.getLogger(__name__)

# Extensions recognised as single-file imports (lower-cased, incl. the dot).
XDF_SUFFIXES = (".xdf", ".xdfz")
FIF_SUFFIXES = (".fif", ".fif.gz")


def load_session_any(path: Path | str) -> LoadedSession:
    """Load a recorded session from any supported source.

    Dispatches on what ``path`` points at:

    * a **directory** (or a ``metadata.json`` inside one) -> native format;
    * a ``.xdf`` / ``.xdfz`` file -> :func:`load_xdf`;
    * a ``.fif`` / ``.fif.gz`` file -> :func:`load_fif`.
    """
    p = Path(path)
    if p.is_dir():
        return load_session(p)
    if not p.exists():
        raise FileNotFoundError(f"No such session path: {path}")
    name = p.name.lower()
    if p.name == "metadata.json":
        return load_session(p.parent)
    if name.endswith(XDF_SUFFIXES):
        return load_xdf(p)
    if name.endswith(FIF_SUFFIXES):
        return load_fif(p)
    raise ValueError(
        f"Unsupported session format: {p.name!r}. Expected a recording "
        f"directory, a .xdf/.xdfz, or a .fif/.fif.gz file."
    )


# --------------------------------------------------------------------------- #
# XDF
# --------------------------------------------------------------------------- #

def load_xdf(path: Path | str, *, eeg_stream: str | None = None) -> LoadedSession:
    """Load an LSL ``.xdf`` recording.

    The numeric EEG stream becomes the session data; any string ("Markers")
    stream is mapped onto sample indices via its LSL timestamps so the
    original event labels resurface during replay. ``eeg_stream`` selects a
    specific data stream by name when a file carries several.
    """
    try:
        import pyxdf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Reading .xdf files requires the 'pyxdf' package (pip install pyxdf)."
        ) from exc

    path = Path(path)
    streams, _header = pyxdf.load_xdf(str(path))
    return _session_from_xdf_streams(streams, path, eeg_stream=eeg_stream)


def _session_from_xdf_streams(
    streams: list, path: Path, *, eeg_stream: str | None = None
) -> LoadedSession:
    """Build a :class:`LoadedSession` from parsed pyxdf streams (pure helper)."""
    eeg = _pick_eeg_stream(streams, eeg_stream)
    if eeg is None:
        raise ValueError(
            f"No numeric data stream found in {path.name!r}"
            + (f" matching name {eeg_stream!r}." if eeg_stream else ".")
        )
    info = eeg["info"]
    ts = np.asarray(eeg["time_stamps"], dtype=np.float64)
    n_samples = ts.shape[0]

    data = np.asarray(eeg["time_series"], dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n_channels = data.shape[1]

    names, kinds, units = _xdf_channels(info, n_channels)
    scale, unit_label = _unit_scale(units[0] if units else None)
    if scale != 1.0:
        data = (data.astype(np.float64) * scale).astype(np.float32)

    sfreq = _xdf_sfreq(info, ts)
    markers = _xdf_markers(streams, ts, sfreq)

    stream_name = _first(info.get("name")) or path.stem
    meta = {
        "source_kind": "replay",
        "imported_from": "xdf",
        "source_file": str(path),
        "stream_name": stream_name,
        "units": unit_label,
        "sfreq_nominal": sfreq,
        "n_channels": n_channels,
        "channel_names": names,
        "channel_kinds": kinds,
        "participant_id": _participant_from_path(path),
    }
    logger.info(
        "Loaded XDF %s: stream %r, %d samples, %d ch @ %.1f Hz, %d markers.",
        path.name, stream_name, n_samples, n_channels, sfreq, len(markers),
    )
    return LoadedSession(path=path, meta=meta, data=data, timestamps=ts,
                         markers=markers)


def _pick_eeg_stream(streams: list, name: str | None) -> dict | None:
    """Choose the data stream to replay.

    Honours an explicit ``name``; otherwise prefers a stream typed ``EEG``,
    falling back to the widest regularly-sampled numeric stream.
    """
    numeric = [s for s in streams if not _is_string_stream(s)]
    if name is not None:
        for s in numeric:
            if (_first(s["info"].get("name")) or "") == name:
                return s
        return None
    typed_eeg = [
        s for s in numeric
        if (_first(s["info"].get("type")) or "").upper() == "EEG"
    ]
    pool = typed_eeg or numeric
    if not pool:
        return None
    # Widest channel count wins; ties broken by sample count.
    return max(pool, key=lambda s: (_channel_count(s), len(s["time_stamps"])))


def _xdf_channels(info: dict, n_channels: int) -> tuple[list[str], list[str], list[str]]:
    names: list[str] = []
    kinds: list[str] = []
    units: list[str] = []
    for ch in _xdf_channel_descs(info):
        label = _first(ch.get("label")) or f"Ch{len(names) + 1}"
        names.append(label)
        kinds.append(classify_channel(label, _first(ch.get("type"))))
        units.append(_first(ch.get("unit")) or "")
    while len(names) < n_channels:
        i = len(names)
        name = f"Ch{i + 1}"
        names.append(name)
        kinds.append(classify_channel(name))
        units.append("")
    return names[:n_channels], kinds[:n_channels], units[:n_channels]


def _xdf_channel_descs(info: dict) -> list[dict]:
    desc = info.get("desc")
    if not desc or not desc[0]:
        return []
    channels = desc[0].get("channels")
    if not channels or not channels[0]:
        return []
    ch = channels[0].get("channel")
    return ch if ch else []


def _xdf_sfreq(info: dict, ts: np.ndarray) -> float:
    sfreq = float(_first(info.get("nominal_srate")) or 0.0)
    if sfreq <= 0 and ts.shape[0] > 1:
        span = float(ts[-1] - ts[0])
        sfreq = (ts.shape[0] - 1) / span if span > 0 else 0.0
    if sfreq <= 0:
        raise ValueError("Could not determine the XDF stream's sampling rate.")
    return sfreq


def _xdf_markers(streams: list, eeg_ts: np.ndarray, sfreq: float) -> list[dict]:
    """Map every string-stream event onto an EEG sample index by timestamp."""
    n = eeg_ts.shape[0]
    t0 = float(eeg_ts[0]) if n else 0.0
    out: list[dict] = []
    for s in streams:
        if not _is_string_stream(s):
            continue
        m_ts = np.asarray(s["time_stamps"], dtype=np.float64)
        series = s["time_series"]
        for i, t in enumerate(m_ts):
            label = _marker_label(series[i])
            if not label:
                continue
            if n:
                sample = int(np.clip(np.searchsorted(eeg_ts, t), 0, n - 1))
            else:
                sample = max(int(round((float(t) - t0) * sfreq)), 0)
            out.append({"label": label, "sample": sample, "t": float(t)})
    out.sort(key=lambda m: m["sample"])
    return out


# --------------------------------------------------------------------------- #
# FIF (MNE)
# --------------------------------------------------------------------------- #

def load_fif(path: Path | str) -> LoadedSession:
    """Load an MNE ``.fif`` raw recording (round-trips :func:`export_fif`)."""
    try:
        import mne
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Reading .fif files requires the 'mne' package (pip install mne)."
        ) from exc

    path = Path(path)
    raw = mne.io.read_raw_fif(str(path), preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])

    # MNE stores volts; the platform works in microvolts.
    data = (raw.get_data().T.astype(np.float64) * 1e6).astype(np.float32)
    n_samples = data.shape[0]

    names = list(raw.ch_names)
    kinds = [classify_channel(n, t)
             for n, t in zip(names, raw.get_channel_types())]
    ts = (raw.first_time + raw.times).astype(np.float64)

    markers: list[dict] = []
    for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
        rel = float(onset) - float(raw.first_time)
        sample = int(np.clip(round(rel * sfreq), 0, max(n_samples - 1, 0)))
        markers.append({
            "label": str(desc),
            "sample": sample,
            "t": float(ts[sample]) if n_samples else float(onset),
        })
    markers.sort(key=lambda m: m["sample"])

    meta = {
        "source_kind": "replay",
        "imported_from": "fif",
        "source_file": str(path),
        "stream_name": path.stem,
        "units": "uV",
        "sfreq_nominal": sfreq,
        "n_channels": len(names),
        "channel_names": names,
        "channel_kinds": kinds,
        "participant_id": _participant_from_path(path),
    }
    logger.info(
        "Loaded FIF %s: %d samples, %d ch @ %.1f Hz, %d markers.",
        path.name, n_samples, len(names), sfreq, len(markers),
    )
    return LoadedSession(path=path, meta=meta, data=data, timestamps=ts,
                         markers=markers)


# --------------------------------------------------------------------------- #
# small shared helpers
# --------------------------------------------------------------------------- #

def _first(value):
    """XDF metadata wraps every scalar in a one-element list."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _is_string_stream(stream: dict) -> bool:
    fmt = (_first(stream["info"].get("channel_format")) or "").lower()
    typ = (_first(stream["info"].get("type")) or "").lower()
    return fmt == "string" or "marker" in typ


def _channel_count(stream: dict) -> int:
    try:
        return int(_first(stream["info"].get("channel_count")) or 0)
    except (TypeError, ValueError):
        return 0


def _marker_label(row) -> str:
    if isinstance(row, (list, tuple, np.ndarray)):
        return str(row[0]) if len(row) else ""
    return str(row)


def _unit_scale(unit: str | None) -> tuple[float, str]:
    """Return ``(factor_to_microvolts, label)`` for a declared channel unit."""
    u = (unit or "").strip().lower()
    if u in ("v", "volt", "volts"):
        return 1e6, "uV"
    if u in ("mv", "millivolt", "millivolts"):
        return 1e3, "uV"
    if u in ("uv", "µv", "microvolt", "microvolts"):
        return 1.0, "uV"
    return 1.0, (unit or "uV")


def _participant_from_path(path: Path) -> str | None:
    """Pull a BIDS ``sub-XXX`` label out of the path, if present."""
    for part in path.parts:
        if part.lower().startswith("sub-"):
            return part[4:]
    return None
