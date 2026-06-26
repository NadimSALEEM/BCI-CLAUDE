"""Offline ERP / epoch-average analysis from a loaded session.

This is the headless core behind the "ERP / Epoch Average" workspace. It takes
a :class:`~neurobci.recording.exporter.LoadedSession` (native, XDF or FIF --
all carry markers as ``{"label", "sample", "t"}``) plus a user-defined mapping
of marker labels to *conditions*, and returns per-condition epoch sets and
averages.

The design rule mirrors the rest of the platform: the *way data are
preprocessed is not changed here*. Either the caller's already-configured
:class:`~neurobci.preprocessing.pipeline.Pipeline` is applied to the whole
recording (run in offline/zero-phase mode), or the raw signal is epoched as-is.
This module never invents a new preprocessing scheme.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from neurobci.bci.epoching import EpochSet, extract_epochs
from neurobci.core.stream_info import KIND_EEG
from neurobci.paradigms.base import EpochWindow
from neurobci.preprocessing.pipeline import MODE_OFFLINE, Pipeline
from neurobci.recording.exporter import LoadedSession


@dataclass
class Condition:
    """A named analysis condition built from one or more marker labels."""

    name: str
    labels: list[str]


@dataclass
class ConditionAverage:
    """The trial-averaged evoked response for one condition."""

    name: str
    labels: list[str]
    average: np.ndarray            # (n_channels, n_times)
    sem: np.ndarray               # (n_channels, n_times) std error of the mean
    epochs: EpochSet
    n_onsets: int                  # markers that matched this condition

    @property
    def n_epochs(self) -> int:
        return self.epochs.n_epochs

    @property
    def n_rejected(self) -> int:
        return self.epochs.n_rejected

    @property
    def n_out_of_bounds(self) -> int:
        return self.epochs.n_out_of_bounds


@dataclass
class ErpResult:
    """Result of an offline ERP analysis over one or more conditions."""

    times: np.ndarray
    sfreq: float
    channel_names: list[str]
    channel_kinds: list[str]
    conditions: list[ConditionAverage]
    preprocessing: str                       # human-readable description
    window: EpochWindow
    warnings: list[str] = field(default_factory=list)

    @property
    def eeg_indices(self) -> list[int]:
        return [i for i, k in enumerate(self.channel_kinds) if k == KIND_EEG]

    def condition(self, name: str) -> ConditionAverage | None:
        for c in self.conditions:
            if c.name == name:
                return c
        return None


def marker_label_counts(markers: list[dict]) -> dict[str, int]:
    """Count how many markers carry each distinct label (sorted by label)."""
    counts = Counter(str(m.get("label", "")) for m in markers)
    return dict(sorted(counts.items()))


def select_channels(
    session: LoadedSession,
    keep_indices: list[int],
    names: list[str],
    kinds: list[str],
) -> LoadedSession:
    """Return a copy of ``session`` keeping only ``keep_indices`` columns.

    This replicates a live channel deletion (made in the Channels tab) on a
    file-loaded session, so an offline ERP analysis uses exactly the same
    montage as the rest of the app. ``names``/``kinds`` label the kept columns
    in ``keep_indices`` order. Markers and timestamps are unchanged.
    """
    keep = [int(i) for i in keep_indices]
    data = np.asarray(session.data)[:, keep]
    meta = dict(session.meta)
    meta["channel_names"] = list(names)
    meta["channel_kinds"] = list(kinds)
    meta["n_channels"] = len(keep)
    return dataclasses.replace(session, data=data, meta=meta)


def combine_markers(
    markers: list[dict], labels: list[str], new_label: str
) -> list[dict]:
    """Create a new derived marker pooling the onsets of ``labels``.

    Returns fresh marker dicts (``{"label": new_label, "sample", "t"}``) at the
    union of every onset carried by any of ``labels`` -- de-duplicated on sample
    and sorted. This lets the analyst define a composite event (e.g. merge
    ``stim/left`` + ``stim/right`` into ``stimulus``) without touching the
    recording. Existing markers are left untouched; the result is meant to be
    appended to the marker list.
    """
    wanted = set(labels)
    seen: set[int] = set()
    out: list[dict] = []
    for m in markers:
        if str(m.get("label", "")) not in wanted:
            continue
        sample = int(m.get("sample", 0))
        if sample in seen:
            continue
        seen.add(sample)
        out.append({"label": new_label, "sample": sample, "t": m.get("t")})
    out.sort(key=lambda m: m["sample"])
    return out


def latency_index(times: np.ndarray, latency_s: float) -> int:
    """Nearest sample index in ``times`` to a requested latency (seconds)."""
    if times.size == 0:
        return 0
    return int(np.argmin(np.abs(times - float(latency_s))))


def global_field_power(average: np.ndarray, eeg_indices: list[int]) -> np.ndarray:
    """Global field power: the spatial standard deviation across EEG channels."""
    if not eeg_indices:
        return np.zeros(average.shape[1])
    return average[eeg_indices, :].std(axis=0)


def compute_erp(
    session: LoadedSession,
    conditions: list[Condition],
    window: EpochWindow,
    *,
    preprocess: Pipeline | None = None,
    reject: bool = True,
) -> ErpResult:
    """Epoch a loaded session by condition and average each condition.

    ``conditions`` maps marker labels to named conditions. Markers whose label
    is not claimed by any condition are ignored. If ``preprocess`` is given,
    the *whole continuous recording* is passed through it in offline mode
    before epoching -- this is the platform's configured pipeline, unchanged --
    otherwise the raw signal is epoched. Returns an :class:`ErpResult`.
    """
    data = np.asarray(session.data, dtype=np.float64)   # (n_samples, n_channels)
    sfreq = session.sfreq
    names = list(session.channel_names)
    kinds = list(session.channel_kinds)

    warnings: list[str] = []
    if preprocess is not None and preprocess.enabled:
        processed = preprocess.apply_window(data, mode=MODE_OFFLINE)
        pre_label = _pipeline_label(preprocess)
        skipped = [st.type_name for st in preprocess.stages
                   if st.enabled and st.requires_fit and not st.fitted]
        if skipped:
            warnings.append(
                "Pass-through (not fitted, no effect): " + ", ".join(skipped))
    else:
        processed = data
        pre_label = "Raw (no preprocessing)"

    # Bucket every marker's sample index under the condition that claims its
    # label, preserving the caller's condition order.
    onsets_by_cond: dict[str, list[int]] = {c.name: [] for c in conditions}
    label_to_cond: dict[str, str] = {}
    for c in conditions:
        for lab in c.labels:
            label_to_cond[lab] = c.name
    for m in session.markers:
        cond = label_to_cond.get(str(m.get("label", "")))
        if cond is not None:
            onsets_by_cond[cond].append(int(m.get("sample", 0)))

    averages: list[ConditionAverage] = []
    for c in conditions:
        onsets = np.array(sorted(onsets_by_cond[c.name]), dtype=int)
        es = extract_epochs(
            processed, sfreq, onsets,
            labels=np.zeros(onsets.shape[0], dtype=int),
            window=window, reject=reject,
        )
        if es.n_epochs:
            average = es.X.mean(axis=0)
            sem = es.X.std(axis=0) / np.sqrt(es.n_epochs)
        else:
            n_times = window.n_times(sfreq)
            average = np.zeros((len(names), n_times))
            sem = np.zeros((len(names), n_times))
        averages.append(ConditionAverage(
            name=c.name, labels=list(c.labels), average=average, sem=sem,
            epochs=es, n_onsets=int(onsets.shape[0]),
        ))

    times = window.tmin + np.arange(window.n_times(sfreq)) / sfreq
    return ErpResult(
        times=times, sfreq=sfreq, channel_names=names, channel_kinds=kinds,
        conditions=averages, preprocessing=pre_label, window=window,
        warnings=warnings,
    )


def _pipeline_label(pipeline: Pipeline) -> str:
    active = [st.type_name for st in pipeline.stages if st.enabled]
    body = ", ".join(active) if active else "no enabled stages"
    return f"Configured pipeline (offline): {body}"
