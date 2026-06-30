"""Analysis data manager: replay session -> labelled epoch bundle.

Both the Statistics and Machine-Learning tabs read EEG through this one bridge,
so they see exactly the same epochs, montage and (calibrated) preprocessing the
ERP/Epoch-Average tab does. It reuses the platform's tested epoching core
(:func:`neurobci.bci.erp_analysis.compute_erp`) rather than re-implementing it,
and reuses the *live calibrated* pipeline when it applies to the session (so
fitted ICA/ASR carries into offline analysis).
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field

import numpy as np

from neurobci.bci.erp_analysis import Condition, compute_erp, select_channels
from neurobci.core.stream_info import KIND_EEG, KIND_EOG
from neurobci.paradigms.base import EpochWindow
from neurobci.preprocessing.pipeline import Pipeline

logger = logging.getLogger(__name__)

_RESERVED_MARKER_KEYS = {"label", "sample", "t", "derived"}


@dataclass
class EpochBundle:
    """Condition-labelled epochs plus montage and trial-level metadata.

    ``X`` is ``(n_trials, n_channels, n_times)`` (the platform/MNE/pyriemann
    layout). ``y`` indexes ``condition_names``. ``groups`` identifies the
    subject/session each trial came from (for leave-one-subject/session-out).
    ``metadata`` holds per-trial arrays (onset sample, trial index, and any
    numeric fields the markers carried -- reaction time, score, ...).
    """

    X: np.ndarray
    y: np.ndarray
    condition_names: list[str]
    times: np.ndarray
    sfreq: float
    channel_names: list[str]
    channel_kinds: list[str]
    groups: np.ndarray
    metadata: dict[str, np.ndarray] = field(default_factory=dict)
    preprocessing: str = ""

    # ----- shapes -------------------------------------------------------- #
    @property
    def n_trials(self) -> int:
        return self.X.shape[0]

    @property
    def n_channels(self) -> int:
        return self.X.shape[1]

    @property
    def n_times(self) -> int:
        return self.X.shape[2]

    @property
    def eeg_indices(self) -> list[int]:
        return [i for i, k in enumerate(self.channel_kinds) if k == KIND_EEG]

    # ----- channel/ROI helpers ------------------------------------------ #
    def channel_index(self, name: str) -> int:
        return self.channel_names.index(name)

    def roi_indices(self, names: list[str] | None) -> list[int]:
        """Indices for an ROI; ``None``/empty -> all EEG channels."""
        if not names:
            return self.eeg_indices
        wanted = set(names)
        return [i for i, n in enumerate(self.channel_names) if n in wanted]

    def class_counts(self) -> dict[str, int]:
        u, c = np.unique(self.y, return_counts=True)
        return {self.condition_names[int(k)]: int(v) for k, v in zip(u, c)}

    # ----- selection ----------------------------------------------------- #
    def condition_trials(self, name: str) -> np.ndarray:
        """All epochs (``n, n_ch, n_times``) belonging to one condition."""
        idx = self.condition_names.index(name)
        return self.X[self.y == idx]

    def subset(self, names: list[str]) -> "EpochBundle":
        """A new bundle keeping only ``names`` (re-indexed ``y``)."""
        keep_old = [self.condition_names.index(n) for n in names]
        mask = np.isin(self.y, keep_old)
        remap = {old: new for new, old in enumerate(keep_old)}
        y_new = np.array([remap[int(v)] for v in self.y[mask]], dtype=int)
        meta_new = {k: v[mask] for k, v in self.metadata.items()}
        return dataclasses.replace(
            self, X=self.X[mask], y=y_new, condition_names=list(names),
            groups=self.groups[mask], metadata=meta_new)


# --------------------------------------------------------------------------- #
# Montage adoption + pipeline selection (shared with the ERP tab's behaviour)
# --------------------------------------------------------------------------- #

def adopt_montage(session, engine, config):
    """Relabel a file-loaded session's channels to the curated montage.

    Mirrors the ERP tab: prefer the live Replay stream's selection/names, then
    the configured montage, else keep the file's own names. Returns
    ``(session, note)``.
    """
    n = len(session.channel_names)
    info = getattr(engine, "stream_info", None)
    native = getattr(engine, "native_stream_info", None)
    keep = getattr(engine, "keep_indices", None)

    if (info is not None and keep is not None and native is not None
            and native.n_channels == n):
        session = select_channels(session, keep, info.channel_names,
                                  info.channel_kinds)
        return session, (f"montage from the live Replay stream "
                         f"({len(keep)} of {n} channels)")
    if info is not None and info.n_channels == n:
        session.meta["channel_names"] = list(info.channel_names)
        session.meta["channel_kinds"] = list(info.channel_kinds)
        return session, "names adopted from the live Replay stream"

    cfg = config.channels
    if cfg.n_channels == n:
        names = list(cfg.all_channels)
        eog = set(cfg.eog_channels)
        session.meta["channel_names"] = names
        session.meta["channel_kinds"] = [
            KIND_EOG if x in eog else KIND_EEG for x in names]
        return session, "names adopted from the configured montage"

    return session, "using the file's own channel names (montage count mismatch)"


def offline_pipeline(session, config, engine, *, use_preprocessing: bool):
    """Choose the preprocessing pipeline for offline epoching.

    Returns ``(pipeline_or_None, note)``. Reuses the engine's *calibrated* live
    pipeline when it matches the session montage/rate (so fitted ICA/ASR is
    applied); otherwise rebuilds from config (uncalibrated) and says so.
    """
    if not use_preprocessing:
        return None, "Raw (no preprocessing)"
    live = getattr(engine, "pipeline", None)
    if live is not None and live.requires_fit and live.fitted:
        if live.matches_montage(session.sfreq, session.channel_names):
            return live, "configured pipeline incl. reused calibration (offline)"
        note = ("calibrated stages NOT reused (session montage/rate differs "
                "from the calibrated live stream); rebuilt uncalibrated")
        return _fresh_pipeline(session, config), note
    return _fresh_pipeline(session, config), "configured pipeline (offline)"


def _fresh_pipeline(session, config) -> Pipeline:
    return Pipeline.from_config(
        config.preprocessing, sfreq=session.sfreq,
        ch_kinds=list(session.channel_kinds),
        ch_names=list(session.channel_names))


# --------------------------------------------------------------------------- #
# Bundle construction
# --------------------------------------------------------------------------- #

def _numeric_marker_fields(markers: list[dict]) -> list[str]:
    keys: list[str] = []
    for m in markers:
        for k, v in m.items():
            if k in _RESERVED_MARKER_KEYS or k in keys:
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                keys.append(k)
    return keys


def bundle_from_session(
    session,
    conditions: list[Condition],
    window: EpochWindow,
    *,
    preprocess: Pipeline | None = None,
    reject: bool = True,
    group: int = 0,
) -> EpochBundle:
    """Epoch ``session`` by condition and stack into one labelled bundle.

    Empty conditions are dropped (and reported in ``condition_names`` only if
    they have trials). Trial metadata carries the onset sample, a within-bundle
    trial index, and any numeric fields the matching markers held.
    """
    result = compute_erp(session, conditions, window,
                         preprocess=preprocess, reject=reject)

    # Map onset sample -> marker dict for behavioural metadata lookup.
    extra_keys = _numeric_marker_fields(list(session.markers))
    sample_to_marker: dict[int, dict] = {}
    for m in session.markers:
        sample_to_marker.setdefault(int(m.get("sample", -1)), m)

    Xs, ys, onsets, names = [], [], [], []
    extra = {k: [] for k in extra_keys}
    new_index = 0
    for cond in result.conditions:
        es = cond.epochs
        if es.n_epochs == 0:
            continue
        names.append(cond.name)
        Xs.append(es.X)
        ys.append(np.full(es.n_epochs, new_index, dtype=int))
        onsets.append(np.asarray(es.kept_indices, dtype=int))
        for onset in es.kept_indices:
            mk = sample_to_marker.get(int(onset), {})
            for k in extra_keys:
                v = mk.get(k, np.nan)
                extra[k].append(float(v) if isinstance(v, (int, float)) else np.nan)
        new_index += 1

    if Xs:
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
        onset_arr = np.concatenate(onsets, axis=0)
    else:
        n_ch = len(result.channel_names)
        X = np.empty((0, n_ch, len(result.times)))
        y = np.empty(0, dtype=int)
        onset_arr = np.empty(0, dtype=int)

    metadata = {"onset_sample": onset_arr,
                "trial_index": np.arange(X.shape[0], dtype=int)}
    for k in extra_keys:
        metadata[k] = np.asarray(extra[k], dtype=float)

    return EpochBundle(
        X=X, y=y, condition_names=names, times=result.times, sfreq=result.sfreq,
        channel_names=list(result.channel_names),
        channel_kinds=list(result.channel_kinds),
        groups=np.full(X.shape[0], group, dtype=int),
        metadata=metadata, preprocessing=result.preprocessing)


def concat_bundles(bundles: list[EpochBundle]) -> EpochBundle:
    """Concatenate per-session bundles for multi-subject analysis.

    Bundles must share the montage, time axis and condition set. ``groups`` is
    preserved so leave-one-subject/session-out CV can split on it.
    """
    if not bundles:
        raise ValueError("No bundles to concatenate.")
    ref = bundles[0]
    for b in bundles[1:]:
        if b.condition_names != ref.condition_names:
            raise ValueError("Bundles have different conditions; cannot concat.")
        if b.n_times != ref.n_times or b.channel_names != ref.channel_names:
            raise ValueError("Bundles have different montage/time axis.")
    meta_keys = set().union(*(b.metadata.keys() for b in bundles))
    metadata = {}
    for k in meta_keys:
        cols = [b.metadata.get(k, np.full(b.n_trials, np.nan)) for b in bundles]
        metadata[k] = np.concatenate(cols)
    return dataclasses.replace(
        ref,
        X=np.concatenate([b.X for b in bundles], axis=0),
        y=np.concatenate([b.y for b in bundles], axis=0),
        groups=np.concatenate([b.groups for b in bundles], axis=0),
        metadata=metadata)
