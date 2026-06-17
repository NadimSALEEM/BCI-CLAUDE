"""Calibration: turn labelled epochs into a trained, evaluated model.

Provides the full short-calibration flow:

1. obtain labelled epochs (from the synthetic generator, or from a recorded
   session via :func:`epochs_from_recording`),
2. compare candidate models with leakage-free cross-validation,
3. refit the best model on all the data,
4. attach its cross-validated metrics,

and clearly reports class balance / rejected trials so chance-level
calibration is never mistaken for success.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from neurobci.bci.epoching import EpochSet, extract_epochs, onsets_from_timestamps
from neurobci.bci.evaluation import EvaluationResult, compare_models
from neurobci.bci.models import ParadigmModel, make_model
from neurobci.paradigms.base import Paradigm
from neurobci.paradigms.synthetic import make_p300_dataset
from neurobci.paradigms.synthetic_paradigms import (
    make_errp_dataset,
    make_mi_dataset,
    make_ssvep_dataset,
)

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    paradigm: str
    sfreq: float
    channel_names: list[str]
    n_epochs: int
    class_counts: dict[int, int]
    n_rejected: int
    results: dict[str, EvaluationResult]
    best_model_name: str | None
    model: object | None                  # fitted ParadigmModel (best)
    messages: list[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        """True only if the best model is meaningfully above chance."""
        if not self.best_model_name or self.best_model_name not in self.results:
            return False
        return self.results[self.best_model_name].beats_chance


def train_and_select(
    paradigm: Paradigm,
    X: np.ndarray,
    y: np.ndarray,
    channel_names: list[str],
    channel_kinds: list[str],
    sfreq: float,
    model_names: list[str] | None = None,
    n_folds: int = 5,
    groups: np.ndarray | None = None,
    seed: int = 0,
) -> CalibrationResult:
    """Compare models on (X, y), refit the best on all data, report."""

    model_names = model_names or paradigm.model_names
    classes, counts = np.unique(y, return_counts=True)
    class_counts = {int(k): int(v) for k, v in zip(classes, counts)}
    messages: list[str] = []

    if len(classes) < 2 or counts.min() < n_folds:
        messages.append(
            f"Insufficient/imbalanced data for {n_folds}-fold CV "
            f"(class counts {class_counts})."
        )

    results, best = compare_models(
        model_names, X, y, sfreq, paradigm.positive_index, n_folds, groups, seed
    )

    model = None
    if best is not None:
        model = make_model(best, paradigm, channel_names, channel_kinds, sfreq)
        model.fit(X, y)
        r = results[best]
        model.metrics = {
            "balanced_accuracy": r.balanced_accuracy,
            "balanced_accuracy_std": r.balanced_accuracy_std,
            "roc_auc": r.roc_auc,
            "kappa": r.kappa,
            "mcc": r.mcc,
            "n_epochs": int(len(y)),
            "class_counts": class_counts,
        }
        if not r.beats_chance:
            messages.append(
                "Best model is NOT clearly above chance -- calibration data are "
                "insufficient for reliable control. Do not trust online output."
            )

    return CalibrationResult(
        paradigm=paradigm.name,
        sfreq=sfreq,
        channel_names=list(channel_names),
        n_epochs=int(len(y)),
        class_counts=class_counts,
        n_rejected=0,
        results=results,
        best_model_name=best,
        model=model,
        messages=messages,
    )


def _make_synthetic(paradigm, channels, sfreq, n_trials, amp_uv, noise_uv, seed):
    """Dispatch to the right synthetic generator for the paradigm."""
    name = paradigm.name
    if name == "p300":
        return make_p300_dataset(
            n_trials=n_trials, sfreq=sfreq, channels=channels,
            p300_amp_uv=amp_uv, noise_uv=noise_uv, window=paradigm.window, seed=seed)
    if name == "motor_imagery":
        return make_mi_dataset(
            n_trials=n_trials, sfreq=sfreq, channels=channels,
            mu_amp_uv=amp_uv + 2.0, beta_amp_uv=amp_uv * 0.6, noise_uv=noise_uv,
            window=paradigm.window, seed=seed)
    if name == "errp":
        return make_errp_dataset(
            n_trials=n_trials, sfreq=sfreq, channels=channels,
            amp_uv=amp_uv, noise_uv=noise_uv, window=paradigm.window, seed=seed)
    raise ValueError(f"No synthetic generator for paradigm {name!r}")


def run_simulated_calibration(
    paradigm: Paradigm,
    channels,
    sfreq: float = 500.0,
    n_trials: int = 300,
    target_ratio: float = 0.25,          # kept for backward compatibility
    p300_amp_uv: float = 6.0,            # generic signal amplitude (uV)
    noise_uv: float = 4.0,
    model_names: list[str] | None = None,
    n_folds: int = 5,
    seed: int = 0,
) -> CalibrationResult:
    """Generate synthetic epochs for the paradigm, then train + evaluate.

    SSVEP is handled specially (CCA decoder needs no per-class training).
    """

    if paradigm.name == "ssvep":
        return _run_ssvep_calibration(
            paradigm, channels, sfreq, n_trials, p300_amp_uv, noise_uv, seed)

    ds = _make_synthetic(
        paradigm, channels, sfreq, n_trials, p300_amp_uv, noise_uv, seed)
    return train_and_select(
        paradigm, ds.X, ds.y, ds.channel_names, ds.channel_kinds,
        sfreq, model_names, n_folds, seed=seed,
    )


def _run_ssvep_calibration(
    paradigm, channels, sfreq, n_trials, amp_uv, noise_uv, seed
) -> CalibrationResult:
    from neurobci.bci.cca import CCADecoder, evaluate_cca

    freqs = paradigm.frequencies
    n_per = max(n_trials // len(freqs), 4)
    ds = make_ssvep_dataset(
        frequencies=freqs, n_per_class=n_per, sfreq=sfreq, channels=channels,
        amp_uv=amp_uv, noise_uv=noise_uv, n_harmonics=paradigm.n_harmonics,
        window=paradigm.window, seed=seed,
    )
    decoder = CCADecoder(freqs, sfreq, paradigm.n_harmonics)
    rep = evaluate_cca(decoder, ds.X, ds.y)

    import numpy as np
    classes, counts = np.unique(ds.y, return_counts=True)
    evr = EvaluationResult(
        model_name="cca", balanced_accuracy=rep["accuracy"], balanced_accuracy_std=0.0,
        accuracy=rep["accuracy"], roc_auc=float("nan"), kappa=float("nan"),
        mcc=float("nan"), confusion=rep["confusion"], per_fold_balacc=[rep["accuracy"]],
        n_folds=1, class_counts={int(k): int(v) for k, v in zip(classes, counts)},
        majority_accuracy=float(counts.max() / counts.sum()),
        chance_balanced=rep["chance"],
    )
    model = ParadigmModel(
        name="cca", paradigm=paradigm.name, pipeline=decoder,
        channel_names=ds.channel_names, channel_kinds=ds.channel_kinds,
        sfreq=sfreq, window=paradigm.window, positive_index=0,
        metrics={"accuracy": rep["accuracy"], "chance": rep["chance"]},
        trained=True,
    )
    messages = []
    if not evr.beats_chance:
        messages.append("CCA accuracy is not clearly above chance.")
    return CalibrationResult(
        paradigm=paradigm.name, sfreq=sfreq, channel_names=ds.channel_names,
        n_epochs=len(ds.y), class_counts=evr.class_counts, n_rejected=0,
        results={"cca": evr}, best_model_name="cca", model=model, messages=messages,
    )


def epochs_from_recording(
    data: np.ndarray,
    timestamps: np.ndarray,
    markers: list[dict],
    sfreq: float,
    paradigm: Paradigm,
    label_map: dict[str, int],
) -> EpochSet:
    """Build epochs from a recorded session's data + markers.

    ``label_map`` maps marker labels to integer class ids; markers whose
    label is not in the map are ignored.
    """

    onsets_t, labels = [], []
    for mk in markers:
        lab = mk.get("label")
        if lab in label_map:
            onsets_t.append(mk.get("t"))
            labels.append(label_map[lab])
    if not onsets_t:
        return extract_epochs(data, sfreq, np.zeros(0, int), np.zeros(0, int),
                              paradigm.window)
    onsets = onsets_from_timestamps(timestamps, np.array(onsets_t, float))
    return extract_epochs(data, sfreq, onsets, np.array(labels, int), paradigm.window)
