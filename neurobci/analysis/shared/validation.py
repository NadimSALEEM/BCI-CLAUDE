"""Defensive data validation and diagnostics.

These helpers never silently drop data: they *report*. Each returns a list of
:class:`Diagnostic` (info / warning / error) that the UI shows and the report
embeds. This is where the suite's scientific-safety rules live (trial-count
floors, class-imbalance, NaNs, subject leakage, uncorrected mass-testing).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INFO = "info"
WARNING = "warning"
ERROR = "error"


@dataclass
class Diagnostic:
    level: str          # INFO | WARNING | ERROR
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.message}"


def class_counts(y: np.ndarray, names: list[str] | None = None) -> dict:
    """Map class label (or name) -> trial count."""
    y = np.asarray(y)
    u, c = np.unique(y, return_counts=True)
    if names is not None:
        return {names[int(k)] if int(k) < len(names) else int(k): int(v)
                for k, v in zip(u, c)}
    return {int(k): int(v) for k, v in zip(u, c)}


def check_finite(X: np.ndarray, what: str = "data") -> list[Diagnostic]:
    X = np.asarray(X, dtype=float)
    out: list[Diagnostic] = []
    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    if n_nan:
        out.append(Diagnostic(WARNING, f"{what} contains {n_nan} NaN value(s)."))
    if n_inf:
        out.append(Diagnostic(WARNING, f"{what} contains {n_inf} non-finite value(s)."))
    return out


def check_sample_size(n: int, *, floor: int = 12, what: str = "observations") -> list[Diagnostic]:
    if n == 0:
        return [Diagnostic(ERROR, f"No {what} available.")]
    if n < floor:
        return [Diagnostic(
            WARNING,
            f"Only {n} {what} (< {floor}). Results are exploratory and "
            f"unstable; treat any p-value or score with caution.")]
    return []


def check_class_balance(
    y: np.ndarray, names: list[str] | None = None, *, min_per_class: int = 8
) -> list[Diagnostic]:
    """Warn on empty/under-populated classes and strong imbalance."""
    counts = class_counts(y, names)
    out: list[Diagnostic] = []
    if len(counts) < 2:
        out.append(Diagnostic(ERROR, "At least two non-empty classes are required."))
        return out
    out.append(Diagnostic(INFO, "Class counts: "
                          + ", ".join(f"{k}={v}" for k, v in counts.items())))
    small = {k: v for k, v in counts.items() if v < min_per_class}
    if small:
        out.append(Diagnostic(
            WARNING,
            "Under-populated class(es) (< %d trials): %s." % (
                min_per_class, ", ".join(f"{k}={v}" for k, v in small.items()))))
    lo, hi = min(counts.values()), max(counts.values())
    if lo > 0 and hi / lo >= 1.5:
        out.append(Diagnostic(
            WARNING,
            f"Class imbalance {hi}:{lo} (ratio {hi / lo:.1f}). Prefer balanced "
            f"accuracy / F1 and a stratified or balanced split."))
    return out


def check_paired(a: np.ndarray, b: np.ndarray) -> list[Diagnostic]:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape[0] != b.shape[0]:
        return [Diagnostic(ERROR,
                f"Paired test needs equal-length samples (got {a.shape[0]} vs "
                f"{b.shape[0]}).")]
    return check_sample_size(a.shape[0], floor=10, what="pairs")


def check_subject_leakage(groups, cv_is_grouped: bool) -> list[Diagnostic]:
    """Cross-subject prediction with a non-grouped CV leaks subjects across folds."""
    if groups is None:
        return []
    n_groups = len(np.unique(np.asarray(groups)))
    if n_groups > 1 and not cv_is_grouped:
        return [Diagnostic(
            WARNING,
            f"{n_groups} subjects/sessions are present but the cross-validation "
            f"is not grouped: the same subject can appear in train and test "
            f"(subject leakage). Use leave-one-subject/session-out for an "
            f"honest cross-subject estimate.")]
    return []


def check_overfit(train_score: float, test_score: float) -> list[Diagnostic]:
    if train_score - test_score > 0.2:
        return [Diagnostic(
            WARNING,
            f"Train score ({train_score:.2f}) far exceeds test score "
            f"({test_score:.2f}): likely overfitting.")]
    return []


def uncorrected_mass_test_warning(n_tests: int) -> Diagnostic:
    return Diagnostic(
        WARNING,
        f"{n_tests} simultaneous tests without multiple-comparison correction: "
        f"expect ~{0.05 * n_tests:.0f} false positives at alpha=0.05. Treat "
        f"uncorrected maps as exploratory only.")


def worst_level(diags: list[Diagnostic]) -> str:
    order = {INFO: 0, WARNING: 1, ERROR: 2}
    return max((d.level for d in diags), key=lambda lv: order[lv], default=INFO)
