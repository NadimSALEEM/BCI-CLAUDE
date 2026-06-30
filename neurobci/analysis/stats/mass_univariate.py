"""EEG mass-univariate statistics.

Point-by-point tests over time and channel x time, plus cluster-based
permutation testing. Two cluster engines:

* a self-contained 1-D (over time) max-cluster permutation (numpy only, fully
  deterministic given a seed) for an ROI/channel ERP -- the common, defensible
  case from the demo workflow;
* a spatio-temporal cluster test via MNE when a montage is available.

Corrected and uncorrected results are kept separate -- the caller must show
both. Uncorrected maps are exploratory and flagged as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats as st

from neurobci.analysis.shared.deps import require
from neurobci.analysis.shared.validation import (
    Diagnostic, WARNING, uncorrected_mass_test_warning)
from neurobci.analysis.stats.correction import correct


@dataclass
class MassUnivariateResult:
    t_values: np.ndarray            # observed statistic map
    p_values: np.ndarray            # uncorrected p-values (same shape)
    p_corrected: np.ndarray         # corrected p-values
    reject: np.ndarray              # significant after correction (bool)
    correction: str
    alpha: float
    paired: bool
    diagnostics: list = field(default_factory=list)

    @property
    def n_significant_uncorrected(self) -> int:
        return int(np.sum(self.p_values <= self.alpha))

    @property
    def n_significant_corrected(self) -> int:
        return int(np.sum(self.reject))


@dataclass
class Cluster:
    mask: np.ndarray                # boolean, shape of the stat map
    mass: float
    p_value: float

    @property
    def extent(self) -> int:
        return int(self.mask.sum())


@dataclass
class ClusterResult:
    t_values: np.ndarray
    clusters: list                  # list[Cluster], sorted by |mass| desc
    threshold: float
    n_permutations: int
    tail: int
    alpha: float
    paired: bool
    seed: object = None
    engine: str = "builtin-1d"
    diagnostics: list = field(default_factory=list)

    def significant(self) -> list:
        return [c for c in self.clusters if c.p_value <= self.alpha]

    def significance_mask(self) -> np.ndarray:
        mask = np.zeros_like(self.t_values, dtype=bool)
        for c in self.significant():
            mask |= c.mask
        return mask


# --------------------------------------------------------------------------- #
# Point-by-point tests
# --------------------------------------------------------------------------- #

def _point_t(A, B, paired, alternative):
    if paired:
        res = st.ttest_rel(A, B, axis=0, alternative=alternative)
    else:
        res = st.ttest_ind(A, B, axis=0, alternative=alternative)
    return np.asarray(res.statistic, float), np.asarray(res.pvalue, float)


def mass_univariate_test(A, B, *, paired=False, alternative="two-sided",
                         correction="fdr_bh", alpha=0.05):
    """Point-by-point t-test over the trailing axes of ``A``/``B``.

    ``A``/``B`` are ``(n_obs, ...)`` (e.g. ``(n, n_times)`` or
    ``(n, n_channels, n_times)``). Returns a :class:`MassUnivariateResult`.
    """
    A, B = np.asarray(A, float), np.asarray(B, float)
    t, p = _point_t(A, B, paired, alternative)
    reject, p_corr = correct(p, method=correction, alpha=alpha)
    diags: list[Diagnostic] = []
    if correction == "none":
        diags.append(uncorrected_mass_test_warning(int(np.prod(t.shape))))
    return MassUnivariateResult(
        t_values=t, p_values=p, p_corrected=p_corr, reject=reject,
        correction=correction, alpha=alpha, paired=paired, diagnostics=diags)


# --------------------------------------------------------------------------- #
# Built-in 1-D cluster permutation (over time)
# --------------------------------------------------------------------------- #

def _t_map_paired(D):
    n = D.shape[0]
    m = D.mean(0)
    s = D.std(0, ddof=1)
    se = s / np.sqrt(n)
    return np.where(se > 0, m / se, 0.0)


def _t_map_independent(A, B):
    res = st.ttest_ind(A, B, axis=0)
    return np.asarray(res.statistic, float)


def _find_clusters_1d(stat, threshold, tail):
    if tail == 1:
        supra = stat > threshold
    elif tail == -1:
        supra = stat < -threshold
    else:
        supra = np.abs(stat) > threshold
    clusters = []
    i = 0
    n = stat.size
    while i < n:
        if supra[i]:
            j = i
            while j < n and supra[j]:
                j += 1
            mask = np.zeros(n, dtype=bool)
            mask[i:j] = True
            clusters.append((mask, float(stat[i:j].sum())))
            i = j
        else:
            i += 1
    return clusters


def cluster_permutation_time(
    A, B, *, paired=False, cluster_alpha=0.05, n_permutations=1000,
    tail=0, seed=None, alpha=0.05,
):
    """1-D (time) max-cluster permutation test on two conditions.

    ``A``/``B`` are ``(n_obs, n_times)``. For ``paired`` the within-pair
    difference is sign-flipped; otherwise group labels are permuted. Returns a
    :class:`ClusterResult` with per-cluster permutation p-values.
    """
    A, B = np.asarray(A, float), np.asarray(B, float)
    rng = np.random.default_rng(seed)

    if paired:
        if A.shape[0] != B.shape[0]:
            raise ValueError("Paired cluster test needs equal-length samples.")
        D = A - B
        n = D.shape[0]
        df = n - 1
        observed = _t_map_paired(D)
    else:
        df = A.shape[0] + B.shape[0] - 2
        observed = _t_map_independent(A, B)

    # Cluster-forming threshold from a two-sided point alpha.
    thr = float(st.t.ppf(1 - cluster_alpha / 2.0, df))

    obs_clusters = _find_clusters_1d(observed, thr, tail)
    if not obs_clusters:
        return ClusterResult(
            t_values=observed, clusters=[], threshold=thr,
            n_permutations=n_permutations, tail=tail, alpha=alpha,
            paired=paired, seed=seed,
            diagnostics=[Diagnostic("info", "No supra-threshold time clusters.")])

    # Null distribution of the maximum cluster mass (by magnitude).
    max_null = np.empty(n_permutations)
    if paired:
        for k in range(n_permutations):
            signs = rng.choice([-1.0, 1.0], size=(n, 1))
            tmap = _t_map_paired(signs * D)
            cl = _find_clusters_1d(tmap, thr, tail)
            max_null[k] = max((abs(m) for _, m in cl), default=0.0)
    else:
        pooled = np.concatenate([A, B], axis=0)
        na = A.shape[0]
        for k in range(n_permutations):
            perm = rng.permutation(pooled.shape[0])
            ap, bp = pooled[perm[:na]], pooled[perm[na:]]
            tmap = _t_map_independent(ap, bp)
            cl = _find_clusters_1d(tmap, thr, tail)
            max_null[k] = max((abs(m) for _, m in cl), default=0.0)

    clusters = []
    for mask, mass in obs_clusters:
        p = (np.sum(max_null >= abs(mass)) + 1) / (n_permutations + 1)
        clusters.append(Cluster(mask=mask, mass=mass, p_value=float(p)))
    clusters.sort(key=lambda c: abs(c.mass), reverse=True)

    diags = []
    if not any(c.p_value <= alpha for c in clusters):
        diags.append(Diagnostic("info", "No cluster survived permutation correction."))
    return ClusterResult(
        t_values=observed, clusters=clusters, threshold=thr,
        n_permutations=n_permutations, tail=tail, alpha=alpha, paired=paired,
        seed=seed, diagnostics=diags)


# --------------------------------------------------------------------------- #
# Spatio-temporal cluster permutation via MNE (optional)
# --------------------------------------------------------------------------- #

def build_channel_adjacency(channel_names: list[str]):
    """Channel adjacency from a standard 10-20 montage (needs MNE)."""
    mne = require("mne")
    info = mne.create_info(list(channel_names), sfreq=1.0, ch_types="eeg")
    try:
        info.set_montage("standard_1020", match_case=False, on_missing="ignore")
    except Exception:  # noqa: BLE001
        pass
    adjacency, _ = mne.channels.find_ch_adjacency(info, ch_type="eeg")
    return adjacency


def spatiotemporal_cluster(
    A, B, channel_names, *, paired=False, n_permutations=1000, tail=0,
    threshold=None, seed=None, alpha=0.05,
):
    """Channel x time cluster permutation using MNE.

    ``A``/``B`` are ``(n_obs, n_channels, n_times)``. MNE expects
    ``(n_obs, n_times, n_channels)``; we transpose internally. Requires MNE and
    a recognisable montage for spatial adjacency.
    """
    mne = require("mne")
    A = np.asarray(A, float).transpose(0, 2, 1)       # -> (n, n_times, n_ch)
    B = np.asarray(B, float).transpose(0, 2, 1)
    adjacency = build_channel_adjacency(channel_names)

    if paired:
        from mne.stats import permutation_cluster_1samp_test as ptest
        T_obs, clusters, cluster_pv, _ = ptest(
            A - B, n_permutations=n_permutations, tail=tail,
            threshold=threshold, adjacency=adjacency, seed=seed, out_type="mask")
    else:
        from mne.stats import permutation_cluster_test as ptest
        T_obs, clusters, cluster_pv, _ = ptest(
            [A, B], n_permutations=n_permutations, tail=tail,
            threshold=threshold, adjacency=adjacency, seed=seed, out_type="mask")

    objs = [Cluster(mask=np.asarray(m).T, mass=float(T_obs.T[np.asarray(m).T].sum()),
                    p_value=float(pv))
            for m, pv in zip(clusters, cluster_pv)]
    objs.sort(key=lambda c: abs(c.mass), reverse=True)
    diags = []
    if not any(c.p_value <= alpha for c in objs):
        diags.append(Diagnostic(WARNING, "No spatio-temporal cluster survived."))
    return ClusterResult(
        t_values=np.asarray(T_obs).T, clusters=objs, threshold=float(threshold or 0.0),
        n_permutations=n_permutations, tail=tail, alpha=alpha, paired=paired,
        seed=seed, engine="mne-spatiotemporal", diagnostics=diags)
