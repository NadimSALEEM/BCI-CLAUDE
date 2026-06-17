"""Calibrated artifact-removal stages: bad-channel repair, ICA and ASR.

Unlike the simple DSP stages in :mod:`neurobci.preprocessing.stages`, these
*learn* from data. They follow one strict rule for safety and honesty:

* a stage **passes the signal through unchanged until it has been fitted**, so
  an un-calibrated artifact remover can never silently distort the live stream;
* fitting happens **off the real-time path** (on an explicit calibration
  window) and produces a fixed linear transform that is then applied causally.

This is the "fit on calibration data, apply online" design promised in the
preprocessing docs -- not per-window refitting, which would leak and be
unstable.

Scientific honesty: these are research-grade *helpers*, not magic. ICA here is
FastICA with EOG-/kurtosis-based component selection; ASR is a calibrated
subspace-reconstruction (Mullen et al. 2015 style) and is a *simplified*
implementation of the EEGLAB/`clean_rawdata` method. Always verify the result
on your own data (see :mod:`neurobci.quality.verify`).
"""

from __future__ import annotations

import logging

import numpy as np

from neurobci.core.stream_info import KIND_EEG
from neurobci.preprocessing.stages import ParamSpec, ProcessingStage

logger = logging.getLogger(__name__)


def _eeg_indices(ch_kinds: list[str]) -> np.ndarray:
    return np.array([i for i, k in enumerate(ch_kinds) if k == KIND_EEG], dtype=int)


def _robust_std(x: np.ndarray, axis=0) -> np.ndarray:
    """Median-absolute-deviation std estimate (robust to artifacts)."""
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis)
    return 1.4826 * mad


# --------------------------------------------------------------------------- #
# Bad-channel detection + spatial interpolation
# --------------------------------------------------------------------------- #


class InterpolateBad(ProcessingStage):
    """Detect bad EEG channels and replace them by neighbour interpolation.

    This is the practical fix for a single railing/flat electrode poisoning a
    Common Average Reference: a 2000 uV channel dragged into the CAR mean
    contaminates *every* channel. Calibrate this stage (ideally **before**
    CAR) so the bad electrode is repaired from its neighbours first.

    The bad set is *locked at calibration* (not re-decided every window, which
    would be unstable). Channels without a 10-20 position, and EOG, are left
    untouched.
    """

    type_name = "interpolate_bad"
    requires_fit = True
    PARAM_SPECS = (
        ParamSpec("flat_std_uv", "float", 0.7, "Below this std => flat/disconnected.", 0.0, 50.0),
        ParamSpec("extreme_std_uv", "float", 120.0, "Above this std => gross noise.", 10.0, 5000.0),
        ParamSpec("rail_uv", "float", 200.0, "Amplitude counted as railing (uV).", 10.0, 5000.0),
        ParamSpec("rail_fraction", "float", 0.05, "Railing fraction to flag bad.", 0.0, 1.0),
        ParamSpec("neighbors", "int", 4, "Neighbours used to interpolate.", 1, 8),
    )

    def __init__(self, enabled: bool = True, params: dict | None = None) -> None:
        super().__init__(enabled, params)
        self._W: np.ndarray = np.zeros((0, 0))
        self._bad_idx: np.ndarray = np.zeros(0, dtype=int)
        self.bad_names: list[str] = []

    def fit(self, data: np.ndarray) -> None:
        self.fitted = False
        self._bad_idx = np.zeros(0, dtype=int)
        eeg = _eeg_indices(self._ch_kinds)
        n = len(self._ch_kinds)
        self._W = np.zeros((n, n))
        if data.shape[0] < 16 or eeg.size < 3:
            self.fit_summary = "not enough channels/samples"
            self.fitted = True
            return

        std = np.std(data[:, eeg], axis=0)
        rail = np.mean(np.abs(data[:, eeg]) > self.params["rail_uv"], axis=0)
        bad_mask = (
            (std < self.params["flat_std_uv"])
            | (std > self.params["extreme_std_uv"])
            | (rail > self.params["rail_fraction"])
        )
        bad = eeg[bad_mask]
        good = eeg[~bad_mask]

        from neurobci.spectral.topo import channel_positions_2d

        pos, found = channel_positions_2d(list(self._ch_names)) \
            if self._ch_names and len(self._ch_names) == n else (None, None)

        repaired = []
        if pos is not None and good.size >= 1:
            k = max(1, int(self.params["neighbors"]))
            good_pos = [g for g in good if found[g]]
            for b in bad:
                if pos is None or not found[b] or not good_pos:
                    continue
                gp = np.array(good_pos)
                d = np.linalg.norm(pos[gp] - pos[b], axis=1)
                order = np.argsort(d)[:k]
                sel = gp[order]
                w = 1.0 / np.maximum(d[order], 1e-6)
                self._W[b, sel] = w / w.sum()
                repaired.append(b)
        self._bad_idx = np.array(repaired, dtype=int)
        self.bad_names = [self._ch_names[i] for i in self._bad_idx] if self._ch_names else []
        skipped = [b for b in bad if b not in set(repaired)]
        msg = f"bad: {', '.join(self.bad_names) or 'none'}"
        if skipped:
            msg += f"; {len(skipped)} bad channel(s) had no usable neighbours"
        self.fit_summary = msg
        self.fitted = True

    def apply(self, data: np.ndarray) -> np.ndarray:
        if not self.fitted or self._bad_idx.size == 0 or data.shape[0] == 0:
            return data
        out = data.copy()
        out[:, self._bad_idx] = data @ self._W[self._bad_idx].T
        return out

    def validate(self, sfreq):
        if self.requires_fit and not self.fitted:
            return ["Bad-channel interpolation not calibrated yet "
                    "(passing through). Use 'Calibrate artifact removal'."]
        return []


# --------------------------------------------------------------------------- #
# ICA-based artifact removal (fitted; applied as a fixed linear projection)
# --------------------------------------------------------------------------- #


class ICARemoval(ProcessingStage):
    """Remove ocular/artifact components via FastICA (fit once, apply online).

    Fits FastICA on the EEG channels of a calibration window, flags artifact
    components by (a) correlation with the EOG channel and (b) excess kurtosis
    (spiky, blink-like), zeroes them and stores the resulting fixed linear
    cleaning projection. Online it is a single matrix multiply -- causal and
    cheap. Passes through until fitted.
    """

    type_name = "ica"
    requires_fit = True
    PARAM_SPECS = (
        ParamSpec("n_components", "int", 0, "ICA components (0 = #EEG channels).", 0, 64),
        ParamSpec("eog_threshold", "float", 0.5, "Abs corr. with EOG to flag a component.", 0.1, 1.0),
        ParamSpec("kurtosis_threshold", "float", 8.0, "Excess kurtosis to flag a component.", 2.0, 50.0),
        ParamSpec("max_remove", "int", 3, "Max components removed.", 0, 16),
    )

    def __init__(self, enabled: bool = True, params: dict | None = None) -> None:
        super().__init__(enabled, params)
        self._eeg: np.ndarray = np.zeros(0, dtype=int)
        self._eog: int | None = None
        self._mean: np.ndarray = np.zeros(0)
        self._proj: np.ndarray = np.zeros((0, 0))   # (n_eeg, n_eeg)
        self.removed: int = 0

    def prepare(self, sfreq, ch_kinds, ch_names=None):
        super().prepare(sfreq, ch_kinds, ch_names)
        self._eeg = _eeg_indices(ch_kinds)
        eog = [i for i, k in enumerate(ch_kinds) if k == "eog"]
        self._eog = eog[0] if eog else None

    def fit(self, data: np.ndarray) -> None:
        self.fitted = False
        self.removed = 0
        n_eeg = self._eeg.size
        if data.shape[0] < max(64, 4 * n_eeg) or n_eeg < 3:
            self.fit_summary = "not enough data/channels"
            self._proj = np.eye(n_eeg)
            self._mean = np.zeros(n_eeg)
            self.fitted = True
            return

        from scipy.stats import kurtosis
        from sklearn.decomposition import FastICA

        X = np.asarray(data[:, self._eeg], dtype=np.float64)
        self._mean = X.mean(axis=0)
        k = int(self.params["n_components"]) or n_eeg
        k = min(k, n_eeg)
        ica = FastICA(n_components=k, whiten="unit-variance",
                      max_iter=1000, tol=1e-3, random_state=0)
        S = ica.fit_transform(X)                       # (T, k) sources
        W = ica.components_                            # (k, n_eeg) unmixing
        A = ica.mixing_                               # (n_eeg, k) mixing

        scores = np.zeros(k)
        if self._eog is not None:
            eog_sig = np.asarray(data[:, self._eog], dtype=np.float64)
            eog_sig = eog_sig - eog_sig.mean()
            for j in range(k):
                s = S[:, j] - S[:, j].mean()
                denom = (np.linalg.norm(s) * np.linalg.norm(eog_sig)) + 1e-12
                scores[j] = abs(float(s @ eog_sig) / denom)
        kurt = np.abs(kurtosis(S, axis=0, fisher=True))

        artifact = (scores > self.params["eog_threshold"]) | \
                   (kurt > self.params["kurtosis_threshold"])
        # Rank by how artifactual, cap at max_remove.
        rank = np.argsort(-(scores + kurt / 10.0))
        chosen = [j for j in rank if artifact[j]][: int(self.params["max_remove"])]

        A_clean = A.copy()
        A_clean[:, chosen] = 0.0
        self._proj = A_clean @ W                       # (n_eeg, n_eeg)
        self.removed = len(chosen)
        self.fit_summary = (
            f"removed {self.removed} component(s)"
            + (f" (max EOG corr {scores.max():.2f})" if self._eog is not None else "")
        )
        self.fitted = True

    def apply(self, data: np.ndarray) -> np.ndarray:
        if not self.fitted or self.removed == 0 or data.shape[0] == 0:
            return data
        out = data.copy()
        x = data[:, self._eeg] - self._mean
        out[:, self._eeg] = x @ self._proj.T + self._mean
        return out

    def validate(self, sfreq):
        if self.requires_fit and not self.fitted:
            return ["ICA not calibrated yet (passing through). Use "
                    "'Calibrate artifact removal'."]
        return []


# --------------------------------------------------------------------------- #
# ASR: Artifact Subspace Reconstruction (calibrated, simplified)
# --------------------------------------------------------------------------- #


def _sym_sqrt(c: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(c)
    return (v * np.sqrt(np.clip(w, 0.0, None))) @ v.T


class ASR(ProcessingStage):
    """Artifact Subspace Reconstruction (calibrated; simplified Mullen 2015).

    Calibrate channel statistics on a window of *clean* EEG. Online, for each
    short window it finds directions whose variance exceeds ``cutoff``x the
    calibrated variance in that direction and **reconstructs** them from the
    remaining (clean) directions using the calibration covariance -- so
    high-variance transients (movement, pops) are repaired while normal brain
    activity is preserved. Passes through until fitted.

    This is a faithful-in-spirit but *simplified* implementation. For
    publication-grade ASR use EEGLAB `clean_rawdata` / `asrpy` and validate.
    """

    type_name = "asr"
    requires_fit = True
    PARAM_SPECS = (
        ParamSpec("cutoff", "float", 6.0, "Rejection multiplier (lower = stronger).", 1.5, 50.0),
        ParamSpec("window_ms", "float", 500.0, "Analysis window length (ms).", 100.0, 2000.0),
    )

    def __init__(self, enabled: bool = True, params: dict | None = None) -> None:
        super().__init__(enabled, params)
        self._eeg: np.ndarray = np.zeros(0, dtype=int)
        self._mean: np.ndarray = np.zeros(0)
        self._C0: np.ndarray = np.zeros((0, 0))       # clean covariance
        self._M: np.ndarray = np.zeros((0, 0))        # C0^{1/2} (mixing)
        self._win = 0
        self._buf: np.ndarray | None = None           # rolling buffer (live)

    def prepare(self, sfreq, ch_kinds, ch_names=None):
        super().prepare(sfreq, ch_kinds, ch_names)
        self._eeg = _eeg_indices(ch_kinds)
        self._win = max(16, int(round(self.params["window_ms"] * sfreq / 1000.0)))

    def reset(self) -> None:
        self._buf = None

    def fit(self, data: np.ndarray) -> None:
        self.fitted = False
        n_eeg = self._eeg.size
        if data.shape[0] < max(64, 2 * n_eeg) or n_eeg < 2:
            self.fit_summary = "not enough data/channels"
            self.fitted = False
            return
        X = np.asarray(data[:, self._eeg], dtype=np.float64)
        self._mean = X.mean(axis=0)
        Xc = X - self._mean
        self._C0 = (Xc.T @ Xc) / Xc.shape[0]
        self._M = _sym_sqrt(self._C0)
        self._win = max(16, int(round(self.params["window_ms"] * self._sfreq / 1000.0)))
        self._buf = None
        self.fit_summary = f"calibrated on {X.shape[0]} samples, cutoff {self.params['cutoff']:.1f}"
        self.fitted = True

    # --- core: reconstruction operator for one window ------------------- #

    def _reconstruct_matrix(self, win: np.ndarray) -> np.ndarray:
        """Return the (n_eeg, n_eeg) reconstruction matrix R for this window."""
        wc = win - self._mean
        cov = (wc.T @ wc) / max(wc.shape[0], 1)
        lam, V = np.linalg.eigh(cov)                  # ascending
        # Calibrated variance along each window eigenvector.
        cal_var = np.einsum("ji,jk,ki->i", V, self._C0, V)
        thresh = (self.params["cutoff"] ** 2) * np.maximum(cal_var, 1e-12)
        keep = lam <= thresh
        if keep.all():
            return np.eye(win.shape[1])
        Vk = V[:, keep]
        if Vk.shape[1] == 0:                          # everything rejected: keep top dirs
            return np.eye(win.shape[1])
        # Mullen-style: reconstruct rejected dims from kept ones via M = C0^{1/2}.
        R = self._M @ np.linalg.pinv(Vk.T @ self._M) @ Vk.T
        return R

    def _clean_block(self, block: np.ndarray) -> np.ndarray:
        """Apply per-window reconstruction across a standalone block."""
        out = block.copy()
        n = block.shape[0]
        for s in range(0, n, self._win):
            e = min(s + self._win, n)
            seg = block[s:e]
            if seg.shape[0] < max(8, block.shape[1]):
                continue
            R = self._reconstruct_matrix(seg)
            out[s:e] = (seg - self._mean) @ R.T + self._mean
        return out

    def apply(self, data: np.ndarray) -> np.ndarray:
        if not self.fitted or data.shape[0] == 0:
            return data
        out = data.copy()
        out[:, self._eeg] = self._clean_block(data[:, self._eeg])
        return out

    def process_chunk(self, data: np.ndarray) -> np.ndarray:
        if not self.fitted or data.shape[0] == 0:
            return data
        x = data[:, self._eeg]
        # Maintain a rolling analysis window; derive R from it, apply to chunk.
        self._buf = x.copy() if self._buf is None else np.vstack([self._buf, x])[-self._win:]
        if self._buf.shape[0] < max(8, x.shape[1]):
            return data
        R = self._reconstruct_matrix(self._buf)
        out = data.copy()
        out[:, self._eeg] = (x - self._mean) @ R.T + self._mean
        return out

    def validate(self, sfreq):
        if self.requires_fit and not self.fitted:
            return ["ASR not calibrated yet (passing through). Use "
                    "'Calibrate artifact removal' on a clean window."]
        return []


# Register the calibrated stages into the shared preprocessing registry.
from neurobci.preprocessing.stages import STAGE_REGISTRY  # noqa: E402

for _cls in (InterpolateBad, ICARemoval, ASR):
    STAGE_REGISTRY[_cls.type_name] = _cls
