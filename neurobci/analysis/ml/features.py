"""Feature families for machine learning.

Two shapes flow into models:

* **tabular** ``(n_trials, n_features)`` for ordinary estimators (LDA, SVM, RF,
  ...). Built here from ERP-window, band-power, raw/downsampled and metadata
  families, which the user can combine.
* **tensor** ``(n_trials, n_channels, n_times)`` for spatial pipelines
  (CSP / xDAWN / Riemannian), which learn their own features *inside* the
  cross-validation fold to avoid leakage -- so the raw tensor is passed through.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neurobci.analysis.stats.features import (
    BandPowerSpec, ERPFeatureSpec, band_power_feature_per_channel,
    erp_feature_per_channel, window_mask)


@dataclass
class RawEpochsSpec:
    downsample: int = 1
    channels: list[str] | None = None
    tmin: float | None = None
    tmax: float | None = None


@dataclass
class MetadataSpec:
    keys: list[str] = field(default_factory=list)


@dataclass
class MLFeatureConfig:
    """An ordered set of feature families to concatenate into one matrix."""

    families: list = field(default_factory=list)

    def describe(self) -> list[str]:
        return [type(f).__name__ for f in self.families]


def _raw_family(bundle, spec: RawEpochsSpec):
    ch_idx = bundle.roi_indices(spec.channels)
    X = bundle.X[:, ch_idx, :]
    times = bundle.times
    if spec.tmin is not None or spec.tmax is not None:
        m = window_mask(times, spec.tmin if spec.tmin is not None else times[0],
                        spec.tmax if spec.tmax is not None else times[-1])
        X = X[:, :, m]
        times = times[m]
    if spec.downsample > 1:
        # Anti-aliased decimation (low-pass then subsample), so downsampling
        # never aliases high-frequency content even on an un-low-passed signal.
        from scipy.signal import decimate
        X = decimate(X, spec.downsample, axis=2, ftype="iir", zero_phase=True)
        times = times[::spec.downsample][:X.shape[2]]
    names = [f"{bundle.channel_names[c]}@{t:.3f}s"
             for c in ch_idx for t in times]
    return X.reshape(X.shape[0], -1), names


def _metadata_family(bundle, spec: MetadataSpec):
    cols, names = [], []
    for k in spec.keys:
        if k in bundle.metadata:
            cols.append(np.asarray(bundle.metadata[k], float))
            names.append(f"meta:{k}")
    if not cols:
        return np.empty((bundle.n_trials, 0)), []
    return np.column_stack(cols), names


def extract_features(bundle, config: MLFeatureConfig):
    """Concatenate all configured families into ``(X, names)`` (tabular)."""
    mats, names = [], []
    for spec in config.families:
        if isinstance(spec, ERPFeatureSpec):
            vals, ns = erp_feature_per_channel(bundle, spec)
        elif isinstance(spec, BandPowerSpec):
            vals, ns = band_power_feature_per_channel(bundle, spec)
        elif isinstance(spec, RawEpochsSpec):
            vals, ns = _raw_family(bundle, spec)
        elif isinstance(spec, MetadataSpec):
            vals, ns = _metadata_family(bundle, spec)
        else:
            raise TypeError(f"Unknown feature family {type(spec).__name__}")
        mats.append(np.asarray(vals, float))
        names.extend(ns)
    if not mats:
        raise ValueError("No feature families configured.")
    X = np.concatenate(mats, axis=1)
    return X, names


def tensor_X(bundle, channels: list[str] | None = None) -> np.ndarray:
    """Raw ``(n_trials, n_channels, n_times)`` tensor for spatial pipelines."""
    if channels:
        idx = bundle.roi_indices(channels)
        return bundle.X[:, idx, :]
    return bundle.X
