"""Multiple-comparison correction.

Self-contained implementations (no hard statsmodels dependency) so the result
is deterministic and testable. Always returns corrected p-values alongside the
raw ones -- the UI shows both, never hides the uncorrected map.
"""

from __future__ import annotations

import numpy as np

METHODS = ("none", "bonferroni", "holm", "fdr_bh", "fdr_by")

_LABELS = {
    "none": "No correction",
    "bonferroni": "Bonferroni",
    "holm": "Holm-Bonferroni",
    "fdr_bh": "FDR (Benjamini-Hochberg)",
    "fdr_by": "FDR (Benjamini-Yekutieli)",
}


def method_label(method: str) -> str:
    return _LABELS.get(method, method)


def correct(pvals, method: str = "fdr_bh", alpha: float = 0.05):
    """Return ``(reject, pvals_corrected)`` for a flat array of p-values.

    ``reject`` is a boolean mask at ``alpha``; ``pvals_corrected`` are adjusted
    p-values clipped to ``[0, 1]`` and preserve the input shape.
    """
    p = np.asarray(pvals, dtype=float)
    shape = p.shape
    flat = p.ravel()
    finite = np.isfinite(flat)
    out = np.full_like(flat, np.nan)
    pv = flat[finite]
    m = pv.size
    if m == 0:
        return np.zeros(shape, dtype=bool), p

    if method == "none":
        corr = pv.copy()
    elif method == "bonferroni":
        corr = np.minimum(pv * m, 1.0)
    elif method == "holm":
        order = np.argsort(pv)
        corr_sorted = np.empty(m)
        running = 0.0
        for rank, idx in enumerate(order):
            val = (m - rank) * pv[idx]
            running = max(running, val)
            corr_sorted[idx] = min(running, 1.0)
        corr = corr_sorted
    elif method in ("fdr_bh", "fdr_by"):
        order = np.argsort(pv)
        ranks = np.arange(1, m + 1)
        c = 1.0
        if method == "fdr_by":
            c = np.sum(1.0 / ranks)               # harmonic penalty (dependence)
        p_sorted = pv[order]
        adj_sorted = p_sorted * m * c / ranks
        # enforce monotonicity from the largest p downward
        adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
        adj = np.empty(m)
        adj[order] = np.clip(adj_sorted, 0.0, 1.0)
        corr = adj
    else:
        raise ValueError(f"Unknown correction method {method!r}. Use one of {METHODS}.")

    out[finite] = corr
    reject = np.zeros_like(flat, dtype=bool)
    reject[finite] = corr <= alpha
    return reject.reshape(shape), out.reshape(shape)
