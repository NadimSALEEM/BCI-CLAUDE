"""Feature transformers for epoch arrays.

Transformers are scikit-learn compatible and operate on epoch arrays of
shape ``(n_epochs, n_channels, n_times)`` so they slot straight into an
``sklearn.pipeline.Pipeline`` together with pyriemann transformers.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class EpochVectorizer(BaseEstimator, TransformerMixin):
    """Decimate epochs in time and flatten to a 2-D feature matrix.

    Decimation reduces dimensionality (ERP features are smooth, so this
    loses little) which is important for the small datasets produced by
    short calibration. Keeps the transform a pure, leakage-free function of
    each epoch.
    """

    def __init__(self, decim: int = 8) -> None:
        self.decim = max(int(decim), 1)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"expected (n,ch,times), got {X.shape}")
        Xd = X[:, :, :: self.decim]
        return Xd.reshape(Xd.shape[0], -1)
