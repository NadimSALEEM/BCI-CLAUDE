"""Epoch extraction: indexing, baseline, rejection, bounds."""

import unittest

import numpy as np

from neurobci.bci.epoching import extract_epochs, onsets_from_timestamps
from neurobci.paradigms.base import EpochWindow

SF = 100.0


class TestEpoching(unittest.TestCase):
    def _ramp(self, n=1000, ch=3):
        # data[i, c] == i, so an epoch's content is its sample indices.
        col = np.arange(n, dtype=float)
        return np.repeat(col[:, None], ch, axis=1)

    def test_slice_indices_correct(self):
        data = self._ramp()
        win = EpochWindow(tmin=-0.1, tmax=0.1, baseline=None, reject_uv=1e12)
        es = extract_epochs(data, SF, np.array([500]), np.array([1]), win)
        self.assertEqual(es.X.shape, (1, 3, 21))   # -10..+10 inclusive
        np.testing.assert_array_equal(es.X[0, 0], np.arange(490, 511))

    def test_out_of_bounds_dropped(self):
        data = self._ramp()
        win = EpochWindow(tmin=-0.1, tmax=0.1, baseline=None, reject_uv=1e12)
        es = extract_epochs(data, SF, np.array([2, 995, 500]), np.array([0, 0, 1]), win)
        self.assertEqual(es.n_epochs, 1)           # only 500 fits
        self.assertEqual(es.n_out_of_bounds, 2)
        self.assertEqual(list(es.y), [1])

    def test_baseline_correction(self):
        data = self._ramp()
        win = EpochWindow(tmin=-0.1, tmax=0.1, baseline=(-0.1, 0.0), reject_uv=1e12)
        es = extract_epochs(data, SF, np.array([500]), np.array([1]), win)
        times = es.times
        bmask = (times >= -0.1) & (times <= 0.0)
        self.assertAlmostEqual(float(es.X[0, 0, bmask].mean()), 0.0, places=6)

    def test_amplitude_rejection(self):
        data = np.zeros((1000, 2))
        data[:, 0] = 1.0
        data[500, 1] = 10_000.0                    # huge spike near one onset
        win = EpochWindow(tmin=-0.05, tmax=0.05, baseline=None, reject_uv=150.0)
        es = extract_epochs(data, SF, np.array([300, 500]), np.array([0, 1]), win)
        self.assertEqual(es.n_rejected, 1)
        self.assertEqual(es.n_epochs, 1)
        self.assertEqual(list(es.y), [0])

    def test_onsets_from_timestamps(self):
        ts = np.arange(0, 10, 0.01)                # 100 Hz timeline
        onsets = onsets_from_timestamps(ts, np.array([1.0, 5.0]))
        self.assertEqual(list(onsets), [100, 500])


if __name__ == "__main__":
    unittest.main()
