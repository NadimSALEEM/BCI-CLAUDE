"""Cognitive-state index formulas, validity, caveats, baseline."""

import unittest

import numpy as np

from neurobci.core.stream_info import KIND_EEG, StreamInfo
from neurobci.spectral.indices import CAVEAT, compute_indices


def _info():
    # frontal, central, parietal, occipital EEG channels.
    return StreamInfo("t", 500.0, ["F3", "Cz", "Pz", "O1"], [KIND_EEG] * 4)


def _uniform_powers(n, alpha=2.0, theta=1.0, beta=3.0, delta=0.5, gamma=0.2):
    return {
        "alpha": np.full(n, alpha), "theta": np.full(n, theta),
        "beta": np.full(n, beta), "delta": np.full(n, delta),
        "gamma": np.full(n, gamma),
    }


class TestIndices(unittest.TestCase):
    def test_engagement_formula(self):
        info = _info()
        idx = {i.name: i for i in compute_indices(_uniform_powers(4), info)}
        # beta/(alpha+theta) = 3/(2+1) = 1.0
        self.assertAlmostEqual(idx["engagement"].value, 1.0, places=6)
        # (theta+alpha)/beta = (1+2)/3 = 1.0
        self.assertAlmostEqual(idx["drowsiness"].value, 1.0, places=6)
        # theta(frontal)/alpha(parietal+occipital) = 1/2 = 0.5
        self.assertAlmostEqual(idx["workload"].value, 0.5, places=6)

    def test_components_and_caveat_present(self):
        info = _info()
        eng = compute_indices(_uniform_powers(4), info)[0]
        self.assertIn("beta", eng.components)
        self.assertEqual(eng.note, CAVEAT)
        self.assertTrue(eng.channels)

    def test_all_bad_is_invalid(self):
        info = _info()
        idx = compute_indices(_uniform_powers(4), info, bad_channels={0, 1, 2, 3})
        self.assertTrue(all(not i.valid for i in idx))

    def test_baseline_relative(self):
        info = _info()
        idx = {i.name: i for i in compute_indices(
            _uniform_powers(4), info, baseline={"engagement": 0.5})}
        eng = idx["engagement"]
        self.assertEqual(eng.baseline, 0.5)
        self.assertAlmostEqual(eng.relative, 2.0, places=6)  # 1.0 / 0.5


if __name__ == "__main__":
    unittest.main()
