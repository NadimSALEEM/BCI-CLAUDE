"""Data-trust verdicts: clean data is trusted, trashed data is not."""

import unittest

import numpy as np

from neurobci.core.stream_info import KIND_EEG, KIND_EOG, StreamInfo
from neurobci.quality.verify import DataTrust, verify_recording, verify_window

SF = 500.0
NAMES = ["Fz", "Cz", "Pz", "C3", "C4", "F3", "F4", "P3", "EOG"]
KINDS = [KIND_EEG] * 8 + [KIND_EOG]


def _info():
    return StreamInfo("t", SF, NAMES, KINDS, source_kind="replay")


def _eeg_like(T, seed=0):
    """Pink-ish EEG with a 1/f falloff and ~10 uV amplitude."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) / SF
    x = np.zeros((T, len(NAMES)))
    for c in range(len(NAMES)):
        sig = (10 * np.sin(2 * np.pi * 10 * t + rng.random())
               + 6 * np.sin(2 * np.pi * 6 * t + rng.random()))
        # 1/f background via cumulative noise (random walk), scaled.
        walk = np.cumsum(rng.standard_normal(T))
        walk = 8 * walk / (np.std(walk) + 1e-9)
        x[:, c] = sig + walk + rng.standard_normal(T) * 2.0
    return x


class TestVerifyWindow(unittest.TestCase):
    def test_clean_data_is_trusted(self):
        rep = verify_window(_eeg_like(3000), _info())
        self.assertEqual(rep.verdict, DataTrust.TRUST)

    def test_dc_offset_alone_is_not_a_fault(self):
        # A big constant electrode offset (normal for raw Enobio) with healthy
        # AC must NOT be condemned -- this is the real-data false alarm we fix.
        x = _eeg_like(3000, seed=1)
        x[:, [0, 1, 2]] += 20000.0            # huge DC offset, AC unchanged
        rep = verify_window(x, _info())
        self.assertEqual(rep.verdict, DataTrust.TRUST)

    def test_railing_channels_untrustworthy(self):
        rng = np.random.default_rng(11)
        x = _eeg_like(3000, seed=1)
        for c in (2, 3, 4, 5):               # half the EEG genuinely railing
            x[:, c] = 2000.0 * np.sign(rng.standard_normal(3000))  # clipping AC
        rep = verify_window(x, _info())
        self.assertEqual(rep.verdict, DataTrust.UNTRUSTWORTHY)
        self.assertTrue(any("saturate" in r or "unusable" in r for r in rep.reasons))

    def test_flatlined_is_untrustworthy(self):
        x = _eeg_like(3000, seed=2) * 0.0     # dead amplifier
        rep = verify_window(x, _info())
        self.assertEqual(rep.verdict, DataTrust.UNTRUSTWORTHY)

    def test_nan_flagged(self):
        x = _eeg_like(2000, seed=3)
        x[10:40, 1] = np.nan
        rep = verify_window(x, _info())
        self.assertIn(rep.verdict, (DataTrust.CAUTION, DataTrust.UNTRUSTWORTHY))


class TestVerifyRecording(unittest.TestCase):
    def test_aggregates_and_flags_persistent_bad_channel(self):
        rng = np.random.default_rng(12)
        x = _eeg_like(10000, seed=4)
        x[:, 2] = 2000.0 * np.sign(rng.standard_normal(10000))  # Pz railing throughout
        rep = verify_recording(x, NAMES, KINDS, SF, window_s=2.0)
        self.assertEqual(rep.verdict, DataTrust.UNTRUSTWORTHY)
        self.assertIn("Pz", rep.bad_channels)
        self.assertGreater(rep.checks["n_windows"], 3)

    def test_clean_recording_trusted(self):
        rep = verify_recording(_eeg_like(8000, seed=5), NAMES, KINDS, SF, window_s=2.0)
        self.assertEqual(rep.verdict, DataTrust.TRUST)


if __name__ == "__main__":
    unittest.main()
