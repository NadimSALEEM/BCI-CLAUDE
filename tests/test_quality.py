"""Signal-quality ratings on signals with known defects."""

import unittest

import numpy as np

from neurobci.core.stream_info import KIND_EEG, KIND_EOG, StreamInfo
from neurobci.quality.metrics import QualityRating, compute_quality


class TestQuality(unittest.TestCase):
    def setUp(self):
        self.sfreq = 500.0
        self.n = 1500
        self.t = np.arange(self.n) / self.sfreq
        self.rng = np.random.default_rng(0)

    def _info(self, names, kinds):
        return StreamInfo("t", self.sfreq, names, kinds)

    def test_flat_channel_is_bad(self):
        good = 12 * np.sin(2 * np.pi * 10 * self.t) + self.rng.normal(0, 3, self.n)
        flat = 0.1 * self.rng.normal(0, 1, self.n)
        data = np.column_stack([good, flat])
        info = self._info(["O1", "Cz"], [KIND_EEG, KIND_EEG])
        report = compute_quality(data, info)
        by_name = {c.name: c for c in report.channels}
        self.assertEqual(by_name["Cz"].rating, QualityRating.BAD)
        self.assertIn("flatline", by_name["Cz"].reasons[0])
        self.assertNotEqual(by_name["O1"].rating, QualityRating.BAD)

    def test_clean_channel_is_acceptable(self):
        clean = 10 * np.sin(2 * np.pi * 10 * self.t) + self.rng.normal(0, 3, self.n)
        data = clean[:, None]
        info = self._info(["O1"], [KIND_EEG])
        report = compute_quality(data, info)
        self.assertIn(
            report.channels[0].rating,
            (QualityRating.GOOD, QualityRating.FAIR),
        )

    def test_extreme_noise_is_bad(self):
        noisy = self.rng.normal(0, 200, self.n)
        data = noisy[:, None]
        info = self._info(["P3"], [KIND_EEG])
        report = compute_quality(data, info)
        self.assertEqual(report.channels[0].rating, QualityRating.BAD)

    def test_line_noise_flagged(self):
        base = 8 * np.sin(2 * np.pi * 10 * self.t)
        line = 25 * np.sin(2 * np.pi * 50 * self.t)  # heavy 50 Hz
        data = (base + line)[:, None]
        info = self._info(["Pz"], [KIND_EEG])
        report = compute_quality(data, info)
        reasons = " ".join(report.channels[0].reasons)
        self.assertIn("line noise", reasons)

    def test_eog_not_penalised_for_amplitude(self):
        # Large slow EOG-like swing should not be rated BAD purely for size.
        eog = 80 * np.sin(2 * np.pi * 1.0 * self.t) + self.rng.normal(0, 3, self.n)
        data = eog[:, None]
        info = self._info(["EOG"], [KIND_EOG])
        report = compute_quality(data, info)
        self.assertNotEqual(report.channels[0].rating, QualityRating.BAD)

    def test_too_few_samples_returns_empty(self):
        data = np.zeros((4, 2))
        info = self._info(["a", "b"], [KIND_EEG, KIND_EEG])
        report = compute_quality(data, info)
        self.assertEqual(report.channels, [])
        self.assertEqual(report.overall_rating, QualityRating.NODATA)


if __name__ == "__main__":
    unittest.main()
