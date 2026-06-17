"""End-to-end: simulated calibration -> trained model -> online selection."""

import unittest

from neurobci.bci.calibration import run_simulated_calibration
from neurobci.bci.online import OnlineP300Decoder
from neurobci.config.schema import ChannelConfig
from neurobci.paradigms.p300 import P300Paradigm
from neurobci.paradigms.synthetic import make_p300_selection_run


class TestCalibrationOnline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paradigm = P300Paradigm()
        cls.channels = ChannelConfig()
        cls.result = run_simulated_calibration(
            cls.paradigm, cls.channels, sfreq=500.0, n_trials=300,
            p300_amp_uv=6.0, noise_uv=4.0, model_names=["vec_lda"], seed=0,
        )

    def test_calibration_produces_usable_model(self):
        self.assertIsNotNone(self.result.model)
        self.assertTrue(self.result.model.trained)
        self.assertTrue(self.result.is_usable)
        self.assertEqual(sum(self.result.class_counts.values()), 300)
        self.assertIn("balanced_accuracy", self.result.model.metrics)

    def test_online_selects_attended_item(self):
        decoder = OnlineP300Decoder(self.result.model)
        for target in (0, 3, 5):
            ds, item_ids, true_item = make_p300_selection_run(
                n_items=6, n_repetitions=12, target_item=target,
                channels=self.channels, sfreq=500.0,
                p300_amp_uv=6.0, noise_uv=4.0, window=self.paradigm.window,
                seed=10 + target,
            )
            selected, means = decoder.decide_selection(ds.X, item_ids, n_items=6)
            self.assertEqual(selected, true_item, f"target {target}")

    def test_insufficient_data_flagged_not_usable(self):
        bad = run_simulated_calibration(
            self.paradigm, self.channels, sfreq=500.0, n_trials=120,
            p300_amp_uv=0.0, noise_uv=5.0, model_names=["vec_lda"], seed=2,
        )
        self.assertFalse(bad.is_usable)
        self.assertTrue(any("chance" in m.lower() for m in bad.messages))


if __name__ == "__main__":
    unittest.main()
