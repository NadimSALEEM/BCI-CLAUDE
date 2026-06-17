"""Motor imagery: decoders beat chance; streaming control via DecisionLayer."""

import unittest

import numpy as np

from neurobci.bci.calibration import run_simulated_calibration
from neurobci.bci.evaluation import cross_validate
from neurobci.config.schema import ChannelConfig, DecisionConfig
from neurobci.control.decision import DecisionLayer
from neurobci.control.simulated_driver import simulate_mi_stream
from neurobci.paradigms.motor_imagery import MotorImageryParadigm
from neurobci.paradigms.synthetic_paradigms import make_mi_dataset

PARA = MotorImageryParadigm()


class TestMotorImagery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ch = ChannelConfig()
        cls.ds = make_mi_dataset(n_trials=120, channels=cls.ch,
                                 window=PARA.window, seed=0)

    def test_csp_lda_beats_chance(self):
        r = cross_validate("csp_lda", self.ds.X, self.ds.y, self.ds.sfreq, 1, n_folds=4)
        self.assertGreater(r.balanced_accuracy, 0.7)
        self.assertTrue(r.beats_chance)

    def test_chance_on_no_erd(self):
        noise = make_mi_dataset(n_trials=120, channels=self.ch, window=PARA.window,
                                erd=0.0, seed=3)
        r = cross_validate("csp_lda", noise.X, noise.y, noise.sfreq, 1, n_folds=4)
        self.assertLess(r.balanced_accuracy, 0.65)
        self.assertFalse(r.beats_chance)

    def test_calibration_usable(self):
        res = run_simulated_calibration(PARA, self.ch, sfreq=500.0, n_trials=120,
                                        model_names=["cov_ts_lr"], n_folds=4, seed=0)
        self.assertTrue(res.is_usable)
        self.assertIsNotNone(res.model)

    def test_streaming_emits_correct_command(self):
        res = run_simulated_calibration(PARA, self.ch, sfreq=500.0, n_trials=120,
                                        model_names=["cov_ts_lr"], n_folds=4, seed=0)
        dl = DecisionLayer(DecisionConfig(confidence_threshold=0.5, min_agree=3,
                                          vote_window=5, refractory_s=0.0))
        decisions = simulate_mi_stream(res.model, dl, true_class=1, channels=self.ch,
                                       sfreq=500.0, n_windows=15, seed=1)
        emitted = [d for d in decisions if d.emitted]
        self.assertTrue(emitted, "no command emitted from MI stream")
        self.assertEqual(emitted[0].command.action, "right")  # class index 1


if __name__ == "__main__":
    unittest.main()
