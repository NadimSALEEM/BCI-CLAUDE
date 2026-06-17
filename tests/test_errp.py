"""ErrP: model beats chance; corrector rejects errors, accepts correct."""

import unittest

import numpy as np

from neurobci.bci.calibration import run_simulated_calibration
from neurobci.config.schema import ChannelConfig
from neurobci.control.errp_correction import ErrPCorrector
from neurobci.paradigms.errp import ErrPParadigm
from neurobci.paradigms.synthetic_paradigms import make_errp_dataset

PARA = ErrPParadigm()


class TestErrP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ch = ChannelConfig()
        cls.result = run_simulated_calibration(
            PARA, cls.ch, sfreq=500.0, n_trials=240, model_names=["vec_lda"], seed=0)
        cls.model = cls.result.model

    def test_calibration_usable(self):
        self.assertTrue(self.result.is_usable)
        self.assertEqual(self.model.positive_index, 1)  # 'error' is positive

    def test_corrector_rejects_error_accepts_correct(self):
        corrector = ErrPCorrector(self.model, threshold=0.5)
        ds = make_errp_dataset(n_trials=120, channels=self.ch, window=PARA.window,
                               error_ratio=0.5, seed=7)
        err_idx = np.where(ds.y == 1)[0]
        ok_idx = np.where(ds.y == 0)[0]
        # Aggregate over several epochs to avoid single-trial noise flakiness.
        err_rejects = sum(corrector.review(ds.X[i]).action == "reject"
                          for i in err_idx[:20])
        ok_accepts = sum(corrector.review(ds.X[i]).action == "accept"
                         for i in ok_idx[:20])
        self.assertGreaterEqual(err_rejects, 15)   # most errors flagged
        self.assertGreaterEqual(ok_accepts, 15)    # most correct accepted

    def test_corrector_requires_errp_model(self):
        from neurobci.bci.calibration import run_simulated_calibration as run
        from neurobci.paradigms.p300 import P300Paradigm
        p300 = run(P300Paradigm(), self.ch, n_trials=120,
                   model_names=["vec_lda"], seed=0).model
        with self.assertRaises(ValueError):
            ErrPCorrector(p300)


if __name__ == "__main__":
    unittest.main()
