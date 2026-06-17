"""SSVEP: CCA identifies the attended frequency; calibration is usable."""

import unittest

import numpy as np

from neurobci.bci.calibration import run_simulated_calibration
from neurobci.bci.cca import CCADecoder, evaluate_cca
from neurobci.config.schema import ChannelConfig
from neurobci.paradigms.ssvep import SSVEPParadigm
from neurobci.paradigms.synthetic_paradigms import make_ssvep_dataset

PARA = SSVEPParadigm()


class TestSSVEP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ch = ChannelConfig()

    def test_cca_decodes_frequency(self):
        ds = make_ssvep_dataset(frequencies=PARA.frequencies, n_per_class=12,
                                channels=self.ch, window=PARA.window, seed=0)
        dec = CCADecoder(PARA.frequencies, ds.sfreq, PARA.n_harmonics)
        rep = evaluate_cca(dec, ds.X, ds.y)
        self.assertGreater(rep["accuracy"], 0.8)
        self.assertEqual(rep["chance"], 0.25)

    def test_cca_predict_shapes(self):
        ds = make_ssvep_dataset(frequencies=PARA.frequencies, n_per_class=4,
                                channels=self.ch, window=PARA.window, seed=2)
        dec = CCADecoder(PARA.frequencies, ds.sfreq, PARA.n_harmonics).fit(ds.X, ds.y)
        proba = dec.predict_proba(ds.X)
        self.assertEqual(proba.shape, (len(ds.y), len(PARA.frequencies)))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)

    def test_calibration_usable(self):
        res = run_simulated_calibration(PARA, self.ch, sfreq=500.0, n_trials=48, seed=0)
        self.assertTrue(res.is_usable)
        self.assertEqual(res.best_model_name, "cca")
        self.assertGreater(res.model.metrics["accuracy"], 0.8)


if __name__ == "__main__":
    unittest.main()
