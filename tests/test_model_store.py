"""Model persistence + compatibility checking."""

import tempfile
import unittest

import numpy as np

from neurobci.bci.calibration import run_simulated_calibration
from neurobci.bci.model_store import (
    check_compatibility,
    list_models,
    load_model,
    save_model,
)
from neurobci.config.schema import ChannelConfig
from neurobci.core.stream_info import StreamInfo
from neurobci.paradigms.p300 import P300Paradigm
from neurobci.paradigms.synthetic import make_p300_dataset


class TestModelStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.channels = ChannelConfig()
        cls.result = run_simulated_calibration(
            P300Paradigm(), cls.channels, sfreq=500.0, n_trials=200,
            model_names=["vec_lda"], seed=0,
        )
        cls.model = cls.result.model

    def _info(self, names=None, sfreq=500.0):
        names = names or self.channels.all_channels
        kinds = self.model.channel_kinds[: len(names)]
        return StreamInfo("t", sfreq, names, kinds)

    def test_save_load_roundtrip_predictions_equal(self):
        ds = make_p300_dataset(n_trials=40, channels=self.channels,
                               window=P300Paradigm().window, seed=9)
        before = self.model.target_scores(ds.X)
        with tempfile.TemporaryDirectory() as d:
            path = save_model(self.model, root=d)
            self.assertIn(path, list_models(d))
            loaded = load_model(path)
            after = loaded.target_scores(ds.X)
        np.testing.assert_allclose(before, after, rtol=1e-9)

    def test_untrained_refused(self):
        from neurobci.bci.models import make_model
        m = make_model("vec_lda", P300Paradigm(),
                       self.channels.all_channels, self.model.channel_kinds, 500.0)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                save_model(m, root=d)

    def test_compatibility_checks(self):
        self.assertEqual(check_compatibility(self.model, self._info()), [])
        # Wrong sampling rate.
        self.assertTrue(check_compatibility(self.model, self._info(sfreq=250.0)))
        # Fewer channels.
        bad = check_compatibility(self.model, self._info(names=["F3", "F4"]))
        self.assertTrue(bad)


if __name__ == "__main__":
    unittest.main()
