"""Model + evaluation: beats chance on real signal, stays at chance on noise."""

import unittest

from neurobci.bci.evaluation import cross_validate
from neurobci.config.schema import ChannelConfig
from neurobci.paradigms.base import EpochWindow
from neurobci.paradigms.synthetic import make_p300_dataset

WIN = EpochWindow()


class TestModelsEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ch = ChannelConfig()
        cls.ds = make_p300_dataset(
            n_trials=240, channels=cls.ch, window=WIN,
            p300_amp_uv=6.0, noise_uv=4.0, seed=0,
        )

    def test_vec_lda_beats_chance_on_signal(self):
        res = cross_validate("vec_lda", self.ds.X, self.ds.y, self.ds.sfreq, 1, n_folds=5)
        self.assertGreater(res.balanced_accuracy, 0.75)
        self.assertTrue(res.beats_chance)
        self.assertGreater(res.roc_auc, 0.8)
        self.assertEqual(len(res.confusion), 2)
        self.assertEqual(sum(res.class_counts.values()), 240)

    def test_chance_level_on_pure_noise(self):
        # No P300 at all -> evaluation must NOT claim success (honesty check).
        noise = make_p300_dataset(
            n_trials=240, channels=self.ch, window=WIN,
            p300_amp_uv=0.0, noise_uv=4.0, seed=1,
        )
        res = cross_validate("vec_lda", noise.X, noise.y, noise.sfreq, 1, n_folds=5)
        self.assertLess(res.balanced_accuracy, 0.65)
        self.assertFalse(res.beats_chance)

    def test_riemann_lr_runs_and_separates(self):
        # Optional model; if pyriemann misbehaves cross_validate raises and we
        # skip, but normally it should clearly beat chance.
        try:
            res = cross_validate(
                "riemann_lr", self.ds.X, self.ds.y, self.ds.sfreq, 1, n_folds=4
            )
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"riemann_lr unavailable: {exc}")
        self.assertGreater(res.balanced_accuracy, 0.7)


if __name__ == "__main__":
    unittest.main()
