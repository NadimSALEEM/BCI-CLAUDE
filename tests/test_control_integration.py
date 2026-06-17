"""End-to-end: trained model -> selection -> safety -> adapter -> cursor."""

import unittest

import numpy as np

from neurobci.bci.calibration import run_simulated_calibration
from neurobci.config.schema import ControlConfig, SelectionConfig
from neurobci.control.adapters import BoardModel, InternalBoardAdapter
from neurobci.control.router import CommandRouter
from neurobci.control.simulated_driver import run_selection_trial
from neurobci.control.types import OutcomeStatus
from neurobci.config.schema import ChannelConfig
from neurobci.paradigms.p300 import P300Paradigm


class TestControlIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.channels = ChannelConfig()
        cls.result = run_simulated_calibration(
            P300Paradigm(), cls.channels, sfreq=500.0, n_trials=240,
            model_names=["vec_lda"], seed=0,
        )
        cls.model = cls.result.model
        assert cls.model is not None and cls.result.is_usable

    def _router(self, **ctrl_kw):
        cfg = ControlConfig(n_items=6, **ctrl_kw)
        cfg.safety.min_confidence = 0.0   # isolate selection logic from safety
        board = BoardModel(n_items=6)
        return CommandRouter(InternalBoardAdapter(board), cfg), board, cfg

    def test_selection_moves_cursor_to_intended(self):
        router, board, cfg = self._router()
        sel = SelectionConfig()
        for intended in (0, 2, 4):
            outcome, rec = run_selection_trial(
                self.model, router, sel, intended, self.channels, 500.0,
                p300_amp_uv=6.0, noise_uv=4.0, connected=True, quality_ok=True,
                seed=20 + intended,
            )
            self.assertEqual(outcome.status, OutcomeStatus.DECIDED)
            self.assertTrue(rec.executed)
            self.assertTrue(rec.correct)
            self.assertEqual(board.last_item, intended)
        self.assertEqual(router.metrics.n_correct, 3)
        self.assertIsNotNone(router.metrics.accuracy)
        self.assertGreater(router.metrics.commands_per_min, 0.0)

    def test_emergency_stop_blocks_execution(self):
        router, board, cfg = self._router()
        sel = SelectionConfig()
        outcome, rec = run_selection_trial(
            self.model, router, sel, intended_item=1, channels=self.channels,
            sfreq=500.0, emergency_stop=True, connected=True, quality_ok=True,
            seed=5,
        )
        # A selection was decided, but the command was NOT executed.
        self.assertEqual(outcome.status, OutcomeStatus.DECIDED)
        self.assertFalse(rec.executed)
        self.assertIsNone(board.last_item)
        self.assertIn("emergency", rec.rejected_reason)

    def test_disconnected_blocks_execution(self):
        router, board, cfg = self._router()
        sel = SelectionConfig()
        _, rec = run_selection_trial(
            self.model, router, sel, intended_item=3, channels=self.channels,
            sfreq=500.0, connected=False, quality_ok=True, seed=7,
        )
        self.assertFalse(rec.executed)
        self.assertIsNone(board.last_item)


if __name__ == "__main__":
    unittest.main()
