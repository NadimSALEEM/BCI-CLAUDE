"""DecisionLayer: confidence threshold, voting, refractory, no-command."""

import unittest

from neurobci.config.schema import DecisionConfig
from neurobci.control.decision import DecisionLayer


def _cfg(**kw):
    base = dict(confidence_threshold=0.6, vote_window=5, min_agree=3,
                refractory_s=1.0, neutral_label="none")
    base.update(kw)
    return DecisionConfig(**base)


class TestDecision(unittest.TestCase):
    def test_low_confidence_never_commands(self):
        d = DecisionLayer(_cfg())
        for _ in range(10):
            dec = d.push("left", 0.5, now=0.0)   # below threshold
            self.assertFalse(dec.emitted)
            self.assertEqual(dec.label, "none")

    def test_emits_after_enough_agreement(self):
        d = DecisionLayer(_cfg(min_agree=3))
        self.assertFalse(d.push("left", 0.9, now=0.0).emitted)
        self.assertFalse(d.push("left", 0.9, now=0.0).emitted)
        dec = d.push("left", 0.9, now=0.0)
        self.assertTrue(dec.emitted)
        self.assertEqual(dec.command.action, "left")

    def test_refractory_blocks_immediate_repeat(self):
        d = DecisionLayer(_cfg(min_agree=2, refractory_s=5.0))
        d.push("up", 0.9, now=0.0)
        self.assertTrue(d.push("up", 0.9, now=0.0).emitted)
        # Window cleared; build agreement again but still within refractory.
        d.push("up", 0.9, now=1.0)
        dec = d.push("up", 0.9, now=1.5)
        self.assertFalse(dec.emitted)
        self.assertIn("refractory", dec.reason)
        # After the refractory period it can emit again.
        self.assertTrue(d.push("up", 0.9, now=7.0).emitted)

    def test_disagreement_yields_no_command(self):
        d = DecisionLayer(_cfg(min_agree=3, vote_window=4))
        d.push("left", 0.9, now=0.0)
        d.push("right", 0.9, now=0.0)
        d.push("left", 0.9, now=0.0)
        dec = d.push("right", 0.9, now=0.0)
        self.assertFalse(dec.emitted)


if __name__ == "__main__":
    unittest.main()
