"""SelectionController: early-stop on confident margin, abstain otherwise."""

import unittest

from neurobci.config.schema import SelectionConfig
from neurobci.control.selection import SelectionController
from neurobci.control.types import OutcomeStatus


def _cfg(**kw):
    base = dict(min_repetitions=3, max_repetitions=10, confidence_margin=0.15,
                min_confidence=0.5, timeout_s=1e9)
    base.update(kw)
    return SelectionConfig(**base)


class TestSelection(unittest.TestCase):
    def test_decides_dominant_item(self):
        ctrl = SelectionController(_cfg(), n_items=4)
        outcome = None
        # Item 2 gets high scores, others low; cycle through items.
        for rep in range(10):
            for item in range(4):
                score = 0.9 if item == 2 else 0.1
                outcome = ctrl.add(item, score, now=0.0)
                if outcome.status is not OutcomeStatus.PENDING:
                    break
            if outcome.status is not OutcomeStatus.PENDING:
                break
        self.assertEqual(outcome.status, OutcomeStatus.DECIDED)
        self.assertEqual(outcome.item, 2)
        self.assertGreaterEqual(outcome.margin, 0.15)

    def test_early_stopping(self):
        # With a clear winner it should decide right at min_repetitions.
        ctrl = SelectionController(_cfg(min_repetitions=3), n_items=3)
        outcome = None
        reps_done = 0
        for rep in range(10):
            for item in range(3):
                outcome = ctrl.add(item, 0.95 if item == 0 else 0.05, now=0.0)
            reps_done = rep + 1
            if outcome.status is OutcomeStatus.DECIDED:
                break
        self.assertEqual(outcome.item, 0)
        self.assertLessEqual(reps_done, 4)

    def test_abstains_when_ambiguous(self):
        ctrl = SelectionController(_cfg(max_repetitions=5), n_items=4)
        outcome = None
        for rep in range(5):
            for item in range(4):
                outcome = ctrl.add(item, 0.5, now=0.0)  # all equal -> no margin
        self.assertEqual(outcome.status, OutcomeStatus.ABSTAINED)
        self.assertIsNone(outcome.item)
        self.assertIn("max repetitions", outcome.reason)

    def test_min_confidence_blocks_low_scores(self):
        # Item 1 wins by margin but all scores are below min_confidence.
        ctrl = SelectionController(_cfg(min_confidence=0.6, max_repetitions=4), n_items=3)
        outcome = None
        for rep in range(4):
            for item in range(3):
                outcome = ctrl.add(item, 0.3 if item == 1 else 0.05, now=0.0)
        self.assertEqual(outcome.status, OutcomeStatus.ABSTAINED)


if __name__ == "__main__":
    unittest.main()
