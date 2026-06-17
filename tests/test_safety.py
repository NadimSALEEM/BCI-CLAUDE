"""SafetyMonitor gating and router rate limiting / test mode."""

import unittest

from neurobci.config.schema import ControlConfig, SafetyConfig
from neurobci.control.adapters import BoardModel, InternalBoardAdapter, NullAdapter
from neurobci.control.router import CommandRouter
from neurobci.control.safety import SafetyContext, SafetyMonitor
from neurobci.control.types import Command


class _ExternalAdapter(NullAdapter):
    is_external = True


class TestSafety(unittest.TestCase):
    def _mon(self, **kw):
        base = dict(external_control_enabled=False, min_confidence=0.5,
                    max_commands_per_min=60.0, require_connected=True,
                    require_quality=True)
        base.update(kw)
        return SafetyMonitor(SafetyConfig(**base))

    def test_emergency_stop_blocks(self):
        v = self._mon().check(SafetyContext(emergency_stop=True, confidence=1.0))
        self.assertFalse(v.allowed)
        self.assertTrue(any("emergency" in r for r in v.reasons))

    def test_disconnected_blocks(self):
        v = self._mon().check(SafetyContext(connected=False, confidence=1.0))
        self.assertFalse(v.allowed)

    def test_low_confidence_blocks(self):
        v = self._mon(min_confidence=0.7).check(SafetyContext(confidence=0.6))
        self.assertFalse(v.allowed)

    def test_quality_blocks(self):
        v = self._mon().check(SafetyContext(confidence=1.0, quality_ok=False))
        self.assertFalse(v.allowed)

    def test_external_disabled_blocks(self):
        v = self._mon().check(SafetyContext(confidence=1.0, external_target=True))
        self.assertFalse(v.allowed)

    def test_all_clear(self):
        v = self._mon().check(SafetyContext(confidence=1.0, quality_ok=True))
        self.assertTrue(v.allowed)


class TestRouter(unittest.TestCase):
    def _router(self, **safety):
        cfg = ControlConfig(safety=SafetyConfig(**safety))
        board = BoardModel(n_items=4)
        return CommandRouter(InternalBoardAdapter(board), cfg), board

    def _cmd(self, item=1, conf=0.9):
        return Command(action=f"select:{item}", item=item, confidence=conf)

    def test_executes_when_allowed(self):
        router, board = self._router(min_confidence=0.5)
        rec = router.submit(self._cmd(2), connected=True, quality_ok=True)
        self.assertTrue(rec.executed)
        self.assertEqual(board.last_item, 2)
        self.assertEqual(len(router.history), 1)

    def test_estop_blocks_execution(self):
        router, board = self._router()
        rec = router.submit(self._cmd(2), emergency_stop=True)
        self.assertFalse(rec.executed)
        self.assertIsNone(board.last_item)
        self.assertIn("emergency", rec.rejected_reason)

    def test_test_mode_does_not_execute(self):
        cfg = ControlConfig(test_mode=True)
        board = BoardModel(4)
        router = CommandRouter(InternalBoardAdapter(board), cfg)
        rec = router.submit(self._cmd(1), connected=True, quality_ok=True)
        self.assertFalse(rec.executed)
        self.assertIsNone(board.last_item)
        self.assertIn("test mode", rec.rejected_reason)

    def test_rate_limit(self):
        router, _ = self._router(max_commands_per_min=2, min_confidence=0.0)
        r1 = router.submit(self._cmd(), connected=True, quality_ok=True)
        r2 = router.submit(self._cmd(), connected=True, quality_ok=True)
        r3 = router.submit(self._cmd(), connected=True, quality_ok=True)
        self.assertTrue(r1.executed and r2.executed)
        self.assertFalse(r3.executed)
        self.assertIn("rate limit", r3.rejected_reason)

    def test_correctness_tracking(self):
        router, _ = self._router(min_confidence=0.0)
        rec = router.submit(self._cmd(item=3), connected=True, quality_ok=True, intended=3)
        self.assertTrue(rec.correct)
        self.assertEqual(router.metrics.n_correct, 1)


if __name__ == "__main__":
    unittest.main()
