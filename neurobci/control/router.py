"""Command router: the single choke-point between decisions and actions.

Every proposed command passes through here. The router:

* asks the :class:`SafetyMonitor` whether it is allowed,
* enforces a commands-per-minute **rate limit**,
* honours **test mode** (record but never execute),
* executes via the adapter only if all gates pass,
* records a :class:`CommandRecord` in the history and updates metrics.

Because everything funnels through one method, the emergency stop and the
"no action on uncertainty" rule are guaranteed for every command path.
"""

from __future__ import annotations

import logging
import time
from collections import deque

from neurobci.config.schema import ControlConfig
from neurobci.control.adapters import ControlAdapter
from neurobci.control.metrics import OnlineMetrics
from neurobci.control.safety import SafetyContext, SafetyMonitor
from neurobci.control.types import Command, CommandRecord

logger = logging.getLogger(__name__)


class CommandRouter:
    def __init__(
        self,
        adapter: ControlAdapter,
        config: ControlConfig,
        metrics: OnlineMetrics | None = None,
    ) -> None:
        self.adapter = adapter
        self.config = config
        self.safety = SafetyMonitor(config.safety)
        self.metrics = metrics or OnlineMetrics()
        self.history: list[CommandRecord] = []
        self._exec_times: deque[float] = deque()

    # ----- rate limiting ------------------------------------------------- #

    def _rate_ok(self, now: float) -> bool:
        while self._exec_times and now - self._exec_times[0] > 60.0:
            self._exec_times.popleft()
        return len(self._exec_times) < self.config.safety.max_commands_per_min

    # ----- submission ---------------------------------------------------- #

    def submit(
        self,
        command: Command,
        *,
        emergency_stop: bool = False,
        connected: bool = True,
        quality_ok: bool | None = None,
        intended: int | None = None,
        latency_s: float = 0.0,
    ) -> CommandRecord:
        now = time.time()
        ctx = SafetyContext(
            emergency_stop=emergency_stop,
            connected=connected,
            confidence=command.confidence,
            quality_ok=quality_ok,
            external_target=self.adapter.is_external,
        )
        verdict = self.safety.check(ctx)
        reasons = list(verdict.reasons)

        executed = False
        if not verdict.allowed:
            pass
        elif not self._rate_ok(now):
            reasons.append("rate limit exceeded")
        elif self.config.test_mode:
            reasons.append("test mode (not executed)")
        else:
            try:
                self.adapter.execute(command)
                executed = True
                self._exec_times.append(now)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Adapter execution failed.")
                reasons.append(f"adapter error: {exc}")

        correct = None if intended is None else (command.item == intended)
        rec = CommandRecord(
            command=command,
            executed=executed,
            rejected_reason=None if executed else "; ".join(reasons) or "blocked",
            latency_s=latency_s,
            intended=intended,
            correct=correct,
        )
        self.history.append(rec)
        self.metrics.record_command(rec)
        return rec

    def record_abstention(self, confidence: float = 0.0, intended: int | None = None) -> CommandRecord:
        """Record a 'no command' outcome (safe abstention)."""
        rec = CommandRecord(
            command=Command(action="no_command", item=None, confidence=confidence),
            executed=False,
            rejected_reason="abstained (insufficient evidence)",
            intended=intended,
            correct=None,
        )
        self.history.append(rec)
        self.metrics.record_abstention()
        return rec
