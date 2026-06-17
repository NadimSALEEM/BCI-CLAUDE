"""Safety gating for command execution.

No command reaches a control adapter unless every gate passes. Gates are
evaluated from an explicit :class:`SafetyContext` so the decision is
deterministic and testable, and every block carries a human-readable
reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from neurobci.config.schema import SafetyConfig


@dataclass
class SafetyContext:
    emergency_stop: bool = False
    connected: bool = True
    confidence: float = 1.0
    quality_ok: bool | None = None      # None = unknown / not evaluated
    external_target: bool = False       # does the command drive external HW?


@dataclass
class SafetyVerdict:
    allowed: bool
    reasons: list[str] = field(default_factory=list)


class SafetyMonitor:
    def __init__(self, config: SafetyConfig) -> None:
        self.config = config

    def check(self, ctx: SafetyContext) -> SafetyVerdict:
        reasons: list[str] = []
        c = self.config

        if ctx.emergency_stop:
            reasons.append("emergency stop is active")
        if c.require_connected and not ctx.connected:
            reasons.append("EEG stream not connected/stable")
        if ctx.confidence < c.min_confidence:
            reasons.append(
                f"confidence {ctx.confidence:.2f} < {c.min_confidence:.2f}"
            )
        if c.require_quality and ctx.quality_ok is False:
            reasons.append("signal quality unacceptable")
        if ctx.external_target and not c.external_control_enabled:
            reasons.append("external control is disabled")

        return SafetyVerdict(allowed=not reasons, reasons=reasons)
