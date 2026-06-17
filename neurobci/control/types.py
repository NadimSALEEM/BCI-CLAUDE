"""Shared types for the control layer.

A :class:`Command` is the unit that flows from the decision logic to a
control adapter. Crucially it carries a full **trace** so every executed
action can be explained: which evidence produced it, the model version,
the confidence and the decision rule.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class OutcomeStatus(str, Enum):
    PENDING = "pending"
    DECIDED = "decided"
    ABSTAINED = "abstained"


@dataclass
class Command:
    """A proposed action plus everything needed to justify it."""

    action: str                       # e.g. "select:3"
    item: int | None                  # selected item (None for non-selection)
    confidence: float
    margin: float = 0.0
    created: float = field(default_factory=time.time)
    trace: dict = field(default_factory=dict)


@dataclass
class SelectionOutcome:
    """Result of a P300 selection run / accumulation."""

    status: OutcomeStatus
    item: int | None
    confidence: float
    margin: float
    n_flashes: int
    per_item_mean: list[float]
    reason: str = ""

    @property
    def decided(self) -> bool:
        return self.status is OutcomeStatus.DECIDED


@dataclass
class CommandRecord:
    """A history entry: a command and what happened to it."""

    command: Command
    executed: bool
    rejected_reason: str | None = None
    latency_s: float = 0.0
    intended: int | None = None
    correct: bool | None = None
    timestamp: float = field(default_factory=time.time)
