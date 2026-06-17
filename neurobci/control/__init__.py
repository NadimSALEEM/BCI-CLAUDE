"""Decision layer, safety gating, control adapters and command routing."""

from neurobci.control.adapters import (
    BoardModel,
    ControlAdapter,
    InternalBoardAdapter,
    NullAdapter,
)
from neurobci.control.decision import Decision, DecisionLayer
from neurobci.control.errp_correction import CorrectionDecision, ErrPCorrector
from neurobci.control.mapping import CommandMap
from neurobci.control.metrics import OnlineMetrics
from neurobci.control.router import CommandRouter
from neurobci.control.safety import SafetyContext, SafetyMonitor, SafetyVerdict
from neurobci.control.selection import SelectionController
from neurobci.control.simulated_driver import (
    run_selection_trial,
    simulate_mi_stream,
    simulate_selection,
)
from neurobci.control.types import (
    Command,
    CommandRecord,
    OutcomeStatus,
    SelectionOutcome,
)

__all__ = [
    "DecisionLayer",
    "Decision",
    "SelectionController",
    "SafetyMonitor",
    "SafetyContext",
    "SafetyVerdict",
    "ControlAdapter",
    "InternalBoardAdapter",
    "NullAdapter",
    "BoardModel",
    "CommandRouter",
    "OnlineMetrics",
    "Command",
    "CommandRecord",
    "SelectionOutcome",
    "OutcomeStatus",
    "run_selection_trial",
    "simulate_selection",
    "simulate_mi_stream",
    "CommandMap",
    "ErrPCorrector",
    "CorrectionDecision",
]
