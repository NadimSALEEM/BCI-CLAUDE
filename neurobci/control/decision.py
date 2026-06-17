"""Streaming class-decision layer.

For continuous / active-BCI control where the model emits a class label per
window. Converts that noisy stream into stable commands using:

* a **confidence threshold** (low-confidence predictions count as neutral),
* **majority voting** over a sliding window requiring ``min_agree`` votes,
* a **refractory period** after each emitted command,
* a **neutral / no-command** default -- the response to uncertainty is
  always *no action*.

Each call returns a :class:`Decision` recording exactly why a command was
or was not emitted (traceability).
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field

from neurobci.config.schema import DecisionConfig
from neurobci.control.types import Command


@dataclass
class Decision:
    emitted: bool
    label: str
    confidence: float
    reason: str
    command: Command | None = None
    votes: dict = field(default_factory=dict)


class DecisionLayer:
    def __init__(self, config: DecisionConfig) -> None:
        self.config = config
        self._window: deque[str] = deque(maxlen=max(1, config.vote_window))
        self._last_emit = float("-inf")     # so the first command is never in refractory

    def reset(self) -> None:
        self._window.clear()
        self._last_emit = float("-inf")

    def push(self, label: str, confidence: float, now: float | None = None) -> Decision:
        now = time.time() if now is None else now
        c = self.config

        # Below-threshold predictions are treated as the neutral class.
        effective = label if confidence >= c.confidence_threshold else c.neutral_label
        self._window.append(effective)

        votes = Counter(self._window)
        top_label, top_count = votes.most_common(1)[0]
        vote_dict = dict(votes)

        if top_label == c.neutral_label:
            return Decision(False, c.neutral_label, confidence,
                            "neutral majority / low confidence", votes=vote_dict)

        if top_count < c.min_agree:
            return Decision(False, c.neutral_label, confidence,
                            f"insufficient agreement ({top_count}/{c.min_agree})",
                            votes=vote_dict)

        if now - self._last_emit < c.refractory_s:
            return Decision(False, c.neutral_label, confidence,
                            "in refractory period", votes=vote_dict)

        self._last_emit = now
        self._window.clear()
        cmd = Command(
            action=top_label, item=None, confidence=confidence,
            trace={"rule": f"{top_count}/{c.min_agree} votes in window",
                   "votes": vote_dict},
        )
        return Decision(True, top_label, confidence,
                        f"command '{top_label}' ({top_count} votes)",
                        command=cmd, votes=vote_dict)
