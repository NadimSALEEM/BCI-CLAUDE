"""Online performance metrics for the control demo.

Tracks the BCI-relevant quantities (not just accuracy): correct vs
incorrect commands, abstentions, commands per minute and latency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from neurobci.control.types import CommandRecord


@dataclass
class OnlineMetrics:
    start_time: float = field(default_factory=time.time)
    n_executed: int = 0
    n_rejected: int = 0
    n_abstentions: int = 0
    n_correct: int = 0
    n_incorrect: int = 0
    latencies: list[float] = field(default_factory=list)

    def record_command(self, rec: CommandRecord) -> None:
        if rec.executed:
            self.n_executed += 1
            if rec.latency_s:
                self.latencies.append(rec.latency_s)
        else:
            self.n_rejected += 1
        if rec.correct is True:
            self.n_correct += 1
        elif rec.correct is False:
            self.n_incorrect += 1

    def record_abstention(self) -> None:
        self.n_abstentions += 1

    def reset(self) -> None:
        self.__init__()

    # ----- derived ------------------------------------------------------- #

    @property
    def elapsed_min(self) -> float:
        return max((time.time() - self.start_time) / 60.0, 1e-9)

    @property
    def accuracy(self) -> float | None:
        total = self.n_correct + self.n_incorrect
        return self.n_correct / total if total else None

    @property
    def commands_per_min(self) -> float:
        return self.n_executed / self.elapsed_min

    @property
    def abstention_rate(self) -> float | None:
        total = self.n_executed + self.n_rejected + self.n_abstentions
        return self.n_abstentions / total if total else None

    @property
    def mean_latency(self) -> float | None:
        return sum(self.latencies) / len(self.latencies) if self.latencies else None

    def summary(self) -> str:
        acc = f"{self.accuracy:.2f}" if self.accuracy is not None else "-"
        lat = f"{self.mean_latency:.2f}s" if self.mean_latency is not None else "-"
        return (
            f"executed {self.n_executed}  rejected {self.n_rejected}  "
            f"abstained {self.n_abstentions}  acc {acc}  "
            f"{self.commands_per_min:.1f}/min  lat {lat}"
        )
