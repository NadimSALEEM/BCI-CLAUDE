"""Flexible command mapping.

Decouples *what the BCI decoded* (an item index or class) from *what action
it triggers*. The same selection paradigm can drive a speller, a cursor, a
menu or a robot just by swapping the map -- new control applications need no
changes to the decoding or decision code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from neurobci.control.types import Command


@dataclass
class CommandMap:
    mapping: dict[int, str] = field(default_factory=dict)
    default: str = "none"

    def action_for(self, key: int) -> str:
        return self.mapping.get(int(key), self.default)

    def to_command(self, key: int, confidence: float, margin: float = 0.0,
                   trace: dict | None = None) -> Command:
        action = self.action_for(key)
        return Command(action=action, item=int(key), confidence=confidence,
                       margin=margin, trace=trace or {})

    # ----- presets ------------------------------------------------------- #

    @classmethod
    def identity(cls, n: int) -> "CommandMap":
        return cls({i: f"item_{i}" for i in range(n)})

    @classmethod
    def directional(cls) -> "CommandMap":
        return cls({0: "up", 1: "right", 2: "down", 3: "left",
                    4: "select", 5: "back"})
