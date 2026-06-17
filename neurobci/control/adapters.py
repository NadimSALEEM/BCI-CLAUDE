"""Control adapters: the safe boundary between BCI commands and actions.

An adapter is the *only* thing that turns a :class:`Command` into an
effect. Phase 5 ships **internal** adapters only (they move an on-screen
object inside the app); external/OS control is intentionally absent and is
additionally gated off by :class:`~neurobci.control.safety.SafetyConfig`.

New adapters (robots, games, external apps) are added by subclassing
:class:`ControlAdapter` -- the decision/safety/history machinery is reused
unchanged.
"""

from __future__ import annotations

import abc

import numpy as np

from neurobci.control.types import Command


class ControlAdapter(abc.ABC):
    is_external: bool = False

    @abc.abstractmethod
    def execute(self, command: Command) -> None:
        ...

    def describe(self) -> str:
        return self.__class__.__name__


class NullAdapter(ControlAdapter):
    """Does nothing -- used for 'test mode' (predict but don't act)."""

    def execute(self, command: Command) -> None:  # noqa: D401
        return None


class BoardModel:
    """A tiny grid world the internal demo controls.

    Items are arranged on a circle; selecting an item moves the cursor one
    step toward that item's position. Selecting the centre item ('home')
    re-centres. This gives a visible, safe 'navigation' effect.
    """

    def __init__(self, n_items: int = 6, radius: float = 1.0) -> None:
        self.n_items = n_items
        angles = np.linspace(0, 2 * np.pi, n_items, endpoint=False)
        self.item_xy = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        self.cursor = np.zeros(2)
        self.last_item: int | None = None
        self.trail: list[tuple[float, float]] = [tuple(self.cursor)]

    def select(self, item: int, step: float = 0.5) -> None:
        target = self.item_xy[item]
        self.cursor = self.cursor + step * (target - self.cursor)
        self.last_item = item
        self.trail.append(tuple(self.cursor))
        if len(self.trail) > 200:
            self.trail.pop(0)

    def reset(self) -> None:
        self.cursor = np.zeros(2)
        self.last_item = None
        self.trail = [tuple(self.cursor)]


class InternalBoardAdapter(ControlAdapter):
    """Moves the cursor in a :class:`BoardModel` (in-app, safe)."""

    is_external = False

    def __init__(self, board: BoardModel) -> None:
        self.board = board

    def execute(self, command: Command) -> None:
        if command.item is not None:
            self.board.select(command.item)
