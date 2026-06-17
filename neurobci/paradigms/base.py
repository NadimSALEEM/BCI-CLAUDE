"""Paradigm abstraction.

A *paradigm* defines everything specific to one BCI approach so that the
generic machinery (epoching, training, evaluation, online decoding) stays
paradigm-agnostic and new paradigms can be added without touching it:

* the class labels and which one is the "positive"/target class,
* the marker labels used during calibration,
* the epoch window (tmin/tmax, baseline, rejection threshold),
* which models to try,
* how to turn per-stimulus scores into a final selection.

Phase 4 ships :class:`~neurobci.paradigms.p300.P300Paradigm`; later phases
register SSVEP, motor-imagery, ErrP, etc.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass
class EpochWindow:
    """Epoching parameters, in seconds relative to each stimulus marker."""

    tmin: float = -0.1
    tmax: float = 0.6
    baseline: tuple[float, float] | None = (-0.1, 0.0)
    reject_uv: float = 150.0          # peak-to-peak rejection per epoch

    def n_times(self, sfreq: float) -> int:
        return int(round((self.tmax - self.tmin) * sfreq)) + 1


class Paradigm(abc.ABC):
    """Base class for BCI paradigms."""

    name: str = "base"
    family: str = "reactive"          # reactive | active | passive

    @property
    @abc.abstractmethod
    def class_labels(self) -> list[str]:
        """Ordered class names; index == integer label used by models."""

    @property
    @abc.abstractmethod
    def positive_label(self) -> str:
        """The class of interest (e.g. 'target')."""

    @property
    @abc.abstractmethod
    def window(self) -> EpochWindow:
        ...

    @property
    @abc.abstractmethod
    def model_names(self) -> list[str]:
        """Model builder keys to compare during calibration."""

    @property
    def positive_index(self) -> int:
        return self.class_labels.index(self.positive_label)

    def label_to_int(self, label: str) -> int:
        return self.class_labels.index(label)


_REGISTRY: dict[str, type[Paradigm]] = {}


def register_paradigm(cls: type[Paradigm]) -> type[Paradigm]:
    _REGISTRY[cls.name] = cls
    return cls


def get_paradigm(name: str) -> Paradigm:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown paradigm: {name!r} (have {list(_REGISTRY)})")
    return _REGISTRY[name]()


def available_paradigms() -> list[str]:
    return sorted(_REGISTRY)
