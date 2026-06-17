"""BCI paradigms (P300 first; others register later)."""

from neurobci.paradigms.base import (
    EpochWindow,
    Paradigm,
    available_paradigms,
    get_paradigm,
    register_paradigm,
)
from neurobci.paradigms.p300 import NONTARGET, TARGET, P300Paradigm
from neurobci.paradigms.errp import ErrPParadigm
from neurobci.paradigms.motor_imagery import MotorImageryParadigm
from neurobci.paradigms.ssvep import SSVEPParadigm
from neurobci.paradigms.synthetic import (
    SyntheticDataset,
    make_p300_dataset,
    make_p300_selection_run,
)
from neurobci.paradigms.synthetic_paradigms import (
    make_errp_dataset,
    make_mi_dataset,
    make_ssvep_dataset,
)

__all__ = [
    "Paradigm",
    "EpochWindow",
    "get_paradigm",
    "available_paradigms",
    "register_paradigm",
    "P300Paradigm",
    "MotorImageryParadigm",
    "SSVEPParadigm",
    "ErrPParadigm",
    "TARGET",
    "NONTARGET",
    "SyntheticDataset",
    "make_p300_dataset",
    "make_p300_selection_run",
    "make_mi_dataset",
    "make_ssvep_dataset",
    "make_errp_dataset",
]
