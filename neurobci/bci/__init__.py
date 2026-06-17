"""BCI decoding: epoching, features, models, evaluation, calibration, online."""

from neurobci.bci.calibration import (
    CalibrationResult,
    epochs_from_recording,
    run_simulated_calibration,
    train_and_select,
)
from neurobci.bci.epoching import EpochSet, extract_epochs, onsets_from_timestamps
from neurobci.bci.evaluation import EvaluationResult, compare_models, cross_validate
from neurobci.bci.model_store import (
    check_compatibility,
    list_models,
    load_model,
    save_model,
)
from neurobci.bci.models import MODEL_NAMES, ParadigmModel, build_pipeline, make_model
from neurobci.bci.online import ItemAccumulator, OnlineP300Decoder

__all__ = [
    "extract_epochs",
    "onsets_from_timestamps",
    "EpochSet",
    "build_pipeline",
    "make_model",
    "ParadigmModel",
    "MODEL_NAMES",
    "cross_validate",
    "compare_models",
    "EvaluationResult",
    "save_model",
    "load_model",
    "list_models",
    "check_compatibility",
    "run_simulated_calibration",
    "train_and_select",
    "epochs_from_recording",
    "CalibrationResult",
    "OnlineP300Decoder",
    "ItemAccumulator",
]
