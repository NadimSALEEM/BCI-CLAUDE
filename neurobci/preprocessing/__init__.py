"""Configurable online/offline preprocessing and artifact detection."""

from neurobci.preprocessing.artifacts import (
    ArtifactEvent,
    ArtifactReport,
    ArtifactThresholds,
    detect_artifacts,
)
from neurobci.preprocessing.pipeline import MODE_CAUSAL, MODE_OFFLINE, Pipeline
from neurobci.preprocessing.stages import (
    STAGE_REGISTRY,
    BandPass,
    CommonAverageReference,
    Detrend,
    HighPass,
    LowPass,
    Notch,
    ProcessingStage,
    make_stage,
)

__all__ = [
    "Pipeline",
    "MODE_CAUSAL",
    "MODE_OFFLINE",
    "ProcessingStage",
    "make_stage",
    "STAGE_REGISTRY",
    "HighPass",
    "LowPass",
    "BandPass",
    "Notch",
    "CommonAverageReference",
    "Detrend",
    "detect_artifacts",
    "ArtifactReport",
    "ArtifactEvent",
    "ArtifactThresholds",
]
