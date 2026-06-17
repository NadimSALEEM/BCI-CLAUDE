"""Configurable online/offline preprocessing and artifact detection."""

from neurobci.preprocessing.artifact_removal import ASR, ICARemoval, InterpolateBad
from neurobci.preprocessing.artifacts import (
    ArtifactEvent,
    ArtifactReport,
    ArtifactThresholds,
    detect_artifacts,
)
from neurobci.preprocessing.pipeline import MODE_CAUSAL, MODE_OFFLINE, Pipeline
from neurobci.preprocessing.stages import (
    STAGE_REGISTRY,
    AmplitudeClamp,
    BandPass,
    BandStop,
    CommonAverageReference,
    Detrend,
    HighPass,
    Laplacian,
    LowPass,
    MovingAverage,
    Notch,
    ProcessingStage,
    RobustReference,
    SavitzkyGolay,
    Standardize,
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
    "BandStop",
    "Notch",
    "CommonAverageReference",
    "RobustReference",
    "Laplacian",
    "AmplitudeClamp",
    "MovingAverage",
    "Standardize",
    "Detrend",
    "SavitzkyGolay",
    "InterpolateBad",
    "ICARemoval",
    "ASR",
    "detect_artifacts",
    "ArtifactReport",
    "ArtifactEvent",
    "ArtifactThresholds",
]
