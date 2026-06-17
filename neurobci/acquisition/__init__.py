"""Acquisition layer: sources, montage helpers, engine."""

from neurobci.acquisition.artifacts_inject import (
    ArtifactInjectionConfig,
    ArtifactInjector,
)
from neurobci.acquisition.base import EEGSource
from neurobci.acquisition.discovery import StreamDescription, discover_streams
from neurobci.acquisition.engine import AcquisitionEngine, build_source
from neurobci.acquisition.lsl_publisher import LSLPublisher
from neurobci.acquisition.lsl_source import LSLSource
from neurobci.acquisition.replay_source import ReplaySource
from neurobci.acquisition.simulated import SimulatedSource
from neurobci.acquisition.validation import (
    Severity,
    ValidationReport,
    validate_stream,
)

__all__ = [
    "EEGSource",
    "SimulatedSource",
    "LSLSource",
    "ReplaySource",
    "LSLPublisher",
    "ArtifactInjector",
    "ArtifactInjectionConfig",
    "AcquisitionEngine",
    "build_source",
    "discover_streams",
    "StreamDescription",
    "validate_stream",
    "ValidationReport",
    "Severity",
]
