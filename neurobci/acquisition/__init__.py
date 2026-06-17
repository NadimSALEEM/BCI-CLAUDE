"""Acquisition layer: sources, montage helpers, engine."""

from neurobci.acquisition.base import EEGSource
from neurobci.acquisition.discovery import StreamDescription, discover_streams
from neurobci.acquisition.engine import AcquisitionEngine, build_source
from neurobci.acquisition.lsl_source import LSLSource
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
    "AcquisitionEngine",
    "build_source",
    "discover_streams",
    "StreamDescription",
    "validate_stream",
    "ValidationReport",
    "Severity",
]
