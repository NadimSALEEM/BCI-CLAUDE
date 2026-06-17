"""Configuration schema and profile management."""

from neurobci.config.schema import (
    AcquisitionConfig,
    AppConfig,
    ChannelConfig,
    ControlConfig,
    DecisionConfig,
    LoggingConfig,
    PreprocessingConfig,
    RecordingConfig,
    SafetyConfig,
    SelectionConfig,
    SimulationConfig,
    UIConfig,
)
from neurobci.config.manager import ConfigManager, ConfigError

__all__ = [
    "AppConfig",
    "ChannelConfig",
    "AcquisitionConfig",
    "SimulationConfig",
    "PreprocessingConfig",
    "ControlConfig",
    "SelectionConfig",
    "DecisionConfig",
    "SafetyConfig",
    "RecordingConfig",
    "UIConfig",
    "LoggingConfig",
    "ConfigManager",
    "ConfigError",
]
