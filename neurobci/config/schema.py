"""Typed configuration schema.

Configuration is expressed as nested ``dataclasses`` so that:

* every parameter has a documented default in one place,
* the structure is self-describing (used by the UI to build forms),
* (de)serialisation to/from plain JSON is mechanical and safe -- we
  never ``eval`` or import anything named in a config file.

The schema is deliberately broader than Phase 1 needs in a couple of
places (e.g. paradigm name) so that later phases can extend it without
breaking existing profiles: unknown keys are ignored on load and missing
keys fall back to defaults.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, get_type_hints

# --------------------------------------------------------------------------- #
# Individual configuration sections
# --------------------------------------------------------------------------- #


@dataclass
class ChannelConfig:
    """Electrode montage description.

    ``eeg_channels`` are scalp electrodes; ``eog_channels`` are explicitly
    *not* treated as scalp EEG by downstream quality / feature code.
    """

    montage_name: str = "enobio19"
    eeg_channels: list[str] = field(
        default_factory=lambda: [
            "P7", "P4", "Cz", "Pz", "P3", "P8", "O1", "O2", "T8", "F8",
            "C4", "F4", "Fp2", "Fz", "C3", "F3", "Fp1", "T7", "F7",
        ]
    )
    eog_channels: list[str] = field(default_factory=lambda: ["EOG"])

    @property
    def all_channels(self) -> list[str]:
        return list(self.eeg_channels) + list(self.eog_channels)

    @property
    def n_channels(self) -> int:
        return len(self.all_channels)


@dataclass
class AcquisitionConfig:
    """How raw samples enter the system."""

    source_type: str = "simulated"          # simulated | lsl | replay
    lsl_stream_name: str = ""               # empty => auto-discover
    lsl_stream_type: str = "EEG"
    expected_sfreq: float = 500.0           # Enobio default
    buffer_seconds: float = 30.0            # ring-buffer history kept in RAM
    pull_interval_s: float = 0.02           # acquisition loop period (~50 Hz)
    sfreq_tolerance: float = 0.05           # +/- fraction flagged as unstable


@dataclass
class SimulationConfig:
    """Parameters for the synthetic EEG generator.

    Amplitudes are in microvolts. ``seed`` makes runs reproducible so that
    tests have known ground truth.
    """

    seed: int = 1234
    alpha_amp_uv: float = 12.0              # posterior alpha (~10 Hz)
    theta_amp_uv: float = 6.0
    beta_amp_uv: float = 4.0
    pink_noise_uv: float = 8.0              # 1/f background
    white_noise_uv: float = 3.0
    line_noise_uv: float = 2.0              # mains interference
    line_freq_hz: float = 50.0
    drift_uv: float = 5.0                   # slow electrode drift
    blink_rate_hz: float = 0.3             # eye-blinks per second (Poisson)
    blink_amp_uv: float = 90.0
    # Fault injection (for testing robustness of downstream code):
    flat_channels: list[str] = field(default_factory=list)
    noisy_channels: list[str] = field(default_factory=list)


@dataclass
class UIConfig:
    """Visualisation / interaction preferences."""

    raw_window_s: float = 10.0              # seconds shown in the raw view
    refresh_hz: float = 30.0               # plot refresh rate (decoupled!)
    default_scale_uv: float = 75.0          # +/- range per trace
    theme: str = "dark"
    basic_mode: bool = True                 # hide advanced parameters


def default_pipeline() -> list[dict]:
    """A sensible general-purpose causal pipeline.

    Each entry is a plain JSON-friendly dict ``{type, enabled, params}`` so
    the preprocessing configuration round-trips through JSON without any
    custom (de)serialisation. The :mod:`neurobci.preprocessing` package
    turns these into stage objects.
    """

    return [
        {"type": "highpass", "enabled": True, "params": {"cutoff_hz": 0.5, "order": 4}},
        {"type": "notch", "enabled": True, "params": {"freq_hz": 50.0, "quality": 30.0}},
        {"type": "lowpass", "enabled": True, "params": {"cutoff_hz": 40.0, "order": 4}},
        {"type": "car", "enabled": True, "params": {}},
    ]


@dataclass
class PreprocessingConfig:
    """Ordered, configurable online/offline preprocessing.

    ``mode`` selects causal (real-time safe; never uses future samples) or
    offline (zero-phase, may use the whole window). Stages that are not
    real-time safe are skipped in causal mode.
    """

    enabled: bool = True
    mode: str = "causal"                    # causal | offline
    stages: list[dict] = field(default_factory=default_pipeline)


@dataclass
class SelectionConfig:
    """P300 selection decision logic (accumulate evidence over flashes)."""

    min_repetitions: int = 3                # min flashes/item before deciding
    max_repetitions: int = 15               # hard cap before abstaining
    confidence_margin: float = 0.12         # best-minus-second mean score
    min_confidence: float = 0.5             # best item's mean target score
    timeout_s: float = 12.0


@dataclass
class DecisionConfig:
    """Streaming class-decision logic (active BCI / continuous control)."""

    confidence_threshold: float = 0.6       # min prob to consider a class
    vote_window: int = 5                    # sliding majority-vote window
    min_agree: int = 3                      # consecutive/within-window agreement
    refractory_s: float = 1.0               # quiet period after a command
    neutral_label: str = "none"             # the no-command class name


@dataclass
class SafetyConfig:
    """Gates that must pass before any command is executed."""

    external_control_enabled: bool = False  # external adapters off by default
    min_confidence: float = 0.5
    max_commands_per_min: float = 60.0      # rate limit
    require_connected: bool = True
    require_quality: bool = True


@dataclass
class ControlConfig:
    n_items: int = 6
    test_mode: bool = False                 # predict but do not execute
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)


@dataclass
class SpectralConfig:
    """Spectral / cognitive-state monitoring parameters."""

    window_s: float = 4.0                  # analysis window length
    nperseg_s: float = 1.0                 # Welch segment length (seconds)
    fmax_hz: float = 45.0                  # top of the analysed range
    show_indices: bool = True


@dataclass
class RecordingConfig:
    """Where and how sessions are saved (see neurobci.recording)."""

    directory: str = "recordings"           # session-folder root
    participant_id: str = "anon"            # pseudonymous identifier only
    notes: str = ""


@dataclass
class LoggingConfig:
    level: str = "INFO"                     # DEBUG | INFO | WARNING | ERROR
    to_file: bool = True
    directory: str = "logs"


# --------------------------------------------------------------------------- #
# Top-level container
# --------------------------------------------------------------------------- #


@dataclass
class AppConfig:
    """Root configuration object persisted as one JSON document."""

    schema_version: int = 1
    profile_name: str = "default"
    paradigm: str = "none"                  # reserved for later phases
    channels: ChannelConfig = field(default_factory=ChannelConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # ----- (de)serialisation -------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        """Build a config from a (possibly partial) plain dict.

        Unknown keys are ignored; missing keys keep their defaults. This
        keeps old profiles loadable after the schema grows.
        """

        return _build_dataclass(cls, data or {})


# --------------------------------------------------------------------------- #
# Generic, safe dataclass <- dict construction
# --------------------------------------------------------------------------- #


def _build_dataclass(cls: type, data: dict[str, Any]) -> Any:
    if not dataclasses.is_dataclass(cls):
        return data
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = hints.get(f.name)
        if dataclasses.is_dataclass(ftype) and isinstance(value, dict):
            kwargs[f.name] = _build_dataclass(ftype, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)
