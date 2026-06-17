"""Description of a multichannel signal stream.

``StreamInfo`` is the single, source-agnostic contract that every source
(simulated / LSL / replay) must produce. Downstream code never asks
"where did this come from?" -- it only reads this metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Channel "kinds" recognised by the platform. EOG is tracked separately so
# that scalp-only computations never accidentally include the eye channel.
KIND_EEG = "eeg"
KIND_EOG = "eog"
KIND_MISC = "misc"


@dataclass
class StreamInfo:
    """Metadata describing an active stream."""

    name: str
    sfreq: float
    channel_names: list[str]
    channel_kinds: list[str]
    source_kind: str = "unknown"          # simulated | lsl | replay
    units: str = "uV"

    def __post_init__(self) -> None:
        n = len(self.channel_names)
        if len(self.channel_kinds) != n:
            raise ValueError(
                f"channel_names ({n}) and channel_kinds "
                f"({len(self.channel_kinds)}) length mismatch"
            )
        if self.sfreq <= 0:
            raise ValueError(f"sfreq must be positive, got {self.sfreq}")

    @property
    def n_channels(self) -> int:
        return len(self.channel_names)

    def indices_of_kind(self, kind: str) -> list[int]:
        return [i for i, k in enumerate(self.channel_kinds) if k == kind]

    @property
    def eeg_indices(self) -> list[int]:
        return self.indices_of_kind(KIND_EEG)

    @property
    def eog_indices(self) -> list[int]:
        return self.indices_of_kind(KIND_EOG)

    def index_of(self, channel_name: str) -> int:
        return self.channel_names.index(channel_name)
