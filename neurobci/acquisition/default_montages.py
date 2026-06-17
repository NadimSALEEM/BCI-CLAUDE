"""Montage helpers and channel classification.

Knows how to turn a :class:`~neurobci.config.schema.ChannelConfig` into a
:class:`~neurobci.core.stream_info.StreamInfo`, and provides coarse scalp
"region" weights used by the simulator to place rhythms and artifacts
realistically (alpha posterior, blinks frontal, etc.).
"""

from __future__ import annotations

from neurobci.config.schema import ChannelConfig
from neurobci.core.stream_info import KIND_EEG, KIND_EOG, StreamInfo

# Scalp regions inferred from 10-20 electrode prefixes.
REGION_FRONTAL = "frontal"
REGION_CENTRAL = "central"
REGION_PARIETAL = "parietal"
REGION_OCCIPITAL = "occipital"
REGION_TEMPORAL = "temporal"
REGION_OTHER = "other"


def hemisphere_of(channel: str) -> str:
    """Return 'L', 'R' or 'M' (midline) from a 10-20 name.

    In the 10-20 system odd-numbered electrodes are on the left, even on the
    right, and 'z' electrodes are midline.
    """
    name = channel.upper()
    for ch in reversed(name):
        if ch.isdigit():
            return "L" if int(ch) % 2 == 1 else "R"
        if ch == "Z":
            return "M"
    return "M"


def region_of(channel: str) -> str:
    """Classify a 10-20 channel name into a coarse scalp region."""

    name = channel.upper()
    if name.startswith(("FP", "AF", "F")):
        # Temporal F7/F8 are frontotemporal; keep them frontal for weights.
        return REGION_FRONTAL
    if name.startswith("C"):
        return REGION_CENTRAL
    if name.startswith("P"):
        return REGION_PARIETAL
    if name.startswith("O"):
        return REGION_OCCIPITAL
    if name.startswith("T"):
        return REGION_TEMPORAL
    return REGION_OTHER


def build_stream_info(
    channels: ChannelConfig, sfreq: float, source_kind: str, name: str = "NeuroBCI"
) -> StreamInfo:
    """Construct a :class:`StreamInfo` from a channel configuration."""

    eeg = set(channels.eeg_channels)
    eog = set(channels.eog_channels)
    names = channels.all_channels
    kinds = [KIND_EEG if n in eeg else KIND_EOG if n in eog else KIND_EEG for n in names]
    return StreamInfo(
        name=name,
        sfreq=sfreq,
        channel_names=names,
        channel_kinds=kinds,
        source_kind=source_kind,
        units="uV",
    )


# Relative alpha (~10 Hz) strength by region: strongest posterior.
ALPHA_REGION_WEIGHT = {
    REGION_OCCIPITAL: 1.0,
    REGION_PARIETAL: 0.8,
    REGION_TEMPORAL: 0.4,
    REGION_CENTRAL: 0.3,
    REGION_FRONTAL: 0.2,
    REGION_OTHER: 0.3,
}

# Relative blink projection by region: strongest frontal (and EOG handled
# separately as a near-full-amplitude channel).
BLINK_REGION_WEIGHT = {
    REGION_FRONTAL: 1.0,
    REGION_CENTRAL: 0.35,
    REGION_TEMPORAL: 0.3,
    REGION_PARIETAL: 0.15,
    REGION_OCCIPITAL: 0.08,
    REGION_OTHER: 0.3,
}
