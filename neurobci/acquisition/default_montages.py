"""Montage helpers and channel classification.

Knows how to turn a :class:`~neurobci.config.schema.ChannelConfig` into a
:class:`~neurobci.core.stream_info.StreamInfo`, and provides coarse scalp
"region" weights used by the simulator to place rhythms and artifacts
realistically (alpha posterior, blinks frontal, etc.).

The electrode-name primitives (:func:`region_of`, :func:`hemisphere_of`,
:func:`classify_channel`) live in :mod:`neurobci.core.electrodes` and are
re-exported here for backward compatibility.
"""

from __future__ import annotations

from neurobci.config.schema import ChannelConfig
from neurobci.core.electrodes import (  # noqa: F401  (re-exported)
    REGION_CENTRAL,
    REGION_FRONTAL,
    REGION_OCCIPITAL,
    REGION_OTHER,
    REGION_PARIETAL,
    REGION_TEMPORAL,
    classify_channel,
    classify_channels,
    hemisphere_of,
    is_frontal,
    region_of,
)
from neurobci.core.stream_info import KIND_EEG, KIND_EOG, StreamInfo


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


# Standard montage name presets, used to relabel a stream whose channels
# arrive as generic "Ch1.."/"eeg1.." into real 10-20 positions. Order matches
# the usual acquisition order for each cap.
MONTAGE_PRESETS: dict[str, list[str]] = {
    "10-20 (19 EEG)": [
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
        "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
    ],
    "Enobio 20 (19 EEG + EOG)": [
        "P7", "P4", "Cz", "Pz", "P3", "P8", "O1", "O2", "T8", "F8", "C4",
        "F4", "Fp2", "Fz", "C3", "F3", "Fp1", "T7", "F7", "EOG",
    ],
    "10-10 (32 EEG)": [
        "Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8",
        "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
        "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8",
        "PO3", "PO4", "O1", "Oz", "O2",
    ],
    "Midline (Fz Cz Pz Oz)": ["Fz", "Cz", "Pz", "Oz"],
}


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
