"""Electrode-name knowledge shared across the platform.

Pure, dependency-light helpers that interpret 10-20 / 10-10 electrode
labels: which scalp *region* a channel sits in, which hemisphere, and -- the
key entry point for ingesting arbitrary recordings -- what *kind* a channel
is (scalp ``eeg``, ``eog``, or non-neural ``misc``) from its name and any
declared type.

Name-based classification is what lets the platform auto-detect montages of
any size (16/32/64+) coming from XDF / FIF / LSL, even when per-channel type
metadata is missing or a blanket ``"EEG"``. Keeping it here (in ``core``,
which imports almost nothing) means acquisition, preprocessing and recording
can all share one definition without import cycles.
"""

from __future__ import annotations

import re

from neurobci.core.stream_info import KIND_EEG, KIND_EOG, KIND_MISC

# Scalp regions inferred from 10-20 electrode prefixes.
REGION_FRONTAL = "frontal"
REGION_CENTRAL = "central"
REGION_PARIETAL = "parietal"
REGION_OCCIPITAL = "occipital"
REGION_TEMPORAL = "temporal"
REGION_OTHER = "other"

# A scalp electrode = a known region prefix followed by a number or 'z'
# (the midline). Longer prefixes are listed first so e.g. "FC5"/"PO7"/"TP8"
# are matched as such rather than as "F"/"P"/"T". This generalises across
# every standard montage without enumerating individual electrodes.
_SCALP_RE = re.compile(
    r"^(FP|AF|FT|FC|TP|CP|PO|F|C|T|P|O|I)(Z|\d{1,2})$", re.IGNORECASE
)

# Reference / ground leads that ride along in the data but are not scalp EEG.
_REF_NAMES = {"a1", "a2", "m1", "m2", "ref", "gnd", "reference", "ground"}

# Non-neural auxiliary channels, matched by name prefix (covers e.g. "ECG",
# "EMG1", "STI014", "TRIGGER", "GSR", "ACC_X").
_MISC_PREFIXES = (
    "ecg", "ekg", "emg", "gsr", "eda", "resp", "ppg", "pulse", "temp",
    "trig", "trg", "stim", "sti", "status", "marker", "event",
    "acc", "gyr", "aux", "bip", "mast", "photo", "audio",
)

# Declared channel/stream types we trust verbatim (lower-cased).
_EOG_TYPES = {"eog", "veog", "heog"}
_MISC_TYPES = {
    "ecg", "ekg", "emg", "gsr", "eda", "resp", "ppg", "stim", "trig",
    "trigger", "marker", "markers", "aux", "bio", "ref", "mag", "grad",
}


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


def is_frontal(channel: str) -> bool:
    """True for frontal electrodes (where eye-blinks project most strongly)."""
    return region_of(channel) == REGION_FRONTAL


def classify_channel(name: str, declared_type: str | None = None) -> str:
    """Best-effort ``eeg`` / ``eog`` / ``misc`` classification for a channel.

    Combines any declared type with the electrode name. A specific declared
    type (``EOG``, ``ECG``, ``STIM``, …) is trusted outright; otherwise the
    name decides, which is what makes auto-detection work when a recording
    types everything as a blanket ``"EEG"`` or omits types entirely. Unknown
    labels fall back to scalp ``eeg``.
    """
    label = (name or "").strip()
    low = label.lower()
    t = (declared_type or "").strip().lower()

    # 1. Specific, trustworthy declared types win immediately.
    if t in _EOG_TYPES or "eog" in t:
        return KIND_EOG
    if t in _MISC_TYPES:
        return KIND_MISC

    # 2. Name-based detection (robust to missing / blanket types).
    if "eog" in low:
        return KIND_EOG
    if low in _REF_NAMES or low.startswith(_MISC_PREFIXES):
        return KIND_MISC
    if _SCALP_RE.match(label):
        return KIND_EEG

    # 3. Fall back to a vague declared type, else assume scalp EEG.
    if t == "misc":
        return KIND_MISC
    return KIND_EEG


def classify_channels(
    names: list[str], declared_types: list[str | None] | None = None
) -> list[str]:
    """Vectorised :func:`classify_channel` over a montage."""
    types = declared_types if declared_types is not None else [None] * len(names)
    return [classify_channel(n, t) for n, t in zip(names, types)]
