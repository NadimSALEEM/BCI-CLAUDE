"""Validate an incoming stream against the expected montage.

Before a stream is trusted for calibration / online use, its channel
count, names, order, sampling rate and EOG handling are checked against
the configured montage. The result is a structured report the UI can show
and later phases can use as a gate -- problems are *surfaced*, never
silently corrected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from neurobci.config.schema import ChannelConfig
from neurobci.core.stream_info import KIND_EOG, StreamInfo


class Severity(str, Enum):
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Issue:
    severity: Severity
    code: str
    message: str


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)

    def add(self, severity: Severity, code: str, message: str) -> None:
        self.issues.append(Issue(severity, code, message))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @property
    def is_valid(self) -> bool:
        """True if there are no ERROR-level issues (warnings are tolerated)."""
        return not self.errors


def validate_stream(
    info: StreamInfo,
    expected: ChannelConfig,
    expected_sfreq: float,
    sfreq_tolerance: float = 0.05,
) -> ValidationReport:
    """Compare a live :class:`StreamInfo` to the expected montage/rate."""

    report = ValidationReport()
    exp_names = expected.all_channels
    got_names = info.channel_names

    # --- sampling rate --------------------------------------------------- #
    if expected_sfreq > 0:
        rel = abs(info.sfreq - expected_sfreq) / expected_sfreq
        if rel > sfreq_tolerance:
            report.add(
                Severity.ERROR, "sfreq_mismatch",
                f"Sampling rate {info.sfreq:.1f} Hz differs from expected "
                f"{expected_sfreq:.1f} Hz by {rel*100:.0f}%.",
            )

    # --- channel count --------------------------------------------------- #
    if info.n_channels != len(exp_names):
        report.add(
            Severity.ERROR, "count_mismatch",
            f"Stream has {info.n_channels} channels; montage expects "
            f"{len(exp_names)}.",
        )

    # --- duplicate names ------------------------------------------------- #
    dupes = {n for n in got_names if got_names.count(n) > 1}
    if dupes:
        report.add(
            Severity.ERROR, "duplicate_names",
            f"Duplicate channel name(s): {', '.join(sorted(dupes))}.",
        )

    # --- name set / order ------------------------------------------------ #
    got_set, exp_set = set(got_names), set(exp_names)
    missing = [n for n in exp_names if n not in got_set]
    extra = [n for n in got_names if n not in exp_set]
    if missing:
        report.add(
            Severity.WARNING, "missing_channels",
            f"Expected channel(s) absent from stream: {', '.join(missing)}.",
        )
    if extra:
        report.add(
            Severity.WARNING, "unexpected_channels",
            f"Stream has channel(s) not in montage: {', '.join(extra)}.",
        )
    if not missing and not extra and got_names != exp_names:
        report.add(
            Severity.WARNING, "order_mismatch",
            "Channels match by name but are in a different order; they will "
            "be reordered logically by name.",
        )

    # --- EOG handling ---------------------------------------------------- #
    has_eog = KIND_EOG in info.channel_kinds
    expects_eog = len(expected.eog_channels) > 0
    if expects_eog and not has_eog:
        report.add(
            Severity.WARNING, "eog_missing",
            "No channel is marked as EOG; eye-artifact handling will be "
            "limited. Mark the EOG channel in the Channels workspace.",
        )

    if report.is_valid and not report.warnings:
        report.add(Severity.OK, "ok", "Stream matches the expected montage.")
    return report
