"""Reproducible analysis configuration.

Every analysis (stats or ML) is described by a single JSON-serialisable
:class:`AnalysisConfig` capturing *everything* needed to reproduce it: the data
source, the selection (conditions / channels / windows / bands), feature
settings, the test/model parameters, the analysis level and design, the random
seed, and a snapshot of the preprocessing state. The same object is stored in
the analysis history and embedded in every exported report.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# Analysis levels / designs (free-form but enumerated for the UI).
LEVELS = (
    "single_subject_trial",
    "single_subject_condition",
    "multi_subject",
)
DESIGNS = (
    "within_subject",
    "between_subject",
    "mixed",
    "exploratory",
    "hypothesis_driven",
)


@dataclass
class AnalysisConfig:
    """A complete, reproducible description of one analysis run."""

    kind: str                       # "stats" | "ml"
    analysis_type: str              # e.g. "paired_ttest", "cluster_perm", "classification"
    created: str = field(default_factory=_now_iso)
    seed: int = 42
    level: str = "single_subject_trial"
    design: str = "exploratory"
    data_source: dict = field(default_factory=dict)   # file, session, subject(s), sfreq, n_ch
    selection: dict = field(default_factory=dict)      # conditions, channels/ROI, time, bands
    features: dict = field(default_factory=dict)       # feature-extraction settings
    params: dict = field(default_factory=dict)         # test/model parameters
    preprocessing: dict = field(default_factory=dict)  # snapshot of pipeline state
    notes: str = ""

    # ----- serialisation ------------------------------------------------- #

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AnalysisConfig":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "AnalysisConfig":
        return cls.from_dict(json.loads(text))

    def fingerprint(self) -> str:
        """Stable hash of the *inputs* (everything but the timestamp/notes).

        Two configs with the same fingerprint produce the same result, so the
        history can flag duplicate runs and re-runs can be verified.
        """
        payload = self.to_dict()
        payload.pop("created", None)
        payload.pop("notes", None)
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]
