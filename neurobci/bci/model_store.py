"""Persist and load fitted models with reproducibility metadata.

A saved model is a directory containing the pickled :class:`ParadigmModel`
(fitted sklearn/pyriemann pipeline + paradigm metadata) and a
human-readable ``meta.json``. Loading checks **compatibility** with the
current stream (channels + sampling rate) and refuses silent mismatches.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from neurobci.bci.models import ParadigmModel
from neurobci.core.stream_info import StreamInfo
from neurobci.version import __version__

logger = logging.getLogger(__name__)

MODEL_FILE = "model.pkl"
META_FILE = "meta.json"


def save_model(
    model: ParadigmModel,
    root: Path | str = "models",
    extra_meta: dict | None = None,
) -> Path:
    if not model.trained:
        raise ValueError("Refusing to save an untrained model.")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(root) / f"{model.paradigm}_{model.name}_{ts}"
    path.mkdir(parents=True, exist_ok=True)

    with open(path / MODEL_FILE, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        "software_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "paradigm": model.paradigm,
        "model_name": model.name,
        "sfreq": model.sfreq,
        "channel_names": model.channel_names,
        "channel_kinds": model.channel_kinds,
        "window": asdict(model.window),
        "positive_index": model.positive_index,
        "threshold": model.threshold,
        "metrics": model.metrics,
    }
    if extra_meta:
        meta.update(extra_meta)
    (path / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Saved model to %s", path)
    return path


def load_model(path: Path | str) -> ParadigmModel:
    path = Path(path)
    with open(path / MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    if not isinstance(model, ParadigmModel):
        raise TypeError(f"{path} does not contain a ParadigmModel.")
    return model


def list_models(root: Path | str = "models") -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(
        (p for p in root.iterdir() if p.is_dir() and (p / MODEL_FILE).exists()),
        reverse=True,
    )


def check_compatibility(model: ParadigmModel, info: StreamInfo, sfreq_tol: float = 0.02) -> list[str]:
    """Return a list of incompatibility messages (empty == compatible)."""
    issues: list[str] = []
    if info.channel_names != model.channel_names:
        if set(info.channel_names) == set(model.channel_names):
            issues.append("Channel order differs from the trained model.")
        else:
            issues.append(
                "Channel set differs from the trained model "
                f"({info.n_channels} vs {len(model.channel_names)})."
            )
    if model.sfreq > 0 and abs(info.sfreq - model.sfreq) / model.sfreq > sfreq_tol:
        issues.append(
            f"Sampling rate {info.sfreq:.1f} Hz differs from trained "
            f"{model.sfreq:.1f} Hz."
        )
    return issues
