"""Discover available LSL streams on the network.

Used by the Acquisition workspace's "Scan" button. Everything here is
defensive: ``pylsl`` is imported lazily, the resolve call is time-bounded,
and any failure (no liblsl, no streams, network hiccup) yields an empty
list rather than an exception.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StreamDescription:
    name: str
    stype: str
    n_channels: int
    sfreq: float
    source_id: str
    hostname: str

    @property
    def label(self) -> str:
        return (
            f"{self.name}  [{self.stype}]  "
            f"{self.n_channels}ch @ {self.sfreq:.0f}Hz  ({self.hostname})"
        )


def discover_streams(timeout: float = 1.5) -> list[StreamDescription]:
    """Return descriptions of all resolvable LSL streams (possibly empty)."""

    try:
        import pylsl
    except Exception as exc:  # noqa: BLE001
        logger.warning("LSL discovery unavailable: %s", exc)
        return []

    try:
        infos = pylsl.resolve_streams(wait_time=timeout)
    except Exception:  # noqa: BLE001
        logger.exception("LSL stream resolution failed.")
        return []

    out: list[StreamDescription] = []
    for info in infos:
        try:
            out.append(
                StreamDescription(
                    name=info.name(),
                    stype=info.type(),
                    n_channels=info.channel_count(),
                    sfreq=float(info.nominal_srate() or 0.0),
                    source_id=info.source_id(),
                    hostname=info.hostname(),
                )
            )
        except Exception:  # noqa: BLE001
            logger.debug("Skipping a stream with unreadable metadata.", exc_info=True)
    logger.info("LSL discovery found %d stream(s).", len(out))
    return out
