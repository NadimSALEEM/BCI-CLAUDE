"""Topographic layout and scalp interpolation.

Uses a standard normalised 10-20 layout (nose up, unit head radius) so a
per-channel value vector can be rendered as a scalp map. Channels without a
known position (e.g. EOG) or flagged as bad are simply not used as
interpolation anchors -- artifact-aware by construction.

A small curated 10-20 table covers the common cap; any other standard
electrode name (``Oz``, ``POz``, ``FCz``, ``CP3`` …) is resolved on demand
from MNE's ``standard_1020`` montage, rescaled to match the curated layout,
so renaming generic channels to real positions "just works".
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

# Normalised 10-20 positions: x = left(-)/right(+), y = back(-)/front(+),
# unit head radius. Approximate but standard topographic layout.
POS_1020: dict[str, tuple[float, float]] = {
    "Fp1": (-0.27, 0.91), "Fp2": (0.27, 0.91),
    "F7": (-0.81, 0.59), "F3": (-0.40, 0.67), "Fz": (0.0, 0.72),
    "F4": (0.40, 0.67), "F8": (0.81, 0.59),
    "T7": (-1.0, 0.0), "C3": (-0.5, 0.0), "Cz": (0.0, 0.0),
    "C4": (0.5, 0.0), "T8": (1.0, 0.0),
    "P7": (-0.81, -0.59), "P3": (-0.40, -0.67), "Pz": (0.0, -0.72),
    "P4": (0.40, -0.67), "P8": (0.81, -0.59),
    "O1": (-0.27, -0.91), "O2": (0.27, -0.91),
    # Common aliases.
    "T3": (-1.0, 0.0), "T4": (1.0, 0.0), "T5": (-0.81, -0.59), "T6": (0.81, -0.59),
}

# Lazily-built case-insensitive lookup: curated table extended with MNE's
# standard_1020 montage (rescaled to the curated layout).
_POS_LUT: dict[str, tuple[float, float]] | None = None


def _position_lut() -> dict[str, tuple[float, float]]:
    global _POS_LUT
    if _POS_LUT is not None:
        return _POS_LUT
    lut = {name.lower(): xy for name, xy in POS_1020.items()}
    try:
        import mne

        montage = mne.channels.make_standard_montage("standard_1020")
        ch_pos = montage.get_positions()["ch_pos"]
        xy = {n: (float(p[0]), float(p[1])) for n, p in ch_pos.items()}
        # Rescale MNE coords onto the curated layout using electrodes common
        # to both (median radius ratio), so the two sets are consistent.
        ratios = []
        for n, (mx, my) in xy.items():
            cur = lut.get(n.lower())
            mr = float(np.hypot(mx, my))
            if cur is not None and mr > 1e-6:
                ratios.append(float(np.hypot(*cur)) / mr)
        scale = float(np.median(ratios)) if ratios else 1.0
        for n, (mx, my) in xy.items():
            x, y = mx * scale, my * scale
            r = float(np.hypot(x, y))
            if r > 1.0:  # keep rim electrodes (Oz, Iz…) on the head disc
                x, y = x / r, y / r
            lut.setdefault(n.lower(), (x, y))
    except Exception:  # noqa: BLE001 - MNE optional / any layout hiccup
        logger.debug("Could not extend electrode positions from MNE.", exc_info=True)
    _POS_LUT = lut
    return _POS_LUT


def channel_positions_2d(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(positions (n,2), found_mask (n,))`` for the given channels.

    Matching is case-insensitive and spans the curated 10-20 table plus the
    MNE-derived extension, so real montage names resolve to scalp positions.
    """
    lut = _position_lut()
    pos = np.full((len(names), 2), np.nan)
    found = np.zeros(len(names), dtype=bool)
    for i, n in enumerate(names):
        xy = lut.get((n or "").strip().lower())
        if xy is not None:
            pos[i] = xy
            found[i] = True
    return pos, found


def interpolate_topomap(
    values: np.ndarray,
    positions: np.ndarray,
    found: np.ndarray,
    res: int = 64,
    smooth_sigma: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate channel values onto a head-disc grid.

    Returns ``(grid_x, grid_y, grid_z)`` where ``grid_z`` is NaN outside the
    unit head circle. Only channels with a known position and a finite value
    are used as anchors. A light Gaussian blur (``smooth_sigma`` grid cells,
    scaled to the resolution) removes interpolation facets for a smooth map.
    """
    use = found & np.isfinite(values)
    gx, gy = np.meshgrid(np.linspace(-1.1, 1.1, res), np.linspace(-1.1, 1.1, res))
    if use.sum() < 3:
        return gx, gy, np.full_like(gx, np.nan)

    pts = positions[use]
    vals = np.asarray(values, dtype=float)[use]
    grid = griddata(pts, vals, (gx, gy), method="cubic")
    # Fill cubic NaNs with nearest so the blur has a complete field to work on.
    holes = np.isnan(grid)
    if holes.any():
        grid[holes] = griddata(pts, vals, (gx[holes], gy[holes]), method="nearest")
    if smooth_sigma > 0:
        grid = gaussian_filter(grid, sigma=smooth_sigma * res / 64.0)
    grid[(gx**2 + gy**2) > 1.05**2] = np.nan
    return gx, gy, grid
