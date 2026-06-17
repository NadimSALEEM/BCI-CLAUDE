"""NeuroBCI - a modular real-time EEG / BCI research platform.

This package is intentionally split into independent layers so that
acquisition, processing, modelling, decision-making and the user
interface can evolve (and be tested) in isolation:

    neurobci.config        configuration schema + profile management
    neurobci.core          logging, ring buffer, app state, event bus
    neurobci.acquisition   stream sources (simulated / LSL / replay)
    neurobci.quality       live signal-quality metrics
    neurobci.ui            PyQt5 desktop interface

The platform is a *research tool*, not a medical device, and only
decodes pre-defined, calibrated EEG responses.
"""

from neurobci.version import __version__, APP_NAME

__all__ = ["__version__", "APP_NAME"]
