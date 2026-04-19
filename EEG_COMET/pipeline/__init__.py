"""Pipeline package for EEG-COMET.

Exposes a stable :class:`PipelineCallbacks` protocol and a :class:`PipelineRunner`
facade that drives each analysis stage (preprocessing, clustering, labeling,
backfitting, feature extraction, source localization, microstate-source
identification) through a uniform interface.
"""

from .callbacks import NullCallbacks, PipelineCallbacks
from .runner import PipelineRunner

__all__ = ["NullCallbacks", "PipelineCallbacks", "PipelineRunner"]
