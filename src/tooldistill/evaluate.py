"""Offline parsing of <tool_call> blocks and structured-call metrics.

Phase 3 module — NOT IMPLEMENTED in Phase 0. Calling it raises ``PhaseNotImplemented``.
See master spec I2, I3.
"""

from __future__ import annotations

from . import PhaseNotImplemented


def evaluate_predictions(*args, **kwargs):
    raise PhaseNotImplemented(__name__, 3)
