"""Transparent quality-proxy scoring (tool identity, schema validity, grounding, consistency).

Phase 2 module — NOT IMPLEMENTED in Phase 0. Calling it raises ``PhaseNotImplemented``.
See master spec F1.
"""

from __future__ import annotations

from . import PhaseNotImplemented


def score_quality(*args, **kwargs):
    raise PhaseNotImplemented(__name__, 2)
