"""Deterministic inference (do_sample=False) for base and adapters.

Phase 3 module — NOT IMPLEMENTED in Phase 0. Calling it raises ``PhaseNotImplemented``.
See master spec I1.
"""

from __future__ import annotations

from . import PhaseNotImplemented


def generate(*args, **kwargs):
    raise PhaseNotImplemented(__name__, 3)
