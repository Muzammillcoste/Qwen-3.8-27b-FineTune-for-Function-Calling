"""Subset selection (random_1k, quality_1k, quality_diverse_1k, random_3k) fitted on TRAIN only.

Phase 2 module — NOT IMPLEMENTED in Phase 0. Calling it raises ``PhaseNotImplemented``.
See master spec F3, G1.
"""

from __future__ import annotations

from . import PhaseNotImplemented


def select_subsets(*args, **kwargs):
    raise PhaseNotImplemented(__name__, 2)
