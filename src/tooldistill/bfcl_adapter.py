"""Bridge to the OFFICIAL BFCL evaluator (bfcl-eval); never a home-grown substitute.

Phase 3 module — NOT IMPLEMENTED in Phase 0. Calling it raises ``PhaseNotImplemented``.
See spec B2, I3 (docs/STUDY_DESIGN.md).
"""

from __future__ import annotations

from . import PhaseNotImplemented


def run_bfcl_subset(*args, **kwargs):
    raise PhaseNotImplemented(__name__, 3)
