"""Kaggle hardware preflight (nvidia-smi, CUDA, RAM, disk); fails safely with no GPU.

Phase 3 module — NOT IMPLEMENTED in Phase 0. Calling it raises ``PhaseNotImplemented``.
See master spec H3.
"""

from __future__ import annotations

from . import PhaseNotImplemented


def preflight(*args, **kwargs):
    raise PhaseNotImplemented(__name__, 3)
