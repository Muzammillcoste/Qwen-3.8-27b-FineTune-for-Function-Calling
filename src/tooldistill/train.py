"""QLoRA adapter training (one run per session, checkpoint/resume).

Phase 3 module — NOT IMPLEMENTED in Phase 0. Calling it raises ``PhaseNotImplemented``.
See master spec H1-H3.
"""

from __future__ import annotations

from . import PhaseNotImplemented


def train_adapter(*args, **kwargs):
    raise PhaseNotImplemented(__name__, 3)
