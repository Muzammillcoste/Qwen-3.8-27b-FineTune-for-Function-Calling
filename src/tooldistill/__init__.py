"""ToolDistill-Lab: data-selection research code for small-LM tool-calling SFT.

Phase 0 implements inspection, normalization, validation, provenance, deduplication and
group-aware splitting. Later-phase modules are stubs that raise ``PhaseNotImplemented``.
"""

__version__ = "0.0.1"


class PhaseNotImplemented(NotImplementedError):
    """Raised by modules whose implementation belongs to a later, not-yet-approved phase."""

    def __init__(self, module: str, phase: int):
        super().__init__(
            f"{module} is scheduled for Phase {phase} and is not implemented yet. "
            "It requires explicit human approval of that phase (see master spec section K)."
        )
