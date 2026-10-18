"""Typed record structures for the canonical normalized xLAM representation (spec E3)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "tooldistill.normalized.v1"
UNKNOWN_TEACHER = "unknown"


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class NormalizedRecord:
    sample_id: str
    source_dataset: str
    source_revision: str
    original_id: int | None
    id_origin: str  # "source_field:<name>" | "source_row_index" | "missing"
    source_row_index: int  # position in the ORIGINAL unshuffled source file
    teacher_source: str
    query: str
    tools: list[dict[str, Any]]  # OpenAI/Qwen style {"type":"function","function":{...}}
    expected_calls: list[ToolCall]
    call_count: int
    raw_hash: str
    normalized_hash: str
    source_group_id: str | None = None
    exact_dup_key: str | None = None
    duplicate_of: str | None = None
    quality_proxy: float | None = None
    split: str | None = None
    schema_version: str = SCHEMA_VERSION
    metadata: dict[str, Any] = field(
        default_factory=lambda: {"conversion_notes": [], "validation_flags": []}
    )

    @property
    def flags(self) -> list[str]:
        return self.metadata.setdefault("validation_flags", [])

    @property
    def notes(self) -> list[str]:
        return self.metadata.setdefault("conversion_notes", [])

    def add_flag(self, flag: str) -> None:
        if flag not in self.flags:
            self.flags.append(flag)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NormalizedRecord:
        d = dict(d)
        d["expected_calls"] = [ToolCall(**c) for c in d["expected_calls"]]
        return cls(**d)


@dataclass
class RejectedRecord:
    """A source row that could not be normalized. Logged with a reason; never silently dropped."""

    source_row_index: int
    original_id: Any
    reason: str
    detail: str
    raw_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
