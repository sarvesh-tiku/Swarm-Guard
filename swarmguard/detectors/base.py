"""Shared detector output type.

Detectors flag *risk-relevant structure* for human review. None of them is a
"danger score": every Detection states what it does NOT prove.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Detection:
    name: str
    triggered: bool
    confidence: str                 # "low" | "moderate" | "high"
    why: str                        # why it triggered (or why not)
    does_not_prove: str
    affected_agents: list[str] = field(default_factory=list)
    affected_resources: list[str] = field(default_factory=list)
    evidence_event_ids: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
