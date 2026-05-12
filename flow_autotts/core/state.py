"""Public state records exposed to controllers and evaluators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ParticleStatus = Literal["active", "completed", "pruned"]


@dataclass(frozen=True)
class ParticleSummary:
    id: int
    time: float
    parent_id: int | None
    source_anchor_id: int | None
    nfe_used: int
    status: ParticleStatus
    last_preview_id: int | None
    num_children: int


@dataclass(frozen=True)
class PreviewRecord:
    id: int
    particle_id: int
    time: float
    score: float | None
    score_dict: dict[str, float] = field(default_factory=dict)
    uncertainty: float | None = None
    drift: float | None = None
    embedding_ref: str | None = None


@dataclass(frozen=True)
class EventRecord:
    step_id: int
    action: str
    particle_ids: list[int]
    input_time: float | None
    output_time: float | None
    nfe_cost: int
    budget_left: int
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ControllerState:
    prompt: str
    budget_left: int
    active_particle_ids: list[int]
    completed_particle_ids: list[int]
    pruned_particle_ids: list[int]
    particles: dict[int, ParticleSummary]
    previews: dict[int, PreviewRecord]
    event_log: list[EventRecord]


@dataclass(frozen=True)
class AnswerRecord:
    particle_id: int | None
    preview_id: int | None
    latent: tuple[float, ...]
    reward: float | None
    nfe_used: int
    rule: str
    event_log: list[EventRecord]
    score_dict: dict[str, float] = field(default_factory=dict)
