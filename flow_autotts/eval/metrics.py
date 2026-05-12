"""Metrics computed from controller answers and action logs."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from flow_autotts.core.state import AnswerRecord, EventRecord


@dataclass(frozen=True)
class EvalMetrics:
    final_reward: float | None
    nfe: int
    reward_per_nfe: float | None
    preview_calls: int
    backward_calls: int
    num_particles_spawned: int
    action_counts: dict[str, int]
    preview_final_correlation: float | None
    false_prune_rate: float | None
    wasted_nfe_rate: float | None


def compute_metrics(answer: AnswerRecord) -> EvalMetrics:
    events = answer.event_log
    counts = Counter(event.action for event in events)
    spawned = sum(int(event.details.get("n", len(event.particle_ids))) for event in events if event.action == "SPAWN")
    reward_per_nfe = None
    if answer.reward is not None and answer.nfe_used > 0:
        reward_per_nfe = answer.reward / answer.nfe_used
    return EvalMetrics(
        final_reward=answer.reward,
        nfe=answer.nfe_used,
        reward_per_nfe=reward_per_nfe,
        preview_calls=counts["PREVIEW"],
        backward_calls=counts["BACKWARD"],
        num_particles_spawned=spawned,
        action_counts=dict(counts),
        preview_final_correlation=None,
        false_prune_rate=None,
        wasted_nfe_rate=None,
    )


def event_log_to_dicts(events: list[EventRecord]) -> list[dict[str, Any]]:
    return [asdict(event) for event in events]
