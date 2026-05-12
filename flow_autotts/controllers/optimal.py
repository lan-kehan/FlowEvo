"""Candidate controller optimized for the 8 Gaussians harness."""

from __future__ import annotations

from flow_autotts.core.env import FlowTTSEnv
from flow_autotts.core.state import AnswerRecord


class OptimalController:
    """Use scored anchors to seed a late-time local preview search."""

    def solve(self, env: FlowTTSEnv, beta: float) -> AnswerRecord:
        beta = min(max(float(beta), 0.0), 1.0)
        total_preview_budget = int(8 + 120 * beta)
        total_preview_budget = min(total_preview_budget, max(1, env.budget_left))
        if beta <= 0.0:
            root_fraction = 0.75
        elif beta >= 1.0:
            root_fraction = 0.35
        else:
            root_fraction = 0.25
        root_count = max(2, min(total_preview_budget, int(total_preview_budget * root_fraction)))

        root_ids = env.spawn(root_count)
        for particle_id in root_ids:
            if env.budget_left <= 0:
                break
            env.preview(particle_id, mode="clean_anchor", scorer="default")

        state = env.get_state()
        scored_roots = [
            particle_id
            for particle_id in root_ids
            if state.particles[particle_id].last_preview_id is not None
        ]
        scored_roots.sort(
            key=lambda particle_id: state.previews[
                state.particles[particle_id].last_preview_id
            ].score
            if state.previews[state.particles[particle_id].last_preview_id].score is not None
            else float("-inf"),
            reverse=True,
        )

        keep = max(1, min(len(scored_roots), int(1 + 5 * beta)))
        survivors = scored_roots[:keep]
        remaining_previews = max(0, total_preview_budget - root_count)
        for index, particle_id in enumerate(survivors):
            preview_id = env.get_state().particles[particle_id].last_preview_id
            if preview_id is None or remaining_previews <= 0:
                continue
            share = remaining_previews // (len(survivors) - index)
            if share <= 0:
                continue
            child_ids = env.backward(
                preview_id,
                target_time=0.85,
                noise_policy="fresh_noise",
                num_children=share,
            )
            remaining_previews -= len(child_ids)
            for child_id in child_ids:
                if env.budget_left <= 0:
                    break
                env.preview(child_id, mode="clean_anchor", scorer="default")
        return env.answer(rule="best_preview_score")
