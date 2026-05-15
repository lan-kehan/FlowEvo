"""Candidate controller for SD3.5 PickScore discovery."""

from __future__ import annotations

from flow_autotts.core.env import FlowTTSEnv
from flow_autotts.core.errors import BudgetExceededError, InvalidActionError
from flow_autotts.core.state import AnswerRecord, PreviewRecord


class OptimalController:
    """Completed-anchor best-of-N with beta-scaled late refinement."""

    def solve(self, env: FlowTTSEnv, beta: float) -> AnswerRecord:
        beta = min(max(float(beta), 0.0), 1.0)
        initial_budget = max(0, int(env.budget_left))

        root_id = env.spawn(1)[0]
        try:
            self._finish(env, root_id)
        except BudgetExceededError:
            return env.answer(rule="latest_active")

        if beta <= 0.0 or env.budget_left <= 0:
            return env.answer(rule="latest_active")

        target_nfe = self._target_nfe(initial_budget, beta, spent=initial_budget - env.budget_left)
        self._preview(env, root_id, target_nfe, initial_budget)

        step_count = max(1, len(env.time_grid) - 1)
        full_candidate_cost = step_count + 1
        full_candidates = max(1, min(5, 1 + int(4 * beta), target_nfe // full_candidate_cost))
        extra_roots = max(0, full_candidates - 1)
        if extra_roots > 0:
            for particle_id in env.spawn(extra_roots):
                if self._spent(env, initial_budget) + full_candidate_cost > target_nfe:
                    break
                try:
                    self._finish(env, particle_id)
                    self._preview(env, particle_id, target_nfe, initial_budget)
                except (BudgetExceededError, InvalidActionError):
                    break

        self._refine_from_best_previews(env, beta, target_nfe, initial_budget)
        return env.answer(rule="best_preview_score")

    def _target_nfe(self, initial_budget: int, beta: float, spent: int) -> int:
        if initial_budget <= spent:
            return spent
        extra = int((initial_budget - spent) * beta)
        if beta > 0.0:
            extra = max(1, extra)
        return min(initial_budget, spent + extra)

    def _finish(self, env: FlowTTSEnv, particle_id: int) -> None:
        state = env.get_state()
        current_time = state.particles[particle_id].time
        for target_time in env.time_grid:
            if target_time > current_time:
                env.forward(particle_id, target_time=target_time, solver="euler")
                current_time = target_time

    def _preview(
        self,
        env: FlowTTSEnv,
        particle_id: int,
        target_nfe: int,
        initial_budget: int,
    ) -> None:
        if self._spent(env, initial_budget) + 1 <= target_nfe and env.budget_left > 0:
            env.preview(particle_id, mode="clean_anchor", scorer="default")

    def _refine_from_best_previews(
        self,
        env: FlowTTSEnv,
        beta: float,
        target_nfe: int,
        initial_budget: int,
    ) -> None:
        target_time = self._backward_time(beta)
        noise_policy = "mixed_noise" if beta >= 0.6 else "fresh_noise"
        strength = 0.35 if noise_policy == "mixed_noise" else 1.0
        max_children = max(0, int(1 + 4 * beta))

        children_made = 0
        while children_made < max_children:
            previews = self._ranked_previews(env)
            if not previews:
                return
            anchor = previews[children_made % len(previews)]
            child_cost = self._child_finish_preview_cost(env, target_time)
            if self._spent(env, initial_budget) + child_cost > target_nfe:
                return
            try:
                child_id = env.backward(
                    anchor.id,
                    target_time=target_time,
                    noise_policy=noise_policy,
                    num_children=1,
                    strength=strength,
                )[0]
                self._finish(env, child_id)
                self._preview(env, child_id, target_nfe, initial_budget)
            except (BudgetExceededError, InvalidActionError):
                return
            children_made += 1

    def _ranked_previews(self, env: FlowTTSEnv) -> list[PreviewRecord]:
        state = env.get_state()
        previews = [
            preview
            for preview in state.previews.values()
            if preview.score is not None
            and preview.particle_id in state.particles
            and state.particles[preview.particle_id].status != "pruned"
        ]
        return sorted(
            previews,
            key=lambda preview: (
                float(preview.score),
                -float(preview.uncertainty or 0.0),
                float(preview.time),
            ),
            reverse=True,
        )

    def _child_finish_preview_cost(self, env: FlowTTSEnv, target_time: float) -> int:
        forward_steps = sum(1 for time in env.time_grid if time > target_time)
        return max(1, forward_steps) + 1

    def _backward_time(self, beta: float) -> float:
        if beta >= 0.8:
            return 0.65
        if beta >= 0.5:
            return 0.75
        return 0.85

    def _spent(self, env: FlowTTSEnv, initial_budget: int) -> int:
        return max(0, int(initial_budget - env.budget_left))

