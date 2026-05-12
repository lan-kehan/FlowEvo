"""Baseline controllers from the implementation spec."""

from __future__ import annotations

from flow_autotts.core.env import FlowTTSEnv
from flow_autotts.core.errors import BudgetExceededError
from flow_autotts.core.state import AnswerRecord


def map_beta(beta: float) -> dict[str, float | int]:
    beta = min(max(float(beta), 0.0), 1.0)
    return {
        "max_particles": int(2 + 14 * beta),
        "max_preview_calls": int(2 + 12 * beta),
        "max_backward_children": int(1 + 4 * beta),
        "min_keep": int(1 + 3 * beta),
        "early_time": 0.25,
        "mid_time": 0.55,
        "late_time": 0.80,
    }


class DeterministicController:
    def solve(self, env: FlowTTSEnv, beta: float) -> AnswerRecord:
        particle_id = env.spawn(1)[0]
        for target_time in env.time_grid[1:]:
            env.forward(particle_id, target_time=target_time, solver="euler")
        return env.answer(rule="latest_active")


class BestOfNController:
    def solve(self, env: FlowTTSEnv, beta: float) -> AnswerRecord:
        requested = int(map_beta(beta)["max_particles"])
        per_particle_cost = max(1, len(env.time_grid) - 1) + 1
        affordable = max(1, env.budget // per_particle_cost)
        n = min(requested, affordable)
        ids = env.spawn(n)

        for particle_id in ids:
            for target_time in env.time_grid[1:]:
                env.forward(particle_id, target_time=target_time, solver="euler")
            env.preview(particle_id, scorer="default")

        return env.answer(rule="best_preview_score")


class SDEForwardController:
    """Best-of-N controller using Flow-GRPO-style SDE forward steps."""

    def solve(self, env: FlowTTSEnv, beta: float) -> AnswerRecord:
        beta = min(max(float(beta), 0.0), 1.0)
        requested = int(2 + 14 * beta)
        per_particle_cost = max(1, len(env.time_grid) - 1) + 1
        affordable = max(1, env.budget // per_particle_cost)
        n = min(requested, affordable)
        noise_scale = self._noise_scale(beta)
        ids = env.spawn(n)

        for particle_id in ids:
            for target_time in env.time_grid[1:]:
                env.forward(
                    particle_id,
                    target_time=target_time,
                    solver="sde",
                    cfg={"noise_scale": noise_scale, "sigma_max": 1.25, "min_time": 0.02},
                )
            env.preview(particle_id, scorer="default")

        return env.answer(rule="best_preview_score")

    @staticmethod
    def _noise_scale(beta: float) -> float:
        if beta <= 0.0:
            return 0.03
        if beta < 0.5:
            return 0.0
        if beta < 0.75:
            return 0.01
        return 0.008


class SelfRefineController:
    def solve(self, env: FlowTTSEnv, beta: float) -> AnswerRecord:
        beta = min(max(float(beta), 0.0), 1.0)
        refinement_rounds = int(1 + 3 * beta)
        particle_id = env.spawn(1)[0]

        for target_time in env.time_grid[1:]:
            state = env.get_state().particles[particle_id]
            current_time = state.time
            if current_time < 0.35:
                for _ in range(refinement_rounds):
                    preview = env.preview(
                        particle_id,
                        mode="clean_anchor",
                        scorer=None,
                    )
                    particle_id = env.backward(
                        preview.id,
                        target_time=current_time,
                        noise_policy="fresh_noise",
                        num_children=1,
                    )[0]
            env.forward(particle_id, target_time=target_time, solver="euler")

        return env.answer(rule="latest_active")


class PrismStyleFlowController:
    def solve(self, env: FlowTTSEnv, beta: float) -> AnswerRecord:
        params = map_beta(beta)
        warm_target = float(params["mid_time"])
        finish_grid = [t for t in env.time_grid if t > warm_target]
        if not finish_grid or finish_grid[-1] < 1.0:
            finish_grid.append(1.0)

        per_root_cost = 2
        requested_roots = int(params["max_particles"])
        root_budget = max(1, env.budget // max(1, per_root_cost + len(finish_grid)))
        root_count = min(requested_roots, root_budget)
        ids = env.spawn(root_count)

        previewed_ids: list[int] = []
        for particle_id in ids:
            if env.budget_left < per_root_cost:
                break
            env.forward(particle_id, target_time=warm_target, solver="euler")
            env.preview(particle_id, mode="clean_anchor", scorer="default")
            previewed_ids.append(particle_id)

        if not previewed_ids:
            return env.answer(rule="latest_active")

        state = env.get_state()
        ranked = sorted(
            previewed_ids,
            key=lambda pid: (
                state.previews[state.particles[pid].last_preview_id].score
                if state.particles[pid].last_preview_id is not None
                and state.previews[state.particles[pid].last_preview_id].score is not None
                else float("-inf")
            ),
            reverse=True,
        )
        keep = min(len(ranked), int(params["min_keep"]))
        survivors = ranked[:keep]
        env.prune([pid for pid in ranked if pid not in survivors])

        child_ids: list[int] = []
        children_per_anchor = int(params["max_backward_children"])
        for particle_id in survivors:
            preview_id = env.get_state().particles[particle_id].last_preview_id
            if preview_id is None:
                continue
            child_ids.extend(
                env.backward(
                    preview_id,
                    target_time=warm_target,
                    noise_policy="fresh_noise",
                    num_children=children_per_anchor,
                )
            )

        if not child_ids:
            return env.answer(rule="best_preview_score")

        finished: list[int] = []
        for particle_id in child_ids:
            needed = len([t for t in finish_grid if t > warm_target])
            if env.budget_left < needed + 1:
                break
            try:
                for target_time in finish_grid:
                    if env.get_state().particles[particle_id].status == "active":
                        env.forward(particle_id, target_time=target_time, solver="euler")
                env.preview(particle_id, scorer="default")
                finished.append(particle_id)
            except BudgetExceededError:
                break

        if not finished:
            return env.answer(rule="best_preview_score")
        return env.answer(rule="best_preview_score")
