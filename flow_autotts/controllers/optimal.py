"""Candidate controller for SD3.5 PickScore discovery."""

from __future__ import annotations

from flow_autotts.core.env import FlowTTSEnv
from flow_autotts.core.errors import BudgetExceededError, InvalidActionError
from flow_autotts.core.state import AnswerRecord, PreviewRecord


_EPS = 1e-9


class OptimalController:
    """Beta-scaled preview search with cheap late sibling probes."""

    def solve(self, env: FlowTTSEnv, beta: float) -> AnswerRecord:
        beta = min(max(float(beta), 0.0), 1.0)
        initial_budget = max(0, int(env.budget_left))
        schedule = self._schedule(env, beta)
        target_nfe = self._target_nfe(env, initial_budget, beta)

        try:
            if beta <= 0.0:
                root_id = env.spawn(1)[0]
                self._finish(env, root_id)
                return env.answer(rule="latest_active")

            if bool(schedule["use_low_late_route"]):
                built = self._low_late_probe_route(env, schedule, target_nfe, initial_budget)
                if not built:
                    self._complete_roots(env, schedule, target_nfe, initial_budget)
            elif bool(schedule["use_scouts"]):
                built = self._scout_prune_finish(env, schedule, target_nfe, initial_budget)
                if not built:
                    self._complete_roots(env, schedule, target_nfe, initial_budget)
            else:
                self._complete_roots(env, schedule, target_nfe, initial_budget)

            self._adaptive_full_refine(env, schedule, target_nfe, initial_budget)
            self._late_probe_refine(env, schedule, target_nfe, initial_budget)
            self._prune_active_losers(env, schedule)
            return self._answer(env)
        except (BudgetExceededError, InvalidActionError):
            return self._answer(env)

    def _schedule(self, env: FlowTTSEnv, beta: float) -> dict[str, float | int | bool]:
        root_count = 1 + int(4.0 * beta + _EPS)
        use_low_late_route = 0.0 < beta < 0.40
        use_scouts = beta >= 0.65
        scout_fraction = 0.70 - 0.10 * beta
        low_scout_fraction = 0.82 + 0.18 * beta
        return {
            "root_count": root_count,
            "use_low_late_route": use_low_late_route,
            "use_scouts": use_scouts,
            "scout_time": self._grid_time_at_or_after(env, scout_fraction),
            "scout_prune": 1 if use_scouts else 0,
            "low_scout_count": 2 + int(2.0 * beta + _EPS),
            "low_scout_time": self._grid_time_at_or_after(env, low_scout_fraction),
            "low_probe_time": self._grid_time_at_or_after(env, 0.88),
            "low_probe_children": 1 + int(8.0 * beta + _EPS),
            "late_anchor_time": 0.95,
            "deep_refine_time": 0.75,
            "local_refine_time": self._grid_time_at_or_after(env, 0.80),
            "full_refine_children": int(1.0 + 2.5 * beta + _EPS),
            "late_probe_children": int(1.0 + 6.0 * beta + _EPS),
            "late_probe_time": self._grid_time_at_or_after(env, 0.88),
            "anchor_pool": 1 + int(3.0 * beta + _EPS),
            "gap_threshold": 0.006 - 0.002 * beta,
            "prune_margin": 0.006 + 0.004 * beta,
            "mix_strength": 0.45 + 0.35 * beta,
            "uncertainty_penalty": 0.020 * (1.0 - beta),
        }

    def _target_nfe(self, env: FlowTTSEnv, initial_budget: int, beta: float) -> int:
        deterministic_cost = self._forward_cost(env, 0.0, 1.0)
        if initial_budget <= deterministic_cost:
            return initial_budget
        target = deterministic_cost + int((initial_budget - deterministic_cost) * beta)
        return min(initial_budget, max(deterministic_cost, target))

    def _low_late_probe_route(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, float | int | bool],
        target_nfe: int,
        initial_budget: int,
    ) -> bool:
        scout_time = float(schedule["low_scout_time"])
        scout_cost = self._forward_cost(env, 0.0, scout_time) + 1
        requested = max(2, int(schedule["low_scout_count"]))

        scout_count = 0
        for count in range(requested, 1, -1):
            if self._has_room(env, initial_budget, target_nfe, count * scout_cost + 1):
                scout_count = count
                break
        if scout_count < 2:
            return False

        previewed: list[PreviewRecord] = []
        for particle_id in env.spawn(scout_count):
            if not self._has_room(env, initial_budget, target_nfe, scout_cost):
                break
            try:
                self._advance_to(env, particle_id, scout_time)
                preview = self._preview(env, particle_id, target_nfe, initial_budget)
                if preview is not None and preview.score is not None:
                    previewed.append(preview)
            except (BudgetExceededError, InvalidActionError):
                break

        if not previewed:
            return False

        ranked = self._sort_previews(previewed, float(schedule["uncertainty_penalty"]))
        leader_id = ranked[0].particle_id

        finish_cost = self._forward_cost(env, self._particle_time(env, leader_id), 1.0) + 1
        if self._has_room(env, initial_budget, target_nfe, finish_cost + 1):
            try:
                self._finish(env, leader_id)
                self._preview(env, leader_id, target_nfe, initial_budget)
            except (BudgetExceededError, InvalidActionError):
                pass

        self._late_probe_refine(
            env,
            schedule,
            target_nfe,
            initial_budget,
            max_children=int(schedule["low_probe_children"]),
            target_time=float(schedule["low_probe_time"]),
            min_anchor_time=scout_time,
        )

        state = env.get_state()
        active_losers = [
            preview.particle_id
            for preview in ranked[1:]
            if preview.particle_id in state.particles
            and state.particles[preview.particle_id].status == "active"
        ]
        self._try_prune(env, active_losers)
        return True

    def _complete_roots(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, float | int | bool],
        target_nfe: int,
        initial_budget: int,
    ) -> list[int]:
        full_cost = self._forward_cost(env, 0.0, 1.0) + 1
        affordable = max(1, target_nfe // max(1, full_cost))
        root_count = max(1, min(int(schedule["root_count"]), affordable))
        completed: list[int] = []

        for particle_id in env.spawn(root_count):
            if not self._has_room(env, initial_budget, target_nfe, full_cost):
                break
            try:
                self._finish(env, particle_id)
                self._preview(env, particle_id, target_nfe, initial_budget)
                completed.append(particle_id)
            except (BudgetExceededError, InvalidActionError):
                break
        return completed

    def _scout_prune_finish(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, float | int | bool],
        target_nfe: int,
        initial_budget: int,
    ) -> list[int]:
        scout_time = float(schedule["scout_time"])
        scout_cost = self._forward_cost(env, 0.0, scout_time) + 1
        finish_cost = self._forward_cost(env, scout_time, 1.0) + 1
        requested = max(1, int(schedule["root_count"]))
        prune_count = max(0, int(schedule["scout_prune"]))

        scout_count = 0
        keep_count = 0
        for count in range(requested, 1, -1):
            keep = max(1, count - prune_count)
            needed = count * scout_cost + keep * finish_cost
            if self._spent(env, initial_budget) + needed <= target_nfe:
                scout_count = count
                keep_count = keep
                break
        if scout_count <= 1:
            return []

        previewed: list[PreviewRecord] = []
        for particle_id in env.spawn(scout_count):
            if not self._has_room(env, initial_budget, target_nfe, scout_cost):
                break
            try:
                self._advance_to(env, particle_id, scout_time)
                preview = self._preview(env, particle_id, target_nfe, initial_budget)
                if preview is not None and preview.score is not None:
                    previewed.append(preview)
            except (BudgetExceededError, InvalidActionError):
                break

        if not previewed:
            return []

        ranked = self._sort_previews(previewed, float(schedule["uncertainty_penalty"]))
        survivors = [preview.particle_id for preview in ranked[:keep_count]]
        survivor_set = set(survivors)
        losers = [
            preview.particle_id
            for preview in ranked[keep_count:]
            if preview.particle_id not in survivor_set
        ]
        self._try_prune(env, losers)

        finished: list[int] = []
        for particle_id in survivors:
            current_time = self._particle_time(env, particle_id)
            cost = self._forward_cost(env, current_time, 1.0) + 1
            if not self._has_room(env, initial_budget, target_nfe, cost):
                break
            try:
                self._finish(env, particle_id)
                self._preview(env, particle_id, target_nfe, initial_budget)
                finished.append(particle_id)
            except (BudgetExceededError, InvalidActionError):
                break
        return finished

    def _adaptive_full_refine(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, float | int | bool],
        target_nfe: int,
        initial_budget: int,
    ) -> None:
        max_children = max(0, int(schedule["full_refine_children"]))
        if max_children <= 0:
            return

        made = 0
        while made < max_children:
            ranked = self._ranked_previews(
                env,
                min_time=float(schedule["late_anchor_time"]),
                uncertainty_penalty=float(schedule["uncertainty_penalty"]),
            )
            if not ranked:
                ranked = self._ranked_previews(
                    env,
                    min_time=0.0,
                    uncertainty_penalty=float(schedule["uncertainty_penalty"]),
                )
            if not ranked:
                return

            gap = self._effective_gap(ranked, float(schedule["uncertainty_penalty"]))
            target_time = self._full_refine_time(schedule, gap, made)
            child_cost = self._child_finish_preview_cost(env, target_time)
            if not self._has_room(env, initial_budget, target_nfe, child_cost):
                fallback_time = float(schedule["local_refine_time"])
                fallback_cost = self._child_finish_preview_cost(env, fallback_time)
                if (
                    fallback_time < target_time - _EPS
                    or not self._has_room(env, initial_budget, target_nfe, fallback_cost)
                ):
                    return
                target_time = fallback_time

            anchor = self._select_anchor(ranked, schedule, gap, made)
            noise_policy, strength = self._noise_policy(schedule, gap, anchor.id == ranked[0].id)
            try:
                child_ids = env.backward(
                    anchor.id,
                    target_time=target_time,
                    noise_policy=noise_policy,
                    num_children=1,
                    strength=strength,
                )
                if not child_ids:
                    return
                child_id = child_ids[0]
                self._finish(env, child_id)
                self._preview(env, child_id, target_nfe, initial_budget)
            except (BudgetExceededError, InvalidActionError):
                return
            made += 1

    def _late_probe_refine(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, float | int | bool],
        target_nfe: int,
        initial_budget: int,
        max_children: int | None = None,
        target_time: float | None = None,
        min_anchor_time: float | None = None,
    ) -> None:
        child_count = (
            int(schedule["late_probe_children"])
            if max_children is None
            else max(0, int(max_children))
        )
        if child_count <= 0:
            return

        probe_time = float(schedule["late_probe_time"] if target_time is None else target_time)
        anchor_time = (
            float(schedule["late_anchor_time"]) if min_anchor_time is None else float(min_anchor_time)
        )
        for child_index in range(child_count):
            if not self._has_room(env, initial_budget, target_nfe, 1):
                return
            ranked = self._ranked_previews(
                env,
                min_time=anchor_time,
                uncertainty_penalty=float(schedule["uncertainty_penalty"]),
            )
            if not ranked:
                ranked = self._ranked_previews(
                    env,
                    min_time=0.0,
                    uncertainty_penalty=float(schedule["uncertainty_penalty"]),
                )
            if not ranked:
                return

            gap = self._effective_gap(ranked, float(schedule["uncertainty_penalty"]))
            anchor = self._select_anchor(ranked, schedule, gap, child_index)
            _, strength = self._noise_policy(schedule, gap, anchor.id == ranked[0].id)
            if gap <= float(schedule["gap_threshold"]):
                strength = max(0.20, strength - 0.15)
            else:
                strength = min(0.92, strength + 0.08)

            try:
                child_ids = env.backward(
                    anchor.id,
                    target_time=probe_time,
                    noise_policy="mixed_noise",
                    num_children=1,
                    strength=strength,
                )
                if not child_ids:
                    return
                self._preview(env, child_ids[0], target_nfe, initial_budget)
            except (BudgetExceededError, InvalidActionError):
                return

    def _full_refine_time(
        self,
        schedule: dict[str, float | int | bool],
        gap: float,
        child_index: int,
    ) -> float:
        if bool(schedule["use_scouts"]):
            if gap > float(schedule["gap_threshold"]) and child_index == 0:
                return float(schedule["deep_refine_time"])
            return float(schedule["local_refine_time"])
        return float(schedule["deep_refine_time"])

    def _select_anchor(
        self,
        ranked: list[PreviewRecord],
        schedule: dict[str, float | int | bool],
        gap: float,
        child_index: int,
    ) -> PreviewRecord:
        if gap > float(schedule["gap_threshold"]):
            return ranked[0]
        pool = max(1, min(len(ranked), int(schedule["anchor_pool"])))
        return ranked[child_index % pool]

    def _noise_policy(
        self,
        schedule: dict[str, float | int | bool],
        gap: float,
        is_leader: bool,
    ) -> tuple[str, float]:
        strength = float(schedule["mix_strength"])
        if gap <= float(schedule["gap_threshold"]):
            strength -= 0.20
        else:
            strength += 0.05
        if not is_leader:
            strength -= 0.12
        return "mixed_noise", min(0.95, max(0.20, strength))

    def _prune_active_losers(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, float | int | bool],
    ) -> None:
        state = env.get_state()
        active_previews = [
            preview
            for preview in state.previews.values()
            if preview.score is not None
            and preview.particle_id in state.particles
            and state.particles[preview.particle_id].status == "active"
        ]
        if len(active_previews) <= 1:
            return
        ranked = self._sort_previews(active_previews, float(schedule["uncertainty_penalty"]))
        best_score = float(ranked[0].score)
        margin = float(schedule["prune_margin"])
        keep = max(1, int(schedule["anchor_pool"]))
        losers = [
            preview.particle_id
            for preview in ranked[keep:]
            if best_score - float(preview.score) >= margin
        ]
        self._try_prune(env, losers)

    def _finish(self, env: FlowTTSEnv, particle_id: int) -> None:
        self._advance_to(env, particle_id, 1.0)

    def _advance_to(self, env: FlowTTSEnv, particle_id: int, target_time: float) -> None:
        current_time = self._particle_time(env, particle_id)
        for time in self._forward_targets(env, current_time, target_time):
            state = env.get_state()
            particle = state.particles.get(particle_id)
            if particle is None or particle.status != "active":
                return
            env.forward(particle_id, target_time=time, solver="euler")

    def _preview(
        self,
        env: FlowTTSEnv,
        particle_id: int,
        target_nfe: int,
        initial_budget: int,
    ) -> PreviewRecord | None:
        if self._has_room(env, initial_budget, target_nfe, 1):
            return env.preview(particle_id, mode="clean_anchor", scorer="default")
        return None

    def _ranked_previews(
        self,
        env: FlowTTSEnv,
        min_time: float,
        uncertainty_penalty: float,
    ) -> list[PreviewRecord]:
        state = env.get_state()
        previews = [
            preview
            for preview in state.previews.values()
            if preview.score is not None
            and preview.time >= min_time - _EPS
            and preview.particle_id in state.particles
            and state.particles[preview.particle_id].status != "pruned"
        ]
        return self._sort_previews(previews, uncertainty_penalty)

    def _sort_previews(
        self,
        previews: list[PreviewRecord],
        uncertainty_penalty: float,
    ) -> list[PreviewRecord]:
        return sorted(
            previews,
            key=lambda preview: (
                self._effective_score(preview, uncertainty_penalty),
                float(preview.score),
                float(preview.time),
                -int(preview.id),
            ),
            reverse=True,
        )

    def _effective_score(self, preview: PreviewRecord, uncertainty_penalty: float) -> float:
        return float(preview.score) - uncertainty_penalty * float(preview.uncertainty or 0.0)

    def _effective_gap(
        self,
        ranked: list[PreviewRecord],
        uncertainty_penalty: float,
    ) -> float:
        if len(ranked) < 2:
            return float("inf")
        top = self._effective_score(ranked[0], uncertainty_penalty)
        challenger = self._effective_score(ranked[1], uncertainty_penalty)
        return max(0.0, top - challenger)

    def _child_finish_preview_cost(self, env: FlowTTSEnv, target_time: float) -> int:
        return self._forward_cost(env, target_time, 1.0) + 1

    def _forward_cost(self, env: FlowTTSEnv, start_time: float, target_time: float) -> int:
        return len(self._forward_targets(env, start_time, target_time))

    def _forward_targets(
        self,
        env: FlowTTSEnv,
        start_time: float,
        target_time: float,
    ) -> list[float]:
        start = float(start_time)
        target = min(max(float(target_time), 0.0), 1.0)
        targets = [
            float(time)
            for time in env.time_grid
            if float(time) > start + _EPS and float(time) <= target + _EPS
        ]
        if target > start + _EPS and not any(abs(time - target) <= _EPS for time in targets):
            targets.append(target)
        return sorted(set(targets))

    def _grid_time_at_or_after(self, env: FlowTTSEnv, fraction: float) -> float:
        target = min(max(float(fraction), 0.0), 1.0)
        for time in sorted(float(t) for t in env.time_grid):
            if time >= target - _EPS:
                return time
        return 1.0

    def _particle_time(self, env: FlowTTSEnv, particle_id: int) -> float:
        state = env.get_state()
        return float(state.particles[particle_id].time)

    def _has_room(
        self,
        env: FlowTTSEnv,
        initial_budget: int,
        target_nfe: int,
        cost: int,
    ) -> bool:
        cost = max(0, int(cost))
        return env.budget_left >= cost and self._spent(env, initial_budget) + cost <= target_nfe

    def _spent(self, env: FlowTTSEnv, initial_budget: int) -> int:
        return max(0, int(initial_budget - env.budget_left))

    def _try_prune(self, env: FlowTTSEnv, particle_ids: list[int]) -> None:
        unique_ids = sorted(set(int(pid) for pid in particle_ids))
        if not unique_ids:
            return
        try:
            env.prune(unique_ids)
        except InvalidActionError:
            return

    def _answer(self, env: FlowTTSEnv) -> AnswerRecord:
        state = env.get_state()
        if state.previews:
            try:
                return env.answer(rule="best_preview_score")
            except (BudgetExceededError, InvalidActionError):
                pass
        return env.answer(rule="latest_active")
