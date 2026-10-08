"""Bookkeeping shared by every controller environment.

Concrete environments (the toy vector env and the SD3.5 env) differ in how
they store latents and integrate the flow, but they share the same budget
accounting, event log, particle lookup and PRUNE semantics.
"""

from __future__ import annotations

import math
from typing import Any, Generic, Protocol, TypeVar

from flow_autotts.core.errors import BudgetExceededError, InvalidActionError
from flow_autotts.core.state import EventRecord


class _ParticleLike(Protocol):
    status: str


ParticleT = TypeVar("ParticleT", bound=_ParticleLike)
AnchorT = TypeVar("AnchorT")


def preview_uncertainty(time: float, sde_variance: float) -> float:
    """Heuristic preview uncertainty: distance from clean time plus SDE noise."""
    base = 1.0 - float(time)
    stochastic = math.sqrt(max(0.0, float(sde_variance)))
    return max(0.0, min(1.0, base + 0.25 * stochastic))


class EpisodeBase(Generic[ParticleT, AnchorT]):
    """Budget, event-log and record bookkeeping for one prompt episode."""

    def __init__(self, budget: int) -> None:
        if budget < 0:
            raise ValueError("budget must be non-negative")
        self.budget = int(budget)
        self._particles: dict[int, ParticleT] = {}
        self._anchors: dict[int, AnchorT] = {}
        self._events: list[EventRecord] = []
        self._nfe_used = 0
        self._next_particle_id = 0
        self._next_anchor_id = 0
        self._answered = False

    @property
    def nfe_used(self) -> int:
        return self._nfe_used

    @property
    def budget_left(self) -> int:
        return self.budget - self._nfe_used

    def prune(self, particle_ids: list[int]) -> None:
        self._ensure_open()
        for particle_id in particle_ids:
            particle = self._require_particle(particle_id)
            if particle.status == "completed":
                raise InvalidActionError("cannot prune completed particles")
            particle.status = "pruned"

        self._log(
            action="PRUNE",
            particle_ids=list(particle_ids),
            input_time=None,
            output_time=None,
            nfe_cost=0,
            details={"n": len(particle_ids)},
        )

    def _new_particle_id(self) -> int:
        pid = self._next_particle_id
        self._next_particle_id += 1
        return pid

    def _new_anchor_id(self) -> int:
        aid = self._next_anchor_id
        self._next_anchor_id += 1
        return aid

    def _require_particle(self, particle_id: int, status: str | None = None) -> ParticleT:
        try:
            particle = self._particles[int(particle_id)]
        except KeyError as exc:
            raise InvalidActionError(f"unknown particle_id: {particle_id}") from exc
        if status is not None and particle.status != status:
            raise InvalidActionError(
                f"particle {particle_id} has status {particle.status}, expected {status}"
            )
        return particle

    def _require_anchor(self, anchor_id: int) -> AnchorT:
        try:
            return self._anchors[int(anchor_id)]
        except KeyError as exc:
            raise InvalidActionError(f"unknown anchor_id: {anchor_id}") from exc

    def _charge(self, nfe_cost: int) -> None:
        nfe_cost = int(nfe_cost)
        if self._nfe_used + nfe_cost > self.budget:
            raise BudgetExceededError(
                f"action would exceed budget: used {self._nfe_used}, "
                f"cost {nfe_cost}, budget {self.budget}"
            )
        self._nfe_used += nfe_cost

    def _log(
        self,
        action: str,
        particle_ids: list[int],
        input_time: float | None,
        output_time: float | None,
        nfe_cost: int,
        details: dict[str, Any],
    ) -> None:
        self._events.append(
            EventRecord(
                step_id=len(self._events),
                action=action,
                particle_ids=list(particle_ids),
                input_time=input_time,
                output_time=output_time,
                nfe_cost=int(nfe_cost),
                budget_left=self.budget_left,
                details=dict(details),
            )
        )

    def _status_lists(self) -> tuple[list[int], list[int], list[int]]:
        active: list[int] = []
        completed: list[int] = []
        pruned: list[int] = []
        for pid, particle in self._particles.items():
            if particle.status == "active":
                active.append(pid)
            elif particle.status == "completed":
                completed.append(pid)
            elif particle.status == "pruned":
                pruned.append(pid)
        return active, completed, pruned

    def _ensure_open(self) -> None:
        if self._answered:
            raise InvalidActionError("episode already answered")
