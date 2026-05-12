"""Online Flow-TTS sampling environment."""

from __future__ import annotations

import math
import random
from typing import Any, Sequence

from flow_autotts.core.errors import BudgetExceededError, InvalidActionError
from flow_autotts.core.model_api import FlowModelProtocol, IdentityVAE, ScorerProtocol
from flow_autotts.core.state import (
    AnswerRecord,
    ControllerState,
    EventRecord,
    ParticleSummary,
    PreviewRecord,
)
from flow_autotts.core.trajectory import AnchorRecord, ParticleRecord


_TIME_EPS = 1e-9


def _as_vector(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(x) for x in values)


def _add(a: Sequence[float], b: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(x) + float(y) for x, y in zip(a, b, strict=True))


def _sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(x) - float(y) for x, y in zip(a, b, strict=True))


def _mul(scale: float, v: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(scale) * float(x) for x in v)


def _lerp_noise(time: float, clean: Sequence[float], noise: Sequence[float]) -> tuple[float, ...]:
    return _add(_mul(time, clean), _mul(1.0 - time, noise))


class FlowTTSEnv:
    """Constrained online action environment for controller discovery."""

    def __init__(
        self,
        model: FlowModelProtocol,
        vae: Any | None,
        scorer: ScorerProtocol | None,
        prompt: str,
        budget: int,
        time_grid: Sequence[float],
        seed: int,
        latent_shape: tuple[int, ...],
    ) -> None:
        if budget < 0:
            raise ValueError("budget must be non-negative")
        if not latent_shape or math.prod(latent_shape) <= 0:
            raise ValueError("latent_shape must contain at least one dimension")
        if not time_grid:
            raise ValueError("time_grid must not be empty")

        self.model = model
        self.vae = vae if vae is not None else IdentityVAE()
        self.scorer = scorer
        self.prompt = prompt
        self.budget = int(budget)
        self.time_grid = tuple(float(t) for t in time_grid)
        self.latent_shape = tuple(int(x) for x in latent_shape)
        self.seed = int(seed)
        self.rng = random.Random(seed)

        self._latent_dim = math.prod(self.latent_shape)
        self._particles: dict[int, ParticleRecord] = {}
        self._anchors: dict[int, AnchorRecord] = {}
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

    def get_state(self) -> ControllerState:
        particles = {
            pid: ParticleSummary(
                id=p.id,
                time=p.time,
                parent_id=p.parent_id,
                source_anchor_id=p.source_anchor_id,
                nfe_used=p.nfe_used,
                status=p.status,
                last_preview_id=p.last_preview_id,
                num_children=p.num_children,
            )
            for pid, p in self._particles.items()
        }
        previews = {
            aid: PreviewRecord(
                id=a.id,
                particle_id=a.particle_id,
                time=a.time,
                score=a.score,
                score_dict=dict(a.score_dict),
                uncertainty=a.uncertainty,
                drift=a.drift,
                embedding_ref=a.embedding_ref,
            )
            for aid, a in self._anchors.items()
        }
        active, completed, pruned = self._status_lists()
        return ControllerState(
            prompt=self.prompt,
            budget_left=self.budget_left,
            active_particle_ids=active,
            completed_particle_ids=completed,
            pruned_particle_ids=pruned,
            particles=particles,
            previews=previews,
            event_log=list(self._events),
        )

    def spawn(self, n: int) -> list[int]:
        self._ensure_open()
        if n <= 0:
            raise InvalidActionError("SPAWN requires n > 0")

        ids = []
        for _ in range(n):
            pid = self._next_particle_id
            self._next_particle_id += 1
            z0 = tuple(self.rng.gauss(0.0, 1.0) for _ in range(self._latent_dim))
            self._particles[pid] = ParticleRecord(
                id=pid,
                z=z0,
                time=0.0,
                parent_id=None,
                source_anchor_id=None,
                nfe_used=0,
                status="active",
            )
            ids.append(pid)

        self._log(
            action="SPAWN",
            particle_ids=ids,
            input_time=None,
            output_time=0.0,
            nfe_cost=0,
            details={"n": n},
        )
        return ids

    def forward(
        self,
        particle_id: int,
        target_time: float,
        solver: str = "euler",
        cfg: Any | None = None,
    ) -> ParticleSummary:
        self._ensure_open()
        particle = self._require_particle(particle_id, status="active")
        target_time = float(target_time)
        if target_time <= particle.time + _TIME_EPS:
            raise InvalidActionError(
                f"FORWARD target_time must be greater than current time {particle.time}"
            )
        if target_time > 1.0 + _TIME_EPS:
            raise InvalidActionError("FORWARD target_time must be <= 1.0")
        target_time = min(target_time, 1.0)

        if solver == "euler":
            cost = 1
            velocity = self._velocity(particle.z, particle.time)
            z_next = _add(particle.z, _mul(target_time - particle.time, velocity))
            sde_details: dict[str, float] = {}
            sde_variance_next = particle.sde_variance
        elif solver == "heun":
            cost = 2
            v0 = self._velocity(particle.z, particle.time)
            z_predict = _add(particle.z, _mul(target_time - particle.time, v0))
            v1 = self._velocity(z_predict, target_time)
            avg_velocity = tuple((a + b) * 0.5 for a, b in zip(v0, v1, strict=True))
            z_next = _add(particle.z, _mul(target_time - particle.time, avg_velocity))
            sde_details = {}
            sde_variance_next = particle.sde_variance
        elif solver == "sde":
            cost = 1
            velocity = self._velocity(particle.z, particle.time)
            dt = target_time - particle.time
            noise_scale, sigma_max, min_time = self._parse_sde_cfg(cfg)
            sigma = self._sde_sigma(particle.time, noise_scale, sigma_max, min_time)
            score = self._flow_score(particle.z, particle.time, velocity, min_time)
            drift = _add(velocity, _mul(0.5 * sigma * sigma, score))
            noise_std = sigma * math.sqrt(dt)
            diffusion = tuple(self.rng.gauss(0.0, noise_std) for _ in range(self._latent_dim))
            z_next = _add(_add(particle.z, _mul(dt, drift)), diffusion)
            sde_variance_next = particle.sde_variance + noise_std * noise_std
            sde_details = {
                "noise_scale": noise_scale,
                "sigma": sigma,
                "noise_std": noise_std,
                "sde_variance": sde_variance_next,
            }
        else:
            raise InvalidActionError(f"unsupported solver: {solver}")

        self._charge(cost)
        input_time = particle.time
        particle.z = z_next
        particle.sde_variance = sde_variance_next
        particle.time = target_time
        particle.nfe_used += cost
        if target_time >= 1.0 - _TIME_EPS:
            particle.time = 1.0
            particle.status = "completed"

        self._log(
            action="FORWARD",
            particle_ids=[particle_id],
            input_time=input_time,
            output_time=particle.time,
            nfe_cost=cost,
            details={"solver": solver, "cfg": cfg} | sde_details,
        )
        return self.get_state().particles[particle_id]

    def preview(
        self,
        particle_id: int,
        mode: str = "clean_anchor",
        scorer: str | None = "default",
    ) -> PreviewRecord:
        self._ensure_open()
        if mode != "clean_anchor":
            raise InvalidActionError(f"unsupported preview mode: {mode}")
        particle = self._require_particle(particle_id)
        if particle.status == "pruned":
            raise InvalidActionError("cannot preview a pruned particle")

        self._charge(1)
        velocity = self._velocity(particle.z, particle.time)
        z1_hat = _add(particle.z, _mul(1.0 - particle.time, velocity))
        z0_hat = _sub(particle.z, _mul(particle.time, velocity))

        score: float | None = None
        score_dict: dict[str, float] = {}
        if scorer is not None and self.scorer is not None:
            score, score_dict = self.scorer.score(self.vae.decode(z1_hat), self.prompt)
            score = float(score)
            score_dict = {str(k): float(v) for k, v in score_dict.items()}

        aid = self._next_anchor_id
        self._next_anchor_id += 1
        anchor = AnchorRecord(
            id=aid,
            particle_id=particle_id,
            time=particle.time,
            z1_hat=z1_hat,
            z0_hat=z0_hat,
            score=score,
            score_dict=score_dict,
            uncertainty=self._preview_uncertainty(particle.time, particle.sde_variance),
            drift=math.sqrt(sum((a - b) ** 2 for a, b in zip(z1_hat, particle.z, strict=True))),
            embedding_ref=None,
        )
        self._anchors[aid] = anchor
        particle.last_preview_id = aid
        particle.nfe_used += 1

        self._log(
            action="PREVIEW",
            particle_ids=[particle_id],
            input_time=particle.time,
            output_time=particle.time,
            nfe_cost=1,
            details={
                "anchor_id": aid,
                "mode": mode,
                "score": score,
                "uncertainty": anchor.uncertainty,
            },
        )
        return self.get_state().previews[aid]

    def backward(
        self,
        anchor_id: int,
        target_time: float,
        noise_policy: str = "fresh_noise",
        num_children: int = 1,
        mask: Any | None = None,
        strength: float = 1.0,
    ) -> list[int]:
        self._ensure_open()
        if num_children <= 0:
            raise InvalidActionError("BACKWARD requires num_children > 0")
        target_time = float(target_time)
        if target_time < -_TIME_EPS or target_time > 1.0 + _TIME_EPS:
            raise InvalidActionError("BACKWARD target_time must be in [0, 1]")
        target_time = min(max(target_time, 0.0), 1.0)
        anchor = self._require_anchor(anchor_id)
        source = self._require_particle(anchor.particle_id)
        mask_tuple = self._normalize_mask(mask)

        child_ids = []
        for _ in range(num_children):
            noise = self._noise_for_policy(anchor, noise_policy, strength)
            child_z = _lerp_noise(target_time, anchor.z1_hat, noise)
            if mask_tuple is not None:
                child_z = tuple(
                    new if selected else old
                    for selected, new, old in zip(mask_tuple, child_z, source.z, strict=True)
                )

            pid = self._next_particle_id
            self._next_particle_id += 1
            self._particles[pid] = ParticleRecord(
                id=pid,
                z=child_z,
                time=target_time,
                parent_id=anchor.particle_id,
                source_anchor_id=anchor_id,
                nfe_used=0,
                status="completed" if target_time >= 1.0 - _TIME_EPS else "active",
            )
            child_ids.append(pid)

        source.num_children += num_children
        self._log(
            action="BACKWARD",
            particle_ids=child_ids,
            input_time=anchor.time,
            output_time=target_time,
            nfe_cost=0,
            details={
                "anchor_id": anchor_id,
                "noise_policy": noise_policy,
                "num_children": num_children,
                "strength": strength,
            },
        )
        return child_ids

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

    def answer(self, rule: str = "best_preview_score") -> AnswerRecord:
        self._ensure_open()
        if rule == "best_preview_score":
            particle_id, preview_id, latent, reward, score_dict = self._answer_best_preview()
        elif rule == "latest_active":
            particle_id, preview_id, latent, reward, score_dict = self._answer_latest_active()
        else:
            raise InvalidActionError(f"unsupported answer rule: {rule}")

        self._answered = True
        self._log(
            action="ANSWER",
            particle_ids=[] if particle_id is None else [particle_id],
            input_time=None,
            output_time=1.0,
            nfe_cost=0,
            details={"rule": rule, "preview_id": preview_id, "reward": reward},
        )
        return AnswerRecord(
            particle_id=particle_id,
            preview_id=preview_id,
            latent=latent,
            reward=reward,
            nfe_used=self._nfe_used,
            rule=rule,
            event_log=list(self._events),
            score_dict=score_dict,
        )

    def _answer_best_preview(
        self,
    ) -> tuple[int | None, int | None, tuple[float, ...], float | None, dict[str, float]]:
        if not self._anchors:
            return self._answer_latest_active()
        anchors = list(self._anchors.values())
        scored = [a for a in anchors if a.score is not None]
        if scored:
            chosen = max(scored, key=lambda a: float(a.score))
        else:
            chosen = max(anchors, key=lambda a: a.id)
        reward, score_dict = self._score_final(chosen.z1_hat, chosen.score, chosen.score_dict)
        return chosen.particle_id, chosen.id, chosen.z1_hat, reward, score_dict

    def _answer_latest_active(
        self,
    ) -> tuple[int | None, int | None, tuple[float, ...], float | None, dict[str, float]]:
        candidates = [
            p for p in self._particles.values() if p.status in {"active", "completed"}
        ]
        if not candidates:
            return None, None, tuple(), None, {}
        chosen = max(candidates, key=lambda p: (p.time, p.id))
        if chosen.status == "active" and chosen.time < 1.0 - _TIME_EPS:
            self._finish_particle(chosen.id)
            chosen = self._particles[chosen.id]
        reward, score_dict = self._score_final(chosen.z, None, {})
        return chosen.id, chosen.last_preview_id, chosen.z, reward, score_dict

    def _finish_particle(self, particle_id: int) -> None:
        particle = self._require_particle(particle_id, status="active")
        next_times = [t for t in self.time_grid if t > particle.time + _TIME_EPS]
        if not any(t >= 1.0 - _TIME_EPS for t in next_times):
            next_times.append(1.0)
        for target in next_times:
            if self._particles[particle_id].status != "active":
                break
            self.forward(particle_id, target_time=target, solver="euler")

    def _score_final(
        self,
        latent: Sequence[float],
        fallback_score: float | None,
        fallback_dict: dict[str, float],
    ) -> tuple[float | None, dict[str, float]]:
        if self.scorer is None:
            return fallback_score, dict(fallback_dict)
        score, score_dict = self.scorer.score(self.vae.decode(latent), self.prompt)
        return float(score), {str(k): float(v) for k, v in score_dict.items()}

    def _velocity(self, z: Sequence[float], time: float) -> tuple[float, ...]:
        velocity = _as_vector(self.model.velocity(z, float(time), self.prompt))
        if len(velocity) != self._latent_dim:
            raise InvalidActionError(
                f"model velocity has dimension {len(velocity)}, expected {self._latent_dim}"
            )
        return velocity

    def _parse_sde_cfg(self, cfg: Any | None) -> tuple[float, float, float]:
        if cfg is None:
            return 0.30, 1.25, 0.02
        if isinstance(cfg, (int, float)):
            return float(cfg), 1.25, 0.02
        if isinstance(cfg, dict):
            noise_scale = float(cfg.get("noise_scale", 0.30))
            sigma_max = float(cfg.get("sigma_max", 1.25))
            min_time = float(cfg.get("min_time", 0.02))
            return noise_scale, sigma_max, min_time
        raise InvalidActionError("SDE cfg must be None, a number, or a dict")

    def _sde_sigma(
        self,
        time: float,
        noise_scale: float,
        sigma_max: float,
        min_time: float,
    ) -> float:
        t = min(max(float(time), min_time), 1.0 - min_time)
        sigma = float(noise_scale) * math.sqrt(max(1.0 - t, min_time) / max(t, min_time))
        return min(max(sigma, 0.0), max(float(sigma_max), 0.0))

    def _flow_score(
        self,
        z: Sequence[float],
        time: float,
        velocity: Sequence[float],
        min_time: float,
    ) -> tuple[float, ...]:
        # Our convention is z_t=(1-t)z_noise + t*z_clean. Applying the
        # Flow-GRPO derivation with s=1-t gives score = (t*u - z)/(1-t).
        denom = max(1.0 - float(time), min_time)
        return tuple((float(time) * u - x) / denom for x, u in zip(z, velocity, strict=True))

    def _noise_for_policy(
        self,
        anchor: AnchorRecord,
        noise_policy: str,
        strength: float,
    ) -> tuple[float, ...]:
        if noise_policy == "inferred_noise":
            return anchor.z0_hat
        if noise_policy == "fresh_noise":
            return self._fresh_noise()
        if noise_policy == "mixed_noise":
            lam = min(max(float(strength), 0.0), 1.0)
            fresh = self._fresh_noise()
            return _add(_mul(lam, anchor.z0_hat), _mul(1.0 - lam, fresh))
        raise InvalidActionError(f"unsupported noise_policy: {noise_policy}")

    def _fresh_noise(self) -> tuple[float, ...]:
        return tuple(self.rng.gauss(0.0, 1.0) for _ in range(self._latent_dim))

    def _normalize_mask(self, mask: Any | None) -> tuple[bool, ...] | None:
        if mask is None:
            return None
        mask_tuple = tuple(bool(x) for x in mask)
        if len(mask_tuple) != self._latent_dim:
            raise InvalidActionError(
                f"mask has dimension {len(mask_tuple)}, expected {self._latent_dim}"
            )
        return mask_tuple

    def _preview_uncertainty(self, time: float, sde_variance: float) -> float:
        base = 1.0 - float(time)
        stochastic = math.sqrt(max(0.0, float(sde_variance)))
        return max(0.0, min(1.0, base + 0.25 * stochastic))

    def _require_particle(
        self,
        particle_id: int,
        status: str | None = None,
    ) -> ParticleRecord:
        try:
            particle = self._particles[int(particle_id)]
        except KeyError as exc:
            raise InvalidActionError(f"unknown particle_id: {particle_id}") from exc
        if status is not None and particle.status != status:
            raise InvalidActionError(
                f"particle {particle_id} has status {particle.status}, expected {status}"
            )
        return particle

    def _require_anchor(self, anchor_id: int) -> AnchorRecord:
        try:
            return self._anchors[int(anchor_id)]
        except KeyError as exc:
            raise InvalidActionError(f"unknown anchor_id: {anchor_id}") from exc

    def _charge(self, nfe_cost: int) -> None:
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
                nfe_cost=nfe_cost,
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
