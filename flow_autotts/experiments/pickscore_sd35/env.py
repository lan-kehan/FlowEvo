"""Online SD3.5 environment scored by PickScore.

The public surface mirrors :class:`flow_autotts.core.env.FlowTTSEnv`, while
the private state stores real SD3 latent tensors instead of small toy vectors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flow_autotts.core.errors import BudgetExceededError, InvalidActionError
from flow_autotts.core.state import (
    AnswerRecord,
    ControllerState,
    EventRecord,
    ParticleSummary,
    PreviewRecord,
)
from flow_autotts.experiments.pickscore_sd35.scoring import PickScoreBatchScorer


@dataclass(frozen=True)
class SD35EnvConfig:
    """Runtime knobs for the SD3.5 online controller environment."""

    resolution: int = 512
    num_steps: int = 10
    guidance_scale: float = 4.5
    noise_level: float = 0.7
    sde_type: str = "sde"
    max_sequence_length: int = 128


@dataclass
class SD35Resources:
    """Shared heavyweight model resources reused across prompt episodes."""

    pipeline: Any
    scorer: PickScoreBatchScorer
    torch: Any
    device: str
    dtype: Any
    timesteps: Any
    sigmas: Any
    text_encoder_device: str | None = None
    offload_text_encoders_after_encode: bool = False
    prompt_cache: dict[tuple[str, float, int], tuple[Any, Any]] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        model_path: str | Path = "SD_3.5_med",
        pickscore_model_path: str | Path = "PickScore_v1",
        pickscore_processor_path: str | Path | None = None,
        device: str = "cuda",
        text_encoder_device: str | None = None,
        offload_text_encoders_after_encode: bool = False,
        score_device: str | None = None,
        dtype: str = "bfloat16",
        score_dtype: str = "float32",
        num_steps: int = 10,
        local_files_only: bool = False,
        progress: bool = False,
    ) -> "SD35Resources":
        import torch
        from diffusers import StableDiffusion3Pipeline

        torch_dtype = _torch_dtype(torch, dtype)
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )
        pipeline = pipeline.to(device)
        pipeline.set_progress_bar_config(disable=not progress, leave=False)
        pipeline.scheduler.set_timesteps(int(num_steps), device=device)
        encoder_device = text_encoder_device or device

        # Keep VAE decode stable; transformer/text encoders stay in inference dtype.
        pipeline.vae.to(device=device, dtype=torch.float32)
        if encoder_device != device:
            _move_text_encoders(pipeline, encoder_device)
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        elif offload_text_encoders_after_encode:
            pipeline.transformer.to("cpu")
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        if hasattr(pipeline, "safety_checker"):
            pipeline.safety_checker = None

        scorer = PickScoreBatchScorer(
            model_path=pickscore_model_path,
            processor_path=pickscore_processor_path or pickscore_model_path,
            device=score_device or device,
            dtype=score_dtype,
            local_files_only=local_files_only,
        )
        return cls(
            pipeline=pipeline,
            scorer=scorer,
            torch=torch,
            device=device,
            text_encoder_device=encoder_device,
            offload_text_encoders_after_encode=bool(offload_text_encoders_after_encode),
            dtype=torch_dtype,
            timesteps=pipeline.scheduler.timesteps.detach().clone(),
            sigmas=pipeline.scheduler.sigmas.detach().clone(),
        )

    def prompt_embeddings(
        self,
        prompt: str,
        guidance_scale: float,
        max_sequence_length: int,
    ) -> tuple[Any, Any]:
        key = (prompt, float(guidance_scale), int(max_sequence_length))
        cached = self.prompt_cache.get(key)
        if cached is not None:
            return cached

        do_cfg = float(guidance_scale) > 1.0
        pipe = self.pipeline
        pipe._guidance_scale = float(guidance_scale)
        encode_device = self.text_encoder_device or self.device
        if self.offload_text_encoders_after_encode:
            pipe.transformer.to("cpu")
            _move_text_encoders(pipe, encode_device)
            if str(self.device).startswith("cuda"):
                self.torch.cuda.empty_cache()
        encoded = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=None,
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=do_cfg,
            device=encode_device,
            clip_skip=None,
            num_images_per_prompt=1,
            max_sequence_length=int(max_sequence_length),
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = encoded
        if do_cfg:
            prompt_embeds = self.torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = self.torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds],
                dim=0,
            )
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=self.device, dtype=self.dtype)
        if self.offload_text_encoders_after_encode:
            _move_text_encoders(pipe, "cpu")
            pipe.transformer.to(self.device)
            if str(self.device).startswith("cuda"):
                self.torch.cuda.empty_cache()
        cached = (prompt_embeds, pooled_prompt_embeds)
        self.prompt_cache[key] = cached
        return cached


@dataclass
class _SD35Particle:
    id: int
    latents: Any
    step_index: int
    parent_id: int | None
    source_anchor_id: int | None
    nfe_used: int
    status: str
    generator: Any
    last_preview_id: int | None = None
    num_children: int = 0
    sde_variance: float = 0.0


@dataclass
class _SD35Anchor:
    id: int
    particle_id: int
    step_index: int
    clean_latents: Any
    noise_latents: Any
    score: float
    score_dict: dict[str, float]
    uncertainty: float
    drift: float


class SD35PickScoreEnv:
    """Controller environment backed by SD3.5 Medium and PickScore."""

    def __init__(
        self,
        resources: SD35Resources,
        prompt: str,
        seed: int,
        budget: int,
        config: SD35EnvConfig | None = None,
        time_grid: list[float] | tuple[float, ...] | None = None,
    ) -> None:
        self.resources = resources
        self.prompt = str(prompt)
        self.seed = int(seed)
        self.budget = int(budget)
        self.config = config or SD35EnvConfig()
        if self.budget < 0:
            raise ValueError("budget must be non-negative")
        if self.config.num_steps <= 0:
            raise ValueError("num_steps must be positive")

        self.time_grid = tuple(
            float(t)
            for t in (
                time_grid
                if time_grid is not None
                else [i / self.config.num_steps for i in range(self.config.num_steps + 1)]
            )
        )
        self.latent_shape = self._latent_shape()

        self._particles: dict[int, _SD35Particle] = {}
        self._anchors: dict[int, _SD35Anchor] = {}
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
                id=particle.id,
                time=self._step_to_time(particle.step_index),
                parent_id=particle.parent_id,
                source_anchor_id=particle.source_anchor_id,
                nfe_used=particle.nfe_used,
                status=particle.status,
                last_preview_id=particle.last_preview_id,
                num_children=particle.num_children,
            )
            for pid, particle in self._particles.items()
        }
        previews = {
            aid: PreviewRecord(
                id=anchor.id,
                particle_id=anchor.particle_id,
                time=self._step_to_time(anchor.step_index),
                score=anchor.score,
                score_dict=dict(anchor.score_dict),
                uncertainty=anchor.uncertainty,
                drift=anchor.drift,
                embedding_ref=f"sd35_anchor:{anchor.id}",
            )
            for aid, anchor in self._anchors.items()
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

        ids: list[int] = []
        for _ in range(int(n)):
            particle_id = self._next_particle_id
            self._next_particle_id += 1
            generator = self._make_generator(self.seed + particle_id * 104_729)
            latents = self._initial_latents(generator)
            self._particles[particle_id] = _SD35Particle(
                id=particle_id,
                latents=latents,
                step_index=0,
                parent_id=None,
                source_anchor_id=None,
                nfe_used=0,
                status="active",
                generator=generator,
            )
            ids.append(particle_id)

        self._log(
            action="SPAWN",
            particle_ids=ids,
            input_time=None,
            output_time=0.0,
            nfe_cost=0,
            details={"n": int(n)},
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
        if not 0.0 <= float(target_time) <= 1.0:
            raise InvalidActionError("FORWARD target_time must be in [0, 1]")
        if solver not in {"euler", "sde"}:
            raise InvalidActionError(f"unsupported solver: {solver}")
        particle = self._require_particle(particle_id, status="active")
        target_step = self._time_to_step(target_time)
        if target_step <= particle.step_index:
            raise InvalidActionError(
                f"FORWARD target_time must advance beyond {self._step_to_time(particle.step_index)}"
            )
        if target_step > self.config.num_steps:
            raise InvalidActionError("FORWARD target_time must be <= 1.0")

        cost = target_step - particle.step_index
        self._charge(cost)
        input_time = self._step_to_time(particle.step_index)
        sde_cfg = self._parse_sde_cfg(cfg)

        for step_index in range(particle.step_index, target_step):
            velocity = self._predict_velocity(particle.latents, step_index)
            if solver == "euler":
                particle.latents = self._euler_step(particle.latents, velocity, step_index)
            else:
                particle.latents, variance = self._sde_step(
                    particle.latents,
                    velocity,
                    step_index,
                    particle.generator,
                    sde_cfg,
                )
                particle.sde_variance += variance

        particle.step_index = target_step
        particle.nfe_used += cost
        if particle.step_index >= self.config.num_steps:
            particle.status = "completed"

        output_time = self._step_to_time(particle.step_index)
        self._log(
            action="FORWARD",
            particle_ids=[particle_id],
            input_time=input_time,
            output_time=output_time,
            nfe_cost=cost,
            details={"solver": solver, "cfg": cfg, "target_step": target_step},
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
        clean_latents, noise_latents = self._clean_and_noise_estimate(particle)
        score, score_dict = self._score_latents(clean_latents) if scorer is not None else (0.0, {})
        uncertainty = self._preview_uncertainty(particle.step_index, particle.sde_variance)
        drift = self._latent_rmse(clean_latents, particle.latents)

        anchor_id = self._next_anchor_id
        self._next_anchor_id += 1
        self._anchors[anchor_id] = _SD35Anchor(
            id=anchor_id,
            particle_id=particle_id,
            step_index=particle.step_index,
            clean_latents=clean_latents.detach(),
            noise_latents=noise_latents.detach(),
            score=float(score),
            score_dict=score_dict,
            uncertainty=uncertainty,
            drift=drift,
        )
        particle.last_preview_id = anchor_id
        particle.nfe_used += 1

        self._log(
            action="PREVIEW",
            particle_ids=[particle_id],
            input_time=self._step_to_time(particle.step_index),
            output_time=self._step_to_time(particle.step_index),
            nfe_cost=1,
            details={
                "anchor_id": anchor_id,
                "mode": mode,
                "score": float(score),
                "uncertainty": uncertainty,
                "step_index": particle.step_index,
            },
        )
        return self.get_state().previews[anchor_id]

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
        if mask is not None:
            raise InvalidActionError("SD35PickScoreEnv does not support masks yet")
        if not 0.0 <= float(target_time) <= 1.0:
            raise InvalidActionError("BACKWARD target_time must be in [0, 1]")
        if num_children <= 0:
            raise InvalidActionError("BACKWARD requires num_children > 0")
        anchor = self._require_anchor(anchor_id)
        source = self._require_particle(anchor.particle_id)
        target_step = self._time_to_step_floor(target_time)
        sigma = self._sigma(target_step)

        child_ids: list[int] = []
        for _ in range(int(num_children)):
            child_id = self._next_particle_id
            self._next_particle_id += 1
            generator = self._make_generator(self.seed + child_id * 154_858_63)
            noise = self._noise_for_policy(anchor, noise_policy, strength, generator)
            latents = (1.0 - sigma) * anchor.clean_latents + sigma * noise
            self._particles[child_id] = _SD35Particle(
                id=child_id,
                latents=latents.detach(),
                step_index=target_step,
                parent_id=anchor.particle_id,
                source_anchor_id=anchor_id,
                nfe_used=0,
                status="completed" if target_step >= self.config.num_steps else "active",
                generator=generator,
            )
            child_ids.append(child_id)

        source.num_children += int(num_children)
        self._log(
            action="BACKWARD",
            particle_ids=child_ids,
            input_time=self._step_to_time(anchor.step_index),
            output_time=self._step_to_time(target_step),
            nfe_cost=0,
            details={
                "anchor_id": anchor_id,
                "noise_policy": noise_policy,
                "num_children": int(num_children),
                "strength": float(strength),
                "target_step": target_step,
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
            particle_id, preview_id, reward, score_dict = self._answer_best_preview()
        elif rule == "latest_active":
            particle_id, preview_id, reward, score_dict = self._answer_latest_active()
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
            latent=(),
            reward=reward,
            nfe_used=self._nfe_used,
            rule=rule,
            event_log=list(self._events),
            score_dict=score_dict,
        )

    def _answer_best_preview(self) -> tuple[int | None, int | None, float | None, dict[str, float]]:
        if not self._anchors:
            return self._answer_latest_active()
        chosen = max(self._anchors.values(), key=lambda anchor: anchor.score)
        return chosen.particle_id, chosen.id, chosen.score, dict(chosen.score_dict)

    def _answer_latest_active(self) -> tuple[int | None, int | None, float | None, dict[str, float]]:
        candidates = [
            particle for particle in self._particles.values() if particle.status in {"active", "completed"}
        ]
        if not candidates:
            return None, None, None, {}
        chosen = max(candidates, key=lambda particle: (particle.step_index, particle.id))
        if chosen.status == "active":
            self.forward(chosen.id, target_time=1.0, solver="euler")
            chosen = self._particles[chosen.id]
        reward, score_dict = self._score_latents(chosen.latents)
        return chosen.id, chosen.last_preview_id, reward, score_dict

    def _latent_shape(self) -> tuple[int, ...]:
        pipe = self.resources.pipeline
        channels = int(pipe.transformer.config.in_channels)
        latent_h = int(self.config.resolution) // int(pipe.vae_scale_factor)
        latent_w = int(self.config.resolution) // int(pipe.vae_scale_factor)
        return (1, channels, latent_h, latent_w)

    def _initial_latents(self, generator: Any) -> Any:
        pipe = self.resources.pipeline
        latents = pipe.prepare_latents(
            1,
            int(pipe.transformer.config.in_channels),
            int(self.config.resolution),
            int(self.config.resolution),
            self.resources.dtype,
            self.resources.device,
            generator,
            None,
        )
        return latents.float().detach()

    def _predict_velocity(self, latents: Any, step_index: int) -> Any:
        torch = self.resources.torch
        prompt_embeds, pooled_prompt_embeds = self.resources.prompt_embeddings(
            self.prompt,
            guidance_scale=self.config.guidance_scale,
            max_sequence_length=self.config.max_sequence_length,
        )
        do_cfg = self.config.guidance_scale > 1.0
        latent_input = torch.cat([latents] * 2) if do_cfg else latents
        latent_input = latent_input.to(dtype=prompt_embeds.dtype)
        timestep = self.resources.timesteps[step_index].expand(latent_input.shape[0])
        with torch.no_grad():
            noise_pred = self.resources.pipeline.transformer(
                hidden_states=latent_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
        if do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.config.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        return noise_pred.float()

    def _euler_step(self, latents: Any, velocity: Any, step_index: int) -> Any:
        sigma = self._sigma(step_index)
        sigma_next = self._sigma(step_index + 1)
        return (latents.float() + (sigma_next - sigma) * velocity.float()).detach()

    def _sde_step(
        self,
        latents: Any,
        velocity: Any,
        step_index: int,
        generator: Any,
        cfg: dict[str, float | str],
    ) -> tuple[Any, float]:
        torch = self.resources.torch
        sigma = self._sigma(step_index)
        sigma_next = self._sigma(step_index + 1)
        dt = sigma_next - sigma
        noise_level = float(cfg["noise_level"])
        sde_type = str(cfg["sde_type"])
        sample = latents.float()
        model_output = velocity.float()
        variance_noise = self._randn(sample.shape, sample.dtype, generator)

        if sde_type == "cps":
            std_dev_t = sigma_next * math.sin(noise_level * math.pi / 2)
            pred_original_sample = sample - sigma * model_output
            noise_estimate = sample + model_output * (1.0 - sigma)
            residual = torch.clamp(sigma_next**2 - std_dev_t**2, min=0.0)
            prev_mean = (
                pred_original_sample * (1.0 - sigma_next)
                + noise_estimate * torch.sqrt(residual)
            )
            next_latents = prev_mean + std_dev_t * variance_noise
            return next_latents.detach(), float((std_dev_t**2).item())

        if sde_type != "sde":
            raise InvalidActionError(f"unsupported sde_type: {sde_type}")

        sigma_max = self._sigma(1 if len(self.resources.sigmas) > 1 else 0)
        denom = 1.0 - (sigma_max if abs(float(sigma.item()) - 1.0) < 1e-6 else sigma)
        denom = torch.clamp(denom, min=1e-6)
        std_dev_t = torch.sqrt(sigma / denom) * noise_level
        if noise_level <= 0:
            next_latents = sample + dt * model_output
            return next_latents.detach(), 0.0

        prev_mean = sample * (1.0 + std_dev_t**2 / (2 * sigma) * dt) + model_output * (
            1.0 + std_dev_t**2 * (1.0 - sigma) / (2 * sigma)
        ) * dt
        step_std = std_dev_t * torch.sqrt(torch.clamp(-dt, min=0.0))
        next_latents = prev_mean + step_std * variance_noise
        return next_latents.detach(), float((step_std**2).mean().item())

    def _clean_and_noise_estimate(self, particle: _SD35Particle) -> tuple[Any, Any]:
        torch = self.resources.torch
        if particle.step_index >= self.config.num_steps:
            return particle.latents.detach(), torch.zeros_like(particle.latents)
        velocity = self._predict_velocity(particle.latents, particle.step_index)
        sigma = self._sigma(particle.step_index)
        clean = particle.latents.float() - sigma * velocity
        noise = particle.latents.float() + (1.0 - sigma) * velocity
        return clean.detach(), noise.detach()

    def _score_latents(self, latents: Any) -> tuple[float, dict[str, float]]:
        image = self._decode_latents(latents)
        score = float(self.resources.scorer([self.prompt], [image])[0])
        return score, {"pickscore": score}

    def _decode_latents(self, latents: Any) -> Any:
        torch = self.resources.torch
        pipe = self.resources.pipeline
        with torch.no_grad():
            vae_latents = (latents.float() / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
            vae_latents = vae_latents.to(device=self.resources.device, dtype=pipe.vae.dtype)
            image = pipe.vae.decode(vae_latents, return_dict=False)[0]
            return pipe.image_processor.postprocess(image, output_type="pil")[0]

    def _noise_for_policy(
        self,
        anchor: _SD35Anchor,
        noise_policy: str,
        strength: float,
        generator: Any,
    ) -> Any:
        if noise_policy == "inferred_noise":
            return anchor.noise_latents
        fresh = self._randn(anchor.clean_latents.shape, anchor.clean_latents.dtype, generator)
        if noise_policy == "fresh_noise":
            return fresh
        if noise_policy == "mixed_noise":
            lam = min(max(float(strength), 0.0), 1.0)
            return lam * anchor.noise_latents + (1.0 - lam) * fresh
        raise InvalidActionError(f"unsupported noise_policy: {noise_policy}")

    def _parse_sde_cfg(self, cfg: Any | None) -> dict[str, float | str]:
        if cfg is None:
            return {"noise_level": self.config.noise_level, "sde_type": self.config.sde_type}
        if isinstance(cfg, (int, float)):
            return {"noise_level": float(cfg), "sde_type": self.config.sde_type}
        if isinstance(cfg, dict):
            return {
                "noise_level": float(cfg.get("noise_level", cfg.get("noise_scale", self.config.noise_level))),
                "sde_type": str(cfg.get("sde_type", self.config.sde_type)),
            }
        raise InvalidActionError("SDE cfg must be None, a number, or a dict")

    def _sigma(self, step_index: int) -> Any:
        clamped = min(max(int(step_index), 0), self.config.num_steps)
        return self.resources.sigmas[clamped].to(device=self.resources.device, dtype=self.resources.torch.float32)

    def _time_to_step(self, target_time: float) -> int:
        time = min(max(float(target_time), 0.0), 1.0)
        return min(self.config.num_steps, max(0, math.ceil(time * self.config.num_steps - 1e-9)))

    def _time_to_step_floor(self, target_time: float) -> int:
        time = min(max(float(target_time), 0.0), 1.0)
        return min(self.config.num_steps, max(0, math.floor(time * self.config.num_steps + 1e-9)))

    def _step_to_time(self, step_index: int) -> float:
        return float(step_index) / float(self.config.num_steps)

    def _preview_uncertainty(self, step_index: int, sde_variance: float) -> float:
        base = 1.0 - self._step_to_time(step_index)
        stochastic = math.sqrt(max(0.0, float(sde_variance)))
        return max(0.0, min(1.0, base + 0.25 * stochastic))

    def _latent_rmse(self, a: Any, b: Any) -> float:
        return float((a.float() - b.float()).pow(2).mean().sqrt().detach().cpu().item())

    def _make_generator(self, seed: int) -> Any:
        torch = self.resources.torch
        try:
            generator = torch.Generator(device=self.resources.device)
        except RuntimeError:
            generator = torch.Generator()
        return generator.manual_seed(int(seed) % (2**31 - 1))

    def _randn(self, shape: Any, dtype: Any, generator: Any) -> Any:
        torch = self.resources.torch
        return torch.randn(
            shape,
            generator=generator,
            device=self.resources.device,
            dtype=dtype,
        )

    def _require_particle(self, particle_id: int, status: str | None = None) -> _SD35Particle:
        try:
            particle = self._particles[int(particle_id)]
        except KeyError as exc:
            raise InvalidActionError(f"unknown particle_id: {particle_id}") from exc
        if status is not None and particle.status != status:
            raise InvalidActionError(
                f"particle {particle_id} has status {particle.status}, expected {status}"
            )
        return particle

    def _require_anchor(self, anchor_id: int) -> _SD35Anchor:
        try:
            return self._anchors[int(anchor_id)]
        except KeyError as exc:
            raise InvalidActionError(f"unknown anchor_id: {anchor_id}") from exc

    def _charge(self, nfe_cost: int) -> None:
        if self._nfe_used + int(nfe_cost) > self.budget:
            raise BudgetExceededError(
                f"action would exceed budget: used {self._nfe_used}, "
                f"cost {nfe_cost}, budget {self.budget}"
            )
        self._nfe_used += int(nfe_cost)

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
        for particle_id, particle in self._particles.items():
            if particle.status == "active":
                active.append(particle_id)
            elif particle.status == "completed":
                completed.append(particle_id)
            elif particle.status == "pruned":
                pruned.append(particle_id)
        return active, completed, pruned

    def _ensure_open(self) -> None:
        if self._answered:
            raise InvalidActionError("episode already answered")


def _torch_dtype(torch: object, dtype: str):
    normalized = str(dtype).lower()
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32", "no"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {dtype}")


def _move_text_encoders(pipeline: Any, device: str) -> None:
    for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        encoder = getattr(pipeline, name, None)
        if encoder is not None:
            encoder.to(device)
