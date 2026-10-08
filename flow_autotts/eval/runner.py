"""Controller evaluation runners."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Sequence

from flow_autotts.controllers.base import Controller
from flow_autotts.core.env import FlowTTSEnv
from flow_autotts.eval.metrics import compute_metrics, event_log_to_dicts, summarize_episodes


EnvFactory = Callable[[int], FlowTTSEnv]


def evaluate_controller(
    controller: Controller,
    env_factory: EnvFactory,
    beta: float,
    seeds: Sequence[int],
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for seed in seeds:
        env = env_factory(int(seed))
        answer = controller.solve(env, beta)
        metrics = compute_metrics(answer)
        episodes.append(
            {
                "seed": int(seed),
                "answer": {
                    "particle_id": answer.particle_id,
                    "preview_id": answer.preview_id,
                    "latent": list(answer.latent),
                    "reward": answer.reward,
                    "nfe_used": answer.nfe_used,
                    "rule": answer.rule,
                    "score_dict": dict(answer.score_dict),
                },
                "metrics": asdict(metrics),
                "event_log": event_log_to_dicts(answer.event_log),
            }
        )

    return {
        "beta": float(beta),
        "num_seeds": len(seeds),
        **summarize_episodes(episodes),
        "episodes": episodes,
    }


def beta_sweep(
    controller: Controller,
    env_factory: EnvFactory,
    betas: Sequence[float],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    return [
        evaluate_controller(controller, env_factory, beta=float(beta), seeds=seeds)
        for beta in betas
    ]
