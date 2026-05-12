"""Controller evaluation runners."""

from __future__ import annotations

from dataclasses import asdict
from statistics import mean
from typing import Any, Callable, Sequence

from flow_autotts.controllers.base import Controller
from flow_autotts.core.env import FlowTTSEnv
from flow_autotts.eval.metrics import compute_metrics, event_log_to_dicts


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

    rewards = [ep["metrics"]["final_reward"] for ep in episodes if ep["metrics"]["final_reward"] is not None]
    nfes = [ep["metrics"]["nfe"] for ep in episodes]
    reward_per_nfes = [
        ep["metrics"]["reward_per_nfe"]
        for ep in episodes
        if ep["metrics"]["reward_per_nfe"] is not None
    ]
    return {
        "beta": float(beta),
        "num_seeds": len(seeds),
        "final_reward": mean(rewards) if rewards else None,
        "nfe": mean(nfes) if nfes else 0.0,
        "reward_per_nfe": mean(reward_per_nfes) if reward_per_nfes else None,
        "episodes": episodes,
        "action_statistics": _aggregate_actions(episodes),
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


def _aggregate_actions(episodes: list[dict[str, Any]]) -> dict[str, float]:
    if not episodes:
        return {}
    totals: dict[str, int] = {}
    total_nfe = 0
    for episode in episodes:
        metrics = episode["metrics"]
        total_nfe += int(metrics["nfe"])
        for action, count in metrics["action_counts"].items():
            totals[action] = totals.get(action, 0) + int(count)
    return {
        action.lower(): count / len(episodes)
        for action, count in sorted(totals.items())
    } | {"mean_nfe": total_nfe / len(episodes)}
