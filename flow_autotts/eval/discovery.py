"""Small discovery-loop utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_history(round_result: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(round_result, indent=2, sort_keys=True), encoding="utf-8")


def pareto_frontier(
    results: list[dict[str, Any]],
    reward_key: str = "final_reward",
    cost_key: str = "nfe",
) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []
    for candidate in results:
        reward = candidate.get(reward_key)
        cost = candidate.get(cost_key)
        if reward is None or cost is None:
            continue
        dominated = False
        for other in results:
            other_reward = other.get(reward_key)
            other_cost = other.get(cost_key)
            if other_reward is None or other_cost is None or other is candidate:
                continue
            if other_reward >= reward and other_cost <= cost and (
                other_reward > reward or other_cost < cost
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda item: (item[cost_key], -item[reward_key]))


def build_round_result(
    round_id: int,
    controller_name: str,
    beta_sweep_results: list[dict[str, Any]],
) -> dict[str, Any]:
    compact = [
        {
            "beta": result["beta"],
            "reward": result["final_reward"],
            "nfe": result["nfe"],
            "reward_per_nfe": result["reward_per_nfe"],
        }
        for result in beta_sweep_results
    ]
    return {
        "round_id": int(round_id),
        "controller_name": controller_name,
        "beta_sweep": compact,
        "pareto_frontier": pareto_frontier(compact, reward_key="reward", cost_key="nfe"),
        "raw_results": beta_sweep_results,
    }
