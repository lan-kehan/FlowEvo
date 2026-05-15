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
    compact = [_compact_beta_result(result) for result in beta_sweep_results]
    return {
        "round_id": int(round_id),
        "controller_name": controller_name,
        "beta_sweep": compact,
        "pareto_frontier": pareto_frontier(compact, reward_key="reward", cost_key="nfe"),
        "raw_results": beta_sweep_results,
    }


def _compact_beta_result(result: dict[str, Any]) -> dict[str, Any]:
    item: dict[str, Any] = {
        "beta": result["beta"],
        "reward": result["final_reward"],
        "nfe": result["nfe"],
        "reward_per_nfe": result["reward_per_nfe"],
    }
    action_statistics = _compact_action_statistics(result.get("action_statistics"))
    if action_statistics:
        item["action_statistics"] = action_statistics
        item["behavior_summary"] = summarize_action_statistics(action_statistics)
    return item


def _compact_action_statistics(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    compact: dict[str, float] = {}
    for key, raw in sorted(value.items()):
        try:
            numeric = float(raw)
        except (TypeError, ValueError):
            continue
        compact[str(key)] = round(numeric, 4)
    return compact


def summarize_action_statistics(action_statistics: dict[str, float]) -> str:
    """Return a compact behavior label for proposer context packs."""

    if not action_statistics:
        return "no action statistics"

    spawned = action_statistics.get("spawn", 0.0)
    preview = action_statistics.get("preview", 0.0)
    backward = action_statistics.get("backward", 0.0)
    prune = action_statistics.get("prune", 0.0)
    mean_nfe = action_statistics.get("mean_nfe", action_statistics.get("nfe", 0.0))

    if backward > 0.05 and preview > backward + 0.25:
        label = "preview-guided backward refinement"
    elif backward > 0.05:
        label = "backward refinement"
    elif spawned > 1.25 and preview > 0.05:
        label = "multi-root preview search"
    elif preview > 0.05:
        label = "single-root preview"
    else:
        label = "deterministic forward"

    extras: list[str] = [
        f"spawn={spawned:.2f}",
        f"preview={preview:.2f}",
    ]
    if backward > 0.0:
        extras.append(f"backward={backward:.2f}")
    if prune > 0.0:
        extras.append(f"prune={prune:.2f}")
    if mean_nfe > 0.0:
        extras.append(f"nfe={mean_nfe:.2f}")
    return f"{label} ({', '.join(extras)})"
