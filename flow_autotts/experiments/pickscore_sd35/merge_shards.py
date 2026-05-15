"""Merge PickScore SD3.5 sharded harness histories."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from flow_autotts.eval.discovery import build_round_result
from flow_autotts.experiments.pickscore_sd35.harness import _aggregate_actions


def merge_histories(paths: Sequence[str | Path], output: str | Path) -> dict[str, Any]:
    histories = [_read_json(path) for path in paths]
    if not histories:
        raise ValueError("at least one shard history is required")
    _validate_compatible(histories)

    merged = copy.deepcopy(histories[0])
    merged["num_shards"] = len(histories)
    merged["shard_index"] = None
    merged["evaluated_sample_size"] = merged["sample_size"]
    merged["evaluated_prompt_sample"] = merged["prompt_sample"]
    merged["source_shards"] = [str(Path(path)) for path in paths]
    merged["rounds"] = []

    for round_pos in range(len(histories[0]["rounds"])):
        first_round = histories[0]["rounds"][round_pos]
        beta_results = []
        for beta_pos in range(len(first_round["raw_results"])):
            shard_results = [
                history["rounds"][round_pos]["raw_results"][beta_pos]
                for history in histories
            ]
            beta_results.append(_merge_beta_result(shard_results))

        round_result = build_round_result(
            round_id=first_round["round_id"],
            controller_name=first_round["controller_name"],
            beta_sweep_results=beta_results,
        )
        round_result["controller_key"] = first_round.get("controller_key")
        merged["rounds"].append(round_result)

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")
    return merged


def _merge_beta_result(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    beta = float(results[0]["beta"])
    episodes = [
        episode
        for result in results
        for episode in result.get("episodes", [])
    ]
    episodes.sort(key=lambda episode: int(episode["sample_rank"]))
    ranks = [int(episode["sample_rank"]) for episode in episodes]
    if len(ranks) != len(set(ranks)):
        raise ValueError("duplicate sample_rank found while merging shard episodes")

    rewards = [
        episode["metrics"]["final_reward"]
        for episode in episodes
        if episode["metrics"]["final_reward"] is not None
    ]
    nfes = [episode["metrics"]["nfe"] for episode in episodes]
    reward_per_nfes = [
        episode["metrics"]["reward_per_nfe"]
        for episode in episodes
        if episode["metrics"]["reward_per_nfe"] is not None
    ]
    return {
        "beta": beta,
        "num_samples": len(episodes),
        "final_reward": mean(rewards) if rewards else None,
        "nfe": mean(nfes) if nfes else 0.0,
        "reward_per_nfe": mean(reward_per_nfes) if reward_per_nfes else None,
        "episodes": episodes,
        "action_statistics": _aggregate_actions(episodes),
    }


def _validate_compatible(histories: Sequence[dict[str, Any]]) -> None:
    first = histories[0]
    fields = (
        "experiment",
        "dataset",
        "split",
        "sample_size",
        "sample_seed",
        "model_path",
        "pickscore_model_path",
        "betas",
        "budget",
        "env_config",
    )
    for history in histories[1:]:
        for field in fields:
            if history.get(field) != first.get(field):
                raise ValueError(f"incompatible shard field: {field}")
        if len(history.get("rounds", [])) != len(first.get("rounds", [])):
            raise ValueError("all shards must have the same number of rounds")


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    merged = merge_histories(args.inputs, args.output)
    print(
        json.dumps(
            {
                "output": args.output,
                "rounds": [
                    {
                        "round_id": round_result["round_id"],
                        "controller_key": round_result.get("controller_key"),
                        "beta_sweep": round_result["beta_sweep"],
                    }
                    for round_result in merged["rounds"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
