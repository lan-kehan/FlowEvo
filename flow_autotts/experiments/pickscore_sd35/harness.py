"""Run controller rounds on random PickScore train prompts with SD3.5."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from flow_autotts.controllers import (
    BestOfNController,
    DeterministicController,
    OptimalController,
    PrismStyleFlowController,
    SDEForwardController,
    SelfRefineController,
)
from flow_autotts.controllers.base import Controller
from flow_autotts.eval.discovery import build_round_result
from flow_autotts.eval.metrics import compute_metrics, event_log_to_dicts
from flow_autotts.experiments.pickscore_sd35.dataset import PromptSample, sample_prompt_file
from flow_autotts.experiments.pickscore_sd35.env import (
    SD35EnvConfig,
    SD35PickScoreEnv,
    SD35Resources,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


class SD35DefaultDeterministicEulerController(DeterministicController):
    """SD3.5 default deterministic Euler baseline used for round 0 history."""


CONTROLLERS: dict[str, type] = {
    "sd35_default": SD35DefaultDeterministicEulerController,
    "deterministic": DeterministicController,
    "best_of_n": BestOfNController,
    "self_refine": SelfRefineController,
    "sde_forward": SDEForwardController,
    "prism": PrismStyleFlowController,
    "optimal": OptimalController,
}

DEFAULT_ROUND_CONTROLLERS = (
    "optimal",
    "optimal",
    "optimal",
    "optimal",
    "optimal",
)


def run_harness(
    dataset_dir: str | Path | None = None,
    split: str = "train",
    sample_size: int = 100,
    sample_seed: int = 42,
    num_shards: int = 1,
    shard_index: int = 0,
    rounds: int = 5,
    controller_names: Sequence[str] | None = None,
    betas: Sequence[float] = (0.5,),
    budget: int = 64,
    output: str | Path | None = None,
    compact: bool = False,
    model_path: str | Path | None = None,
    pickscore_model_path: str | Path | None = None,
    pickscore_processor_path: str | Path | None = None,
    device: str | None = None,
    text_encoder_device: str | None = None,
    offload_text_encoders_after_encode: bool = False,
    score_device: str | None = None,
    dtype: str | None = None,
    score_dtype: str = "float32",
    resolution: int = 512,
    num_steps: int = 10,
    guidance_scale: float = 4.5,
    noise_level: float = 0.7,
    sde_type: str = "sde",
    local_files_only: bool = True,
    progress: bool = False,
) -> dict[str, Any]:
    dataset = Path(dataset_dir) if dataset_dir is not None else _default_dataset_dir()
    model = Path(model_path) if model_path is not None else _default_model_path()
    pickscore_model = (
        Path(pickscore_model_path)
        if pickscore_model_path is not None
        else _default_pickscore_path()
    )
    pickscore_processor = (
        Path(pickscore_processor_path)
        if pickscore_processor_path is not None
        else pickscore_model
    )
    runtime_device = device or _default_device()
    runtime_dtype = dtype or ("bfloat16" if runtime_device.startswith("cuda") else "float32")

    all_samples = sample_prompt_file(
        dataset_dir=dataset,
        split=split,
        sample_size=sample_size,
        seed=sample_seed,
    )
    sample_ranks = list(range(len(all_samples)))
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    if num_shards > 1:
        ranked = [
            (rank, sample)
            for rank, sample in enumerate(all_samples)
            if rank % num_shards == shard_index
        ]
        sample_ranks = [rank for rank, _sample in ranked]
        samples = [sample for _rank, sample in ranked]
    else:
        samples = all_samples
    env_config = SD35EnvConfig(
        resolution=resolution,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        noise_level=noise_level,
        sde_type=sde_type,
    )
    resources = SD35Resources.load(
        model_path=model,
        pickscore_model_path=pickscore_model,
        pickscore_processor_path=pickscore_processor,
        device=runtime_device,
        text_encoder_device=text_encoder_device,
        offload_text_encoders_after_encode=offload_text_encoders_after_encode,
        score_device=score_device,
        dtype=runtime_dtype,
        score_dtype=score_dtype,
        num_steps=num_steps,
        local_files_only=local_files_only,
        progress=progress,
    )

    selected_controllers = _round_controller_names(rounds, controller_names)
    history: dict[str, Any] = {
        "experiment": "pickscore_sd35",
        "dataset": str(dataset),
        "split": split,
        "sample_size": sample_size,
        "evaluated_sample_size": len(samples),
        "sample_seed": sample_seed,
        "num_shards": int(num_shards),
        "shard_index": int(shard_index),
        "prompt_sample": [sample.to_dict() for sample in all_samples],
        "evaluated_prompt_sample": [sample.to_dict() for sample in samples],
        "model_path": str(model),
        "pickscore_model_path": str(pickscore_model),
        "device": runtime_device,
        "text_encoder_device": text_encoder_device or runtime_device,
        "offload_text_encoders_after_encode": bool(offload_text_encoders_after_encode),
        "score_device": score_device or runtime_device,
        "dtype": runtime_dtype,
        "betas": [float(beta) for beta in betas],
        "budget": int(budget),
        "env_config": asdict(env_config),
        "rounds": [],
    }

    for round_id, controller_name in enumerate(selected_controllers):
        controller_cls = CONTROLLERS[controller_name]
        controller = controller_cls()
        beta_results = [
            evaluate_controller_on_samples(
                controller=controller,
                resources=resources,
                samples=samples,
                sample_ranks=sample_ranks,
                beta=float(beta),
                budget=budget,
                env_config=env_config,
            )
            for beta in betas
        ]
        if compact:
            for result in beta_results:
                result.pop("episodes", None)
        round_result = build_round_result(
            round_id=round_id,
            controller_name=controller_cls.__name__,
            beta_sweep_results=beta_results,
        )
        round_result["controller_key"] = controller_name
        history["rounds"].append(round_result)

        if output is not None:
            _write_json(history, output)

    if output is not None:
        _write_json(history, output)
    return history


def evaluate_controller_on_samples(
    controller: Controller,
    resources: SD35Resources,
    samples: Sequence[PromptSample],
    beta: float,
    budget: int,
    env_config: SD35EnvConfig,
    sample_ranks: Sequence[int] | None = None,
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    if sample_ranks is not None and len(sample_ranks) != len(samples):
        raise ValueError("sample_ranks must have the same length as samples")
    for local_rank, sample in enumerate(samples):
        sample_rank = int(sample_ranks[local_rank]) if sample_ranks is not None else local_rank
        env = SD35PickScoreEnv(
            resources=resources,
            prompt=sample.prompt,
            seed=sample.seed,
            budget=budget,
            config=env_config,
        )
        answer = controller.solve(env, beta)
        metrics = compute_metrics(answer)
        episodes.append(
            {
                "sample_rank": sample_rank,
                "prompt_index": sample.index,
                "prompt": sample.prompt,
                "seed": sample.seed,
                "answer": {
                    "particle_id": answer.particle_id,
                    "preview_id": answer.preview_id,
                    "reward": answer.reward,
                    "nfe_used": answer.nfe_used,
                    "rule": answer.rule,
                    "score_dict": dict(answer.score_dict),
                },
                "metrics": asdict(metrics),
                "event_log": event_log_to_dicts(answer.event_log),
            }
        )
        resources.prompt_cache.clear()
        if str(resources.device).startswith("cuda") and hasattr(resources.torch, "cuda"):
            resources.torch.cuda.empty_cache()

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
        "beta": float(beta),
        "num_samples": len(samples),
        "final_reward": mean(rewards) if rewards else None,
        "nfe": mean(nfes) if nfes else 0.0,
        "reward_per_nfe": mean(reward_per_nfes) if reward_per_nfes else None,
        "episodes": episodes,
        "action_statistics": _aggregate_actions(episodes),
    }


def compact_summary(history: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": history["experiment"],
        "sample_size": history["sample_size"],
        "evaluated_sample_size": history.get("evaluated_sample_size", history["sample_size"]),
        "sample_seed": history["sample_seed"],
        "num_shards": history.get("num_shards", 1),
        "shard_index": history.get("shard_index", 0),
        "betas": history["betas"],
        "budget": history["budget"],
        "rounds": [
            {
                "round_id": item["round_id"],
                "controller": item["controller_key"],
                "controller_name": item["controller_name"],
                "beta_sweep": item["beta_sweep"],
                "pareto_frontier": item["pareto_frontier"],
            }
            for item in history["rounds"]
        ],
    }


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


def _round_controller_names(rounds: int, controller_names: Sequence[str] | None) -> list[str]:
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    names = list(controller_names) if controller_names else list(DEFAULT_ROUND_CONTROLLERS)
    unknown = [name for name in names if name not in CONTROLLERS]
    if unknown:
        raise ValueError(f"unknown controller(s): {', '.join(unknown)}")
    if len(names) >= rounds:
        return names[:rounds]
    return names + [names[-1]] * (rounds - len(names))


def _default_dataset_dir() -> Path:
    return REPO_ROOT / "flow_grpo" / "dataset" / "pickscore"


def _default_model_path() -> Path | str:
    local_path = REPO_ROOT / "SD_3.5_med"
    return local_path if local_path.exists() else "stabilityai/stable-diffusion-3.5-medium"


def _default_pickscore_path() -> Path | str:
    local_path = REPO_ROOT / "PickScore_v1"
    return local_path if local_path.exists() else "yuvalkirstain/PickScore_v1"


def _default_output_path() -> Path:
    return REPO_ROOT / "logs" / "flow_autotts" / "pickscore_sd35" / "history.json"


def _default_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _write_json(payload: dict[str, Any], output: str | Path) -> None:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(_default_dataset_dir()))
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--controllers", nargs="+", choices=sorted(CONTROLLERS), default=None)
    parser.add_argument("--betas", type=float, nargs="+", default=[0.5])
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--output", default=str(_default_output_path()))
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--model", default=str(_default_model_path()))
    parser.add_argument("--pickscore-model", default=str(_default_pickscore_path()))
    parser.add_argument("--pickscore-processor", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--text-encoder-device", default=None)
    parser.add_argument("--offload-text-encoders-after-encode", action="store_true")
    parser.add_argument("--score-device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--score-dtype", default="float32")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--noise-level", type=float, default=0.7)
    parser.add_argument("--sde-type", choices=["sde", "cps"], default="sde")
    parser.add_argument("--allow-remote-files", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    history = run_harness(
        dataset_dir=args.dataset,
        split=args.split,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        rounds=args.rounds,
        controller_names=args.controllers,
        betas=args.betas,
        budget=args.budget,
        output=args.output,
        compact=args.compact,
        model_path=args.model,
        pickscore_model_path=args.pickscore_model,
        pickscore_processor_path=args.pickscore_processor,
        device=args.device,
        text_encoder_device=args.text_encoder_device,
        offload_text_encoders_after_encode=args.offload_text_encoders_after_encode,
        score_device=args.score_device,
        dtype=args.dtype,
        score_dtype=args.score_dtype,
        resolution=args.resolution,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        noise_level=args.noise_level,
        sde_type=args.sde_type,
        local_files_only=not args.allow_remote_files,
        progress=args.progress,
    )
    summary = compact_summary(history)
    if args.summary_output is not None:
        _write_json(summary, args.summary_output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
