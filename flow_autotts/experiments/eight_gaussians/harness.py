"""Run controller sweeps on the 8 Gaussians toy flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from flow_autotts.controllers import (
    BestOfNController,
    DeterministicController,
    OptimalController,
    PrismStyleFlowController,
    SDEForwardController,
    SelfRefineController,
)
from flow_autotts.core.env import FlowTTSEnv
from flow_autotts.eval.discovery import build_round_result, write_history
from flow_autotts.eval.runner import beta_sweep
from flow_autotts.experiments.eight_gaussians.model import (
    EightGaussiansFlowModel,
    EightGaussiansScorer,
)


CONTROLLERS = {
    "deterministic": DeterministicController,
    "best_of_n": BestOfNController,
    "self_refine": SelfRefineController,
    "prism": PrismStyleFlowController,
    "sde_forward": SDEForwardController,
    "optimal": OptimalController,
}


def make_env(
    seed: int,
    budget: int = 128,
    time_grid: Sequence[float] | None = None,
) -> FlowTTSEnv:
    return FlowTTSEnv(
        model=EightGaussiansFlowModel(),
        vae=None,
        scorer=EightGaussiansScorer(),
        prompt="eight_gaussians",
        budget=budget,
        time_grid=time_grid or [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        seed=seed,
        latent_shape=(2,),
    )


def run_harness(
    controller_name: str,
    betas: Sequence[float],
    seeds: Sequence[int],
    budget: int = 128,
    output: str | None = None,
) -> dict:
    controller_cls = CONTROLLERS[controller_name]
    controller = controller_cls()
    results = beta_sweep(
        controller=controller,
        env_factory=lambda seed: make_env(seed=seed, budget=budget),
        betas=betas,
        seeds=seeds,
    )
    round_result = build_round_result(
        round_id=0,
        controller_name=controller_cls.__name__,
        beta_sweep_results=results,
    )
    if output is not None:
        write_history(round_result, Path(output))
    return round_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=sorted(CONTROLLERS), default="prism")
    parser.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = run_harness(
        controller_name=args.controller,
        betas=args.betas,
        seeds=args.seeds,
        budget=args.budget,
        output=args.output,
    )
    compact = {
        "controller_name": result["controller_name"],
        "beta_sweep": result["beta_sweep"],
        "pareto_frontier": result["pareto_frontier"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
