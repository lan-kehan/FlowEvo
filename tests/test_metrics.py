from __future__ import annotations

import unittest

from flow_autotts.controllers import DeterministicController
from flow_autotts.eval.discovery import pareto_frontier
from flow_autotts.eval.metrics import compute_metrics
from flow_autotts.experiments.eight_gaussians.harness import make_env


class MetricsTests(unittest.TestCase):
    def test_compute_metrics_from_answer(self):
        answer = DeterministicController().solve(make_env(seed=0, budget=32), beta=0.0)
        metrics = compute_metrics(answer)
        self.assertEqual(metrics.nfe, answer.nfe_used)
        self.assertEqual(metrics.action_counts["FORWARD"], 5)
        self.assertEqual(metrics.num_particles_spawned, 1)

    def test_pareto_frontier_removes_dominated_points(self):
        results = [
            {"name": "a", "reward": 1.0, "nfe": 10},
            {"name": "b", "reward": 1.0, "nfe": 12},
            {"name": "c", "reward": 1.2, "nfe": 20},
        ]
        frontier = pareto_frontier(results, reward_key="reward", cost_key="nfe")
        self.assertEqual([item["name"] for item in frontier], ["a", "c"])


if __name__ == "__main__":
    unittest.main()
