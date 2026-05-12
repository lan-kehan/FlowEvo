from __future__ import annotations

import math
import unittest

from flow_autotts.experiments.eight_gaussians.harness import run_harness


class EightGaussiansSmokeTests(unittest.TestCase):
    def test_harness_runs_controller_sweep(self):
        result = run_harness(
            controller_name="deterministic",
            betas=[0.0, 0.5],
            seeds=[0, 1],
            budget=32,
        )
        self.assertEqual(result["controller_name"], "DeterministicController")
        self.assertEqual(len(result["beta_sweep"]), 2)
        for item in result["beta_sweep"]:
            self.assertTrue(math.isfinite(item["reward"]))
            self.assertTrue(math.isfinite(item["nfe"]))


if __name__ == "__main__":
    unittest.main()
