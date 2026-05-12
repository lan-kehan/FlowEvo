from __future__ import annotations

import unittest

from flow_autotts.controllers import (
    BestOfNController,
    DeterministicController,
    PrismStyleFlowController,
    SelfRefineController,
)
from flow_autotts.experiments.eight_gaussians.harness import make_env


class BaselineControllerTests(unittest.TestCase):
    def test_deterministic_reaches_clean_time(self):
        env = make_env(seed=0, budget=32)
        answer = DeterministicController().solve(env, beta=0.0)
        self.assertEqual(env.get_state().particles[answer.particle_id].status, "completed")
        self.assertEqual(answer.nfe_used, 5)
        self.assertIsNotNone(answer.reward)

    def test_best_of_n_compute_is_monotonic_in_beta(self):
        low = BestOfNController().solve(make_env(seed=0, budget=128), beta=0.0)
        high = BestOfNController().solve(make_env(seed=0, budget=128), beta=1.0)
        self.assertGreaterEqual(high.nfe_used, low.nfe_used)
        self.assertGreaterEqual(
            sum(event.action == "SPAWN" for event in high.event_log),
            sum(event.action == "SPAWN" for event in low.event_log),
        )

    def test_self_refine_uses_preview_and_backward(self):
        answer = SelfRefineController().solve(make_env(seed=1, budget=64), beta=0.5)
        actions = [event.action for event in answer.event_log]
        self.assertIn("PREVIEW", actions)
        self.assertIn("BACKWARD", actions)

    def test_prism_uses_prune_and_preview(self):
        answer = PrismStyleFlowController().solve(make_env(seed=2, budget=128), beta=0.5)
        actions = [event.action for event in answer.event_log]
        self.assertIn("PREVIEW", actions)
        self.assertIn("PRUNE", actions)


if __name__ == "__main__":
    unittest.main()
