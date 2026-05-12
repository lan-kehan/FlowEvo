from __future__ import annotations

import unittest

from flow_autotts.core.errors import BudgetExceededError
from tests.test_env_actions import make_env


class BudgetAndLoggingTests(unittest.TestCase):
    def test_budget_decreases_on_nfe_actions_only(self):
        env = make_env(budget=2)
        particle_id = env.spawn(1)[0]
        self.assertEqual(env.budget_left, 2)
        env.forward(particle_id, 0.5)
        self.assertEqual(env.budget_left, 1)
        env.preview(particle_id)
        self.assertEqual(env.budget_left, 0)

        actions = [event.action for event in env.get_state().event_log]
        self.assertEqual(actions, ["SPAWN", "FORWARD", "PREVIEW"])

    def test_exceeding_budget_raises(self):
        env = make_env(budget=1)
        particle_id = env.spawn(1)[0]
        env.forward(particle_id, 0.5)
        with self.assertRaises(BudgetExceededError):
            env.preview(particle_id)

    def test_controller_state_hides_latents(self):
        env = make_env()
        particle_id = env.spawn(1)[0]
        state = env.get_state()
        self.assertFalse(hasattr(state.particles[particle_id], "z"))
        self.assertFalse(hasattr(state, "anchors"))


if __name__ == "__main__":
    unittest.main()
