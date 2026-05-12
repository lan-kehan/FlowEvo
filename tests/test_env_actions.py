from __future__ import annotations

import math
import unittest

from flow_autotts.core.env import FlowTTSEnv


class ConstantFlow:
    def velocity(self, z, t, condition=None):
        return (1.0, -1.0)


class DistanceScorer:
    def score(self, z1_hat, condition=None):
        reward = -math.hypot(float(z1_hat[0]), float(z1_hat[1]))
        return reward, {"norm": -reward}


def make_env(budget=16):
    return FlowTTSEnv(
        model=ConstantFlow(),
        vae=None,
        scorer=DistanceScorer(),
        prompt="test",
        budget=budget,
        time_grid=[0.0, 0.5, 1.0],
        seed=0,
        latent_shape=(2,),
    )


class EnvActionTests(unittest.TestCase):
    def test_spawn_forward_preview_backward_prune_answer(self):
        env = make_env()
        particle_id = env.spawn(1)[0]
        state = env.get_state()
        self.assertEqual(state.active_particle_ids, [particle_id])
        self.assertEqual(state.particles[particle_id].time, 0.0)

        summary = env.forward(particle_id, 0.5)
        self.assertEqual(summary.time, 0.5)
        self.assertEqual(env.nfe_used, 1)

        preview = env.preview(particle_id)
        self.assertEqual(preview.particle_id, particle_id)
        self.assertIsNotNone(preview.score)
        self.assertEqual(env.nfe_used, 2)

        child_id = env.backward(preview.id, target_time=0.25, num_children=1)[0]
        self.assertIn(child_id, env.get_state().active_particle_ids)

        env.prune([particle_id])
        state = env.get_state()
        self.assertIn(particle_id, state.pruned_particle_ids)
        self.assertIn("PRUNE", [event.action for event in state.event_log])

        answer = env.answer(rule="best_preview_score")
        self.assertEqual(answer.preview_id, preview.id)
        self.assertIsNotNone(answer.reward)
        self.assertEqual(answer.event_log[-1].action, "ANSWER")

    def test_sde_forward_logs_noise_and_increases_late_uncertainty(self):
        env = make_env()
        particle_id = env.spawn(1)[0]
        env.forward(
            particle_id,
            1.0,
            solver="sde",
            cfg={"noise_scale": 0.2, "sigma_max": 0.5, "min_time": 0.05},
        )
        preview = env.preview(particle_id)
        forward_event = env.get_state().event_log[1]
        self.assertEqual(forward_event.details["solver"], "sde")
        self.assertGreater(forward_event.details["noise_std"], 0.0)
        self.assertGreater(preview.uncertainty, 0.0)


if __name__ == "__main__":
    unittest.main()
