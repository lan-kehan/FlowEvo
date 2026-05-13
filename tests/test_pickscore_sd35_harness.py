from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from flow_autotts.controllers import PrismStyleFlowController
from flow_autotts.experiments.pickscore_sd35.dataset import sample_prompt_file
from flow_autotts.experiments.pickscore_sd35.env import (
    SD35EnvConfig,
    SD35PickScoreEnv,
    SD35Resources,
)
from flow_autotts.experiments.pickscore_sd35.harness import _round_controller_names


class PickScoreSD35HarnessTests(unittest.TestCase):
    def test_prompt_sampling_is_seeded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir)
            (dataset / "train.txt").write_text(
                "\n".join(f"prompt {idx}" for idx in range(20)),
                encoding="utf-8",
            )

            first = sample_prompt_file(dataset, sample_size=5, seed=123)
            second = sample_prompt_file(dataset, sample_size=5, seed=123)
            other = sample_prompt_file(dataset, sample_size=5, seed=124)

        self.assertEqual(first, second)
        self.assertNotEqual([item.index for item in first], [item.index for item in other])
        self.assertEqual(len(first), 5)

    def test_default_five_round_controller_schedule(self):
        self.assertEqual(
            _round_controller_names(5, None),
            ["optimal", "optimal", "optimal", "optimal", "optimal"],
        )

    @unittest.skipIf(torch is None, "torch is provided by the sd35 dependency group")
    def test_prism_controller_runs_on_discrete_sd35_time_grid(self):
        env = SD35PickScoreEnv(
            resources=_fake_resources(),
            prompt="a test prompt",
            seed=7,
            budget=64,
            config=SD35EnvConfig(
                resolution=4,
                num_steps=10,
                guidance_scale=1.0,
                noise_level=0.0,
            ),
        )

        answer = PrismStyleFlowController().solve(env, beta=0.5)

        self.assertIsNotNone(answer.reward)
        self.assertGreater(answer.nfe_used, 0)

if torch is not None:

    class _FakeTransformer:
        config = type("Config", (), {"in_channels": 1})()

        def __call__(self, hidden_states, **_kwargs):
            return (torch.ones_like(hidden_states) * 0.05,)


    class _FakeVAE:
        config = type("Config", (), {"scaling_factor": 1.0, "shift_factor": 0.0})()
        dtype = torch.float32

        def decode(self, latents, return_dict=False):
            del return_dict
            return (latents,)


    class _FakeImageProcessor:
        def postprocess(self, image, output_type="pil"):
            del output_type
            return [float(image.mean().item())]


    class _FakePipeline:
        transformer = _FakeTransformer()
        vae = _FakeVAE()
        image_processor = _FakeImageProcessor()
        vae_scale_factor = 1

        def prepare_latents(self, batch_size, channels, height, width, dtype, device, generator, latents):
            del latents
            return torch.randn(
                (batch_size, channels, height, width),
                generator=generator,
                device=device,
                dtype=dtype,
            )

        def encode_prompt(self, **_kwargs):
            positive = torch.ones((1, 1), dtype=torch.float32)
            negative = torch.zeros((1, 1), dtype=torch.float32)
            return positive, negative, positive, negative


    class _FakeScorer:
        def __call__(self, prompts, images):
            del prompts
            return [float(image) for image in images]


    def _fake_resources() -> SD35Resources:
        sigmas = torch.linspace(1.0, 0.0, 11)
        timesteps = torch.linspace(1000.0, 0.0, 10)
        return SD35Resources(
            pipeline=_FakePipeline(),
            scorer=_FakeScorer(),
            torch=torch,
            device="cpu",
            text_encoder_device="cpu",
            dtype=torch.float32,
            timesteps=timesteps,
            sigmas=sigmas,
        )


if __name__ == "__main__":
    unittest.main()
