"""Prompt loading and random subset selection for PickScore experiments."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptSample:
    """One selected prompt plus deterministic generation metadata."""

    index: int
    prompt: str
    seed: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def load_prompt_file(dataset_dir: str | Path, split: str = "train") -> list[str]:
    path = Path(dataset_dir) / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(f"prompt file not found: {path}")
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [prompt for prompt in prompts if prompt]


def sample_prompt_file(
    dataset_dir: str | Path,
    split: str = "train",
    sample_size: int = 100,
    seed: int = 42,
) -> list[PromptSample]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    prompts = load_prompt_file(dataset_dir=dataset_dir, split=split)
    if sample_size > len(prompts):
        raise ValueError(
            f"sample_size={sample_size} is larger than {split} split size={len(prompts)}"
        )

    rng = random.Random(seed)
    indices = rng.sample(range(len(prompts)), sample_size)
    return [
        PromptSample(
            index=index,
            prompt=prompts[index],
            seed=_stable_seed(seed=seed, prompt_index=index, rank=rank),
        )
        for rank, index in enumerate(indices)
    ]


def _stable_seed(seed: int, prompt_index: int, rank: int) -> int:
    # Keep seeds in torch's portable positive int32 range.
    return (int(seed) * 1_000_003 + int(prompt_index) * 9_176 + int(rank)) % (2**31 - 1)
