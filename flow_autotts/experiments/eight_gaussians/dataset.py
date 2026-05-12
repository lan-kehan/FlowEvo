"""Utilities for a 2D eight-Gaussians target distribution."""

from __future__ import annotations

import math
import random
from typing import Sequence

Vector2 = tuple[float, float]


def eight_gaussian_centers(radius: float = 2.0) -> list[Vector2]:
    return [
        (radius * math.cos(2.0 * math.pi * i / 8.0), radius * math.sin(2.0 * math.pi * i / 8.0))
        for i in range(8)
    ]


def sample_eight_gaussians(
    n: int,
    std: float = 0.08,
    seed: int | None = None,
    radius: float = 2.0,
) -> list[Vector2]:
    rng = random.Random(seed)
    centers = eight_gaussian_centers(radius)
    samples: list[Vector2] = []
    for _ in range(n):
        cx, cy = rng.choice(centers)
        samples.append((rng.gauss(cx, std), rng.gauss(cy, std)))
    return samples


def nearest_center(
    point: Sequence[float],
    radius: float = 2.0,
) -> tuple[Vector2, float]:
    centers = eight_gaussian_centers(radius)
    best = min(centers, key=lambda c: squared_distance(point, c))
    return best, math.sqrt(squared_distance(point, best))


def squared_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b, strict=True))
