"""Analytic 2D toy flow and scorer for eight Gaussians.

This is intentionally lightweight: it behaves like a learned flow model's
velocity interface without requiring torch/numpy or a training step.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from flow_autotts.experiments.eight_gaussians.dataset import (
    eight_gaussian_centers,
    nearest_center,
    squared_distance,
)


class EightGaussiansFlowModel:
    """A deterministic toy velocity field for the eight-Gaussians target."""

    def __init__(self, radius: float = 2.0, min_temperature: float = 0.20) -> None:
        self.radius = float(radius)
        self.min_temperature = float(min_temperature)
        self.centers = eight_gaussian_centers(self.radius)

    def velocity(
        self,
        z: Sequence[float],
        t: float,
        condition: Any = None,
    ) -> tuple[float, float]:
        clean = self.predict_clean_anchor(z, t)
        denom = max(1.0 - float(t), 1e-3)
        return ((clean[0] - float(z[0])) / denom, (clean[1] - float(z[1])) / denom)

    def predict_clean_anchor(self, z: Sequence[float], t: float) -> tuple[float, float]:
        temperature = max(self.min_temperature, 1.35 - 1.10 * float(t))
        weights = []
        for center in self.centers:
            weights.append(math.exp(-squared_distance(z, center) / (2.0 * temperature * temperature)))
        total = sum(weights)
        if total <= 1e-12:
            return nearest_center(z, self.radius)[0]
        x = sum(weight * center[0] for weight, center in zip(weights, self.centers, strict=True)) / total
        y = sum(weight * center[1] for weight, center in zip(weights, self.centers, strict=True)) / total

        norm = math.hypot(x, y)
        if norm < 1e-8:
            center, _ = nearest_center(z, self.radius)
            return center
        # Project toward the data ring so Euler steps converge to clean modes.
        return (self.radius * x / norm, self.radius * y / norm)


class EightGaussiansScorer:
    """Reward clean anchors by proximity to the nearest mode."""

    def __init__(self, radius: float = 2.0, std: float = 0.08) -> None:
        self.radius = float(radius)
        self.std = float(std)

    def score(
        self,
        z1_hat: Sequence[float],
        condition: Any = None,
    ) -> tuple[float, dict[str, float]]:
        _, distance = nearest_center(z1_hat, self.radius)
        log_prob_proxy = -0.5 * (distance / self.std) ** 2
        reward = -distance
        return reward, {
            "nearest_center_distance": distance,
            "log_prob_proxy": log_prob_proxy,
        }
