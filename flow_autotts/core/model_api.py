"""Protocols for toy or production flow models and scorers."""

from __future__ import annotations

from typing import Any, Protocol, Sequence


Vector = tuple[float, ...]


class FlowModelProtocol(Protocol):
    def velocity(
        self,
        z: Sequence[float],
        t: float,
        condition: Any = None,
    ) -> Sequence[float]:
        """Return a velocity estimate at latent state ``z`` and time ``t``."""


class ScorerProtocol(Protocol):
    def score(
        self,
        z1_hat: Sequence[float],
        condition: Any = None,
    ) -> tuple[float, dict[str, float]]:
        """Return a scalar reward and optional diagnostics for a clean latent."""


class IdentityVAE:
    """Minimal VAE-like object for toy latent experiments."""

    def decode(self, latent: Sequence[float]) -> Vector:
        return tuple(float(x) for x in latent)
