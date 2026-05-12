"""Internal latent stores for the online environment.

Controllers never receive these records. They are kept separate from public
state records to preserve the visibility boundary required by the spec.
"""

from __future__ import annotations

from dataclasses import dataclass

from flow_autotts.core.state import ParticleStatus


@dataclass
class ParticleRecord:
    id: int
    z: tuple[float, ...]
    time: float
    parent_id: int | None
    source_anchor_id: int | None
    nfe_used: int
    status: ParticleStatus
    last_preview_id: int | None = None
    num_children: int = 0
    sde_variance: float = 0.0


@dataclass
class AnchorRecord:
    id: int
    particle_id: int
    time: float
    z1_hat: tuple[float, ...]
    z0_hat: tuple[float, ...]
    score: float | None
    score_dict: dict[str, float]
    uncertainty: float | None = None
    drift: float | None = None
    embedding_ref: str | None = None
