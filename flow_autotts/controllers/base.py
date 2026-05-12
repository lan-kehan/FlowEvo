"""Controller protocol."""

from __future__ import annotations

from typing import Protocol

from flow_autotts.core.env import FlowTTSEnv
from flow_autotts.core.state import AnswerRecord


class Controller(Protocol):
    def solve(self, env: FlowTTSEnv, beta: float) -> AnswerRecord:
        """Run a controller policy and terminate with ``env.answer``."""
