"""Prompt text for controller explorer agents."""

EXPLORER_PROMPT = """\
You are an explorer for training-free flow-matching test-time scaling controller discovery.

Implement class OptimalController with:

class OptimalController:
    def solve(self, env, beta):
        ...

Rules:
- Use only public environment methods: spawn, forward, preview, backward, prune, answer.
- PREVIEW is the only semantic observation action.
- Expose exactly one hyperparameter, beta in [0, 1].
- Larger beta should monotonically correspond to larger compute.
- Optimize the reward-NFE Pareto frontier, not only reward at one budget.
"""
