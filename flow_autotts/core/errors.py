"""Environment exceptions."""


class FlowTTSError(RuntimeError):
    """Base class for environment failures."""


class BudgetExceededError(FlowTTSError):
    """Raised when an action would exceed the episode budget."""


class InvalidActionError(FlowTTSError):
    """Raised when a controller requests an invalid action."""
