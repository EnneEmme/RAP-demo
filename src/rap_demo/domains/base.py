"""Base domain abstractions for RAP reasoning problems."""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable
from rap_demo.core.mcts import ReasoningAgent, WorldModel


@runtime_checkable
class Domain(Protocol):
    """Abstract protocol for planning domains in RAP."""

    def get_initial_state(self) -> Any:
        """Returns the initial problem state."""
        ...

    def is_goal_reached(self, state: Any) -> bool:
        """Determines if the given state satisfies the goal condition."""
        ...

    def create_agent(self, **kwargs: Any) -> ReasoningAgent:
        """Creates the reasoning agent for action proposal in this domain."""
        ...

    def create_world_model(self, **kwargs: Any) -> WorldModel:
        """Creates the world model for state transition simulation."""
        ...

    def create_reward_fn(self, **kwargs: Any) -> Callable[..., float]:
        """Creates the reward calculation function for this domain."""
        ...
