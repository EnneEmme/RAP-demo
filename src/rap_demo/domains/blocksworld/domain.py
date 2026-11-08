"""Blocksworld domain implementation adhering to the Domain abstraction."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence
from rap_demo.domains.base import Domain
from rap_demo.domains.blocksworld.agent import BlocksworldAgent
from rap_demo.domains.blocksworld.models import BlockState
from rap_demo.domains.blocksworld.rewards import BlocksworldRewardFunction, calculate_goal_reward
from rap_demo.domains.blocksworld.world_model import BlocksworldWorldModel
from rap_demo.llm.base import BaseLLM


class BlocksworldDomain(Domain):
    """Encapsulates a Blocksworld problem instance, creating agent, world model, and reward fn."""

    def __init__(
        self,
        initial_state: BlockState,
        goal: Sequence[str] | dict[str, str] | str,
        llm: Optional[BaseLLM] = None,
        mode: str = "hybrid",
    ) -> None:
        self.initial_state = initial_state
        self.goal = goal
        self.llm = llm
        self.mode = mode

    def get_initial_state(self) -> BlockState:
        """Returns the initial BlockState."""
        return self.initial_state

    def is_goal_reached(self, state: BlockState) -> bool:
        """Determines if the block configuration satisfies all goal conditions."""
        return calculate_goal_reward(state, self.goal) >= 100.0

    def create_agent(self, **kwargs: Any) -> BlocksworldAgent:
        """Creates BlocksworldAgent instance for action proposal."""
        llm = kwargs.pop("llm", self.llm)
        if llm is None:
            raise ValueError("An LLM backend is required to create a BlocksworldAgent")
        mode = kwargs.pop("mode", self.mode)
        goal = kwargs.pop("goal", self.goal)
        if isinstance(goal, str):
            goal_str = goal
        elif isinstance(goal, dict):
            goal_str = " and ".join(f"the {k} block is on top of the {v} block" for k, v in goal.items())
        else:
            goal_str = " and ".join(goal)

        return BlocksworldAgent(llm=llm, goal=goal_str, mode=mode, **kwargs)

    def create_world_model(self, **kwargs: Any) -> BlocksworldWorldModel:
        """Creates BlocksworldWorldModel instance for state transitions."""
        llm = kwargs.pop("llm", self.llm)
        mode = kwargs.pop("mode", self.mode)
        goal = kwargs.pop("goal", self.goal)
        return BlocksworldWorldModel(goal=goal, llm=llm, mode=mode, **kwargs)

    def create_reward_fn(self, **kwargs: Any) -> Callable[..., float]:
        """Creates reward calculation callable."""
        llm = kwargs.pop("llm", self.llm)
        goal = kwargs.pop("goal", self.goal)
        return BlocksworldRewardFunction(goal=goal, llm=llm, **kwargs)
