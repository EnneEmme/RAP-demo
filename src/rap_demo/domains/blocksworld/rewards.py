"""Reward functions, goal heuristics, and self-evaluation scoring for Blocksworld."""

from __future__ import annotations

import re
from typing import Sequence
from rap_demo.domains.blocksworld.models import BlockAction, BlockState
from rap_demo.domains.blocksworld.prompts import build_self_eval_prompt
from rap_demo.llm.base import BaseLLM


def _is_condition_satisfied(state: BlockState, condition: str) -> bool:
    """Checks whether a single natural language condition holds in the given state."""
    cond = condition.strip().rstrip(".").lower()

    # Pattern: the X block is on top of the Y block
    m_on = re.search(r"the\s+(\w+)\s+block\s+is\s+on\s+top\s+of\s+the\s+(\w+)\s+block", cond)
    if m_on:
        top = m_on.group(1).lower()
        bottom = m_on.group(2).lower()
        return state.on.get(top) == bottom

    # Pattern: the X block is on the table
    m_table = re.search(r"the\s+(\w+)\s+block\s+is\s+on\s+the\s+table", cond)
    if m_table:
        block = m_table.group(1).lower()
        return block in state.on_table

    # Pattern: the X block is clear
    m_clear = re.search(r"the\s+(\w+)\s+block\s+is\s+clear", cond)
    if m_clear:
        block = m_clear.group(1).lower()
        return block in state.clear

    # Pattern: hand is empty
    if "hand is empty" in cond:
        return state.holding is None

    # Pattern: holding X
    m_holding = re.search(r"(?:hand\s+is\s+holding\s+the|holding\s+the)\s+(\w+)\s+block", cond)
    if m_holding:
        return state.holding == m_holding.group(1).lower()

    return False


def calculate_goal_reward(state: BlockState, goals: Sequence[str] | dict[str, str] | str) -> float:
    """Calculates goal heuristic reward: fraction of satisfied conditions (+100 bonus on full satisfaction)."""
    conditions: list[str] = []

    if isinstance(goals, str):
        # Could be single or combined with " and " / ","
        parts = re.split(r",\s*and\s+|\s+and\s+|,", goals)
        conditions = [p.strip() for p in parts if p.strip()]
    elif isinstance(goals, dict):
        for top, bottom in goals.items():
            conditions.append(f"the {top} block is on top of the {bottom} block")
    else:
        for g in goals:
            if isinstance(g, str):
                parts = re.split(r",\s*and\s+|\s+and\s+|,", g)
                conditions.extend([p.strip() for p in parts if p.strip()])

    if not conditions:
        return 100.0

    satisfied = sum(1 for c in conditions if _is_condition_satisfied(state, c))
    total = len(conditions)

    if satisfied == total:
        return 100.0
    return float(satisfied) / float(total)


def calculate_combined_reward(
    state: BlockState,
    action: BlockAction,
    goals: Sequence[str] | dict[str, str] | str,
    action_score: float = 1.0,
    alpha: float = 0.5,
    eval_score: float | None = None,
) -> float:
    """Computes combined step reward combining action likelihood/confidence and goal progress."""
    goal_or_eval = eval_score if eval_score is not None else calculate_goal_reward(state, goals)
    return float(alpha * action_score + (1.0 - alpha) * goal_or_eval)


def score_self_evaluation(llm: BaseLLM, prompt: str) -> float:
    """Calculates normalized self-evaluation probability from LLM."""
    p_yes = llm.get_token_probability(prompt, "Yes")
    p_no = llm.get_token_probability(prompt, "No")

    total = p_yes + p_no
    if total > 0.0:
        return float(p_yes / total)
    return float(p_yes)


class BlocksworldRewardFunction:
    """Callable reward function combining goal heuristic, action scores, and LLM self-eval."""

    def __init__(
        self,
        goal: Sequence[str] | dict[str, str] | str,
        llm: BaseLLM | None = None,
        alpha: float = 0.5,
        use_self_eval: bool = False,
    ) -> None:
        self.goal = goal
        self.llm = llm
        self.alpha = alpha
        self.use_self_eval = use_self_eval

    def __call__(
        self,
        state: BlockState,
        action: BlockAction,
        action_score: float = 1.0,
    ) -> float:
        eval_score: float | None = None
        if self.use_self_eval and self.llm is not None:
            prompt = build_self_eval_prompt(state, action, self.goal)
            eval_score = score_self_evaluation(self.llm, prompt)

        return calculate_combined_reward(
            state=state,
            action=action,
            goals=self.goal,
            action_score=action_score,
            alpha=self.alpha,
            eval_score=eval_score,
        )
