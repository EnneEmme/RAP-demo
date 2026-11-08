"""Prompt templates for Blocksworld planning in hybrid and paper modes."""

from __future__ import annotations

from typing import Sequence
from rap_demo.domains.blocksworld.models import BlockAction, BlockState


HYBRID_SYSTEM_PROMPT = """You are a reasoning and planning agent for the Blocksworld domain.
Analyze the current state and propose the next action in JSON format.

JSON format:
```json
{
  "thought": "<your reasoning here>",
  "action": "<pick_up | put_down | stack | unstack>",
  "block": "<block name>",
  "target": "<target block name or null>"
}
```"""


def format_goal(goal: str | Sequence[str] | dict[str, str]) -> str:
    """Formats goal representation into a canonical natural language string."""
    if isinstance(goal, str):
        return goal
    elif isinstance(goal, dict):
        clauses = [f"the {k} block is on top of the {v} block" for k, v in sorted(goal.items())]
        if not clauses:
            return ""
        if len(clauses) == 1:
            return clauses[0]
        return ", ".join(clauses[:-1]) + " and " + clauses[-1]
    else:
        clauses = list(goal)
        if not clauses:
            return ""
        if len(clauses) == 1:
            return clauses[0]
        return ", ".join(clauses[:-1]) + " and " + clauses[-1]


def build_hybrid_action_prompt(state: BlockState, goal: str | Sequence[str] | dict[str, str] = "") -> str:
    """Builds prompt requesting a structured JSON action proposal with internal reasoning thought."""
    return (
        f"{HYBRID_SYSTEM_PROMPT}\n\n"
        f"Current State: {state.to_text()}\n\n"
        f"Output your proposal in valid JSON:"
    )


def build_paper_action_prompt(state: BlockState, goal: str | Sequence[str] | dict[str, str] = "") -> str:
    """Builds prompt matching the canonical RAP paper format for action generation."""
    return (
        f"[STATE] {state.to_text()}\n"
        f"[ACTION]"
    )


def build_world_update_prompt(state: BlockState, action: BlockAction) -> str:
    """Builds prompt querying LLM for natural language state change description."""
    return (
        f"[STATE] {state.to_text()}\n"
        f"[ACTION] {action.to_text()}\n"
        f"[CHANGE]"
    )


def build_self_eval_prompt(
    state: BlockState,
    action: BlockAction,
    goal: str | Sequence[str] | dict[str, str],
) -> str:
    """Builds prompt querying LLM for self-evaluation of action quality towards goal."""
    goal_str = format_goal(goal)
    return (
        f"Current state: {state.to_text()}\n"
        f"Goal: {goal_str}\n"
        f"Proposed action: {action.to_text()}\n"
        f"Is this action promising to achieve the goal? Answer Yes or No:"
    )
