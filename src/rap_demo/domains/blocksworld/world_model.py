"""World model simulation for Blocksworld."""

from __future__ import annotations

import re
from typing import Optional, Sequence
from rap_demo.core.mcts import WorldModel
from rap_demo.domains.blocksworld.models import ActionType, BlockAction, BlockState
from rap_demo.domains.blocksworld.prompts import build_world_update_prompt
from rap_demo.domains.blocksworld.rewards import calculate_combined_reward, calculate_goal_reward
from rap_demo.llm.base import BaseLLM


def _apply_text_change(state: BlockState, change_text: str) -> BlockState:
    """Applies a natural language state transition description to a BlockState."""
    cleaned = change_text.strip()
    # Remove prefix like [CHANGE]
    cleaned = re.sub(r"^\[change\]:?\s*", "", cleaned, flags=re.IGNORECASE).strip()

    # Reject hallucinated / impossible change texts
    hallucination_indicators = ["disappear", "destroy", "vanish", "teleport", "unknown", "invalid", "broken"]
    if any(h in cleaned.lower() for h in hallucination_indicators):
        raise ValueError(f"State update rejected due to hallucinated physics in LLM output: {change_text!r}")

    new_clear = set(state.clear)
    new_on_table = set(state.on_table)
    new_on = dict(state.on)
    new_holding = state.holding

    raw_clauses = [c.strip().rstrip(".") for c in re.split(r"[,;]\s*(?:and\s+)?", cleaned) if c.strip()]
    if not raw_clauses:
        raise ValueError(f"No valid state change clauses found in: {change_text!r}")

    matched_any = False
    for raw_clause in raw_clauses:
        clause = re.sub(r"^and\s+", "", raw_clause.strip(), flags=re.IGNORECASE)
        c_lower = clause.lower()

        # 1. Hand holding
        m = re.search(r"(?:hand\s+was\s+empty\s+and\s+now\s+is\s+holding\s+the|holding\s+the)\s+(\w+)\s+block", c_lower)
        if m:
            new_holding = m.group(1)
            matched_any = True
            continue

        # 2. Hand now empty
        if ("hand was holding" in c_lower and "now is empty" in c_lower) or "hand is empty" in c_lower:
            new_holding = None
            matched_any = True
            continue

        # 3. Block was on table and now in hand
        m = re.search(r"the\s+(\w+)\s+block\s+was\s+on\s+the\s+table\s+and\s+now\s+is\s+in\s+the\s+hand", c_lower)
        if m:
            b = m.group(1)
            new_on_table.discard(b)
            new_holding = b
            matched_any = True
            continue

        # 4. Block was in hand and now on table
        m = re.search(r"the\s+(\w+)\s+block\s+was\s+in\s+the\s+hand\s+and\s+now\s+is\s+on\s+the\s+table", c_lower)
        if m:
            b = m.group(1)
            new_on_table.add(b)
            new_holding = None
            matched_any = True
            continue

        # 5. Block was in hand and now on top of target
        m = re.search(
            r"the\s+(\w+)\s+block\s+was\s+in\s+the\s+hand\s+and\s+now\s+is\s+on\s+top\s+of\s+the\s+(\w+)\s+block",
            c_lower,
        )
        if m:
            top = m.group(1)
            bottom = m.group(2)
            new_on[top] = bottom
            new_holding = None
            matched_any = True
            continue

        # 6. Block was on top of target and now in hand
        m = re.search(
            r"the\s+(\w+)\s+block\s+was\s+on\s+top\s+of\s+the\s+(\w+)\s+block\s+and\s+now\s+is\s+in\s+the\s+hand",
            c_lower,
        )
        if m:
            top = m.group(1)
            new_on.pop(top, None)
            new_holding = top
            matched_any = True
            continue

        # 7. Block is no longer clear
        m = re.search(r"the\s+(\w+)\s+block\s+is\s+no\s+longer\s+clear", c_lower)
        if m:
            new_clear.discard(m.group(1))
            matched_any = True
            continue

        # 8. Block is now clear
        m = re.search(r"the\s+(\w+)\s+block\s+is\s+(?:now\s+)?clear", c_lower)
        if m:
            new_clear.add(m.group(1))
            matched_any = True
            continue

        # If a clause couldn't be matched, it's an unrecognized physical effect
        raise ValueError(f"Unrecognized transition clause in change text: {clause!r}")

    if not matched_any:
        raise ValueError(f"No recognized state change in text: {change_text!r}")

    return BlockState(
        clear=new_clear,
        on_table=new_on_table,
        on=new_on,
        holding=new_holding,
    )


class BlocksworldWorldModel(WorldModel):
    """Simulates transitions and rewards in Blocksworld in hybrid or paper mode."""

    def __init__(
        self,
        goal: Sequence[str] | dict[str, str] | str,
        llm: Optional[BaseLLM] = None,
        mode: str = "hybrid",
        alpha: float = 0.5,
    ) -> None:
        self.goal = goal
        self.llm = llm
        self.mode = mode
        self.alpha = alpha

    def step(self, state: BlockState, action: BlockAction) -> tuple[BlockState, float, bool]:
        """Given current state and action, returns (next_state, immediate_reward, is_terminal)."""
        if self.mode == "hybrid":
            next_state = state.apply_action(action)
        elif self.mode == "paper":
            if self.llm is None:
                raise ValueError("LLM backend is required for paper mode world model simulation")
            prompt = build_world_update_prompt(state, action)
            change_text = self.llm.generate(prompt)
            next_state = _apply_text_change(state, change_text)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        goal_reward = calculate_goal_reward(next_state, self.goal)
        is_terminal = goal_reward >= 100.0

        if is_terminal:
            step_reward = goal_reward
        else:
            step_reward = calculate_combined_reward(
                state=next_state,
                action=action,
                goals=self.goal,
                action_score=1.0,
                alpha=self.alpha,
            )

        return next_state, float(step_reward), is_terminal
