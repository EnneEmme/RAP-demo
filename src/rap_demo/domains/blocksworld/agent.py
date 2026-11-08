"""Reasoning agent implementation for Blocksworld."""

from __future__ import annotations

import json
import re
from typing import Any
from rap_demo.core.mcts import ReasoningAgent
from rap_demo.domains.blocksworld.models import ActionType, BlockAction, BlockState, HybridActionProposal
from rap_demo.domains.blocksworld.prompts import build_hybrid_action_prompt, build_paper_action_prompt
from rap_demo.llm.base import BaseLLM


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    """Extracts a JSON object from text, handling markdown code blocks and conversational commentary."""
    # 1. Match code blocks with or without json tag
    markdown_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if markdown_match:
        try:
            return json.loads(markdown_match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    # 2. Match first balanced or outermost curly-bracket object
    brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Direct parse of stripped text
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return None


class BlocksworldAgent(ReasoningAgent):
    """Generates candidate reasoning steps/actions for Blocksworld in hybrid or paper mode."""

    def __init__(
        self,
        llm: BaseLLM,
        goal: str,
        mode: str = "hybrid",
        num_candidates: int = 4,
        temperature: float = 0.0,
    ) -> None:
        self.llm = llm
        self.goal = goal
        self.mode = mode
        self.num_candidates = num_candidates
        self.temperature = temperature
        self.last_proposals: list[HybridActionProposal] = []

    def get_actions(self, state: BlockState) -> list[BlockAction]:
        """Proposes candidate actions from current state."""
        if self.mode == "hybrid":
            return self._get_hybrid_actions(state)
        elif self.mode == "paper":
            return self._get_paper_actions(state)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _get_hybrid_actions(self, state: BlockState) -> list[BlockAction]:
        """Hybrid mode: requests structured JSON with thought, parses action, verifies legality."""
        prompt = build_hybrid_action_prompt(state, self.goal)
        response = self.llm.generate(prompt, temperature=self.temperature)

        payload = _extract_json_payload(response)
        if payload is not None:
            try:
                proposal = HybridActionProposal.model_validate(payload)
                self.last_proposals = [proposal]
                action = proposal.to_block_action()
                if state.is_legal_action(action):
                    return [action]
            except Exception:
                pass

        # Fallback to legal actions if parsing fails or proposed action is illegal
        legal_actions = state.get_legal_actions()
        return legal_actions[: self.num_candidates] if legal_actions else []

    def _get_paper_actions(self, state: BlockState) -> list[BlockAction]:
        """Paper mode: uses bare text generation, ranking legal actions by LLM logprobs."""
        self.last_proposals = []
        prompt = build_paper_action_prompt(state, self.goal)
        legal_actions = state.get_legal_actions()
        if not legal_actions:
            return []

        # Attempt standard LLM text generation
        try:
            raw_response = self.llm.generate(prompt, temperature=self.temperature)
            _ = BlockAction.from_text(raw_response)
        except Exception:
            # Paper mode handles deviations by falling back to legal actions
            pass

        # Score and rank legal actions by likelihood (logprobs)
        scored_actions: list[tuple[float, BlockAction]] = []
        for action in legal_actions:
            action_text = action.to_text()
            logprob = self.llm.get_logprobs(prompt, action_text)
            scored_actions.append((logprob, action))

        scored_actions.sort(key=lambda item: item[0], reverse=True)
        return [action for _, action in scored_actions[: self.num_candidates]]
