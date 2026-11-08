"""RAP-Aggregation module implementing weighted majority voting (RAP Paper Section 3.4)."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence
from rap_demo.core.node import MCTSNode


@dataclass
class CandidateTrajectory:
    """Represents an MCTS terminal trajectory or rollout candidate for aggregation."""

    answer: Any
    reward: float
    depth: int = 1
    trajectory: list[MCTSNode] = field(default_factory=list)


@dataclass
class AggregationResult:
    """Outcome of RAP weighted majority voting aggregation."""

    best_answer: Any
    confidence: float
    score: float
    votes: dict[Any, float] = field(default_factory=dict)
    candidates_count: int = 0


class RAPAggregation:
    """Aggregates multiple MCTS search paths into a single robust consensus answer."""

    def __init__(self, depth_weighting: bool = True, min_reward: float = 0.0) -> None:
        self.depth_weighting = depth_weighting
        self.min_reward = min_reward

    def aggregate(self, candidates: Sequence[CandidateTrajectory]) -> AggregationResult:
        """Computes weighted majority vote across candidate trajectories."""
        if not candidates:
            return AggregationResult(
                best_answer=None,
                confidence=0.0,
                score=0.0,
                votes={},
                candidates_count=0,
            )

        valid_candidates = [c for c in candidates if c.reward >= self.min_reward]
        if not valid_candidates:
            return AggregationResult(
                best_answer=None,
                confidence=0.0,
                score=0.0,
                votes={},
                candidates_count=len(candidates),
            )

        votes: dict[Any, float] = {}
        for c in valid_candidates:
            if self.depth_weighting:
                weight = c.reward / max(c.depth, 1)
            else:
                weight = c.reward
            votes[c.answer] = votes.get(c.answer, 0.0) + weight

        # Pick candidate answer with maximum accumulated weight (deterministic tie-breaking)
        best_answer, best_score = max(votes.items(), key=lambda item: item[1])

        total_weight = sum(votes.values())
        if total_weight > 0.0:
            confidence = best_score / total_weight
        else:
            confidence = 1.0 / len(votes) if votes else 0.0

        return AggregationResult(
            best_answer=best_answer,
            confidence=confidence,
            score=best_score,
            votes=votes,
            candidates_count=len(candidates),
        )

    def aggregate_from_tree(
        self,
        root: MCTSNode,
        answer_extractor: Optional[Callable[[MCTSNode], Any]] = None,
    ) -> AggregationResult:
        """Extracts terminal nodes from the MCTS tree and aggregates their predictions."""
        terminal_nodes: list[MCTSNode] = []

        stack = [root]
        while stack:
            curr = stack.pop()
            if curr.is_terminal:
                terminal_nodes.append(curr)
            for child in curr.children:
                stack.append(child)

        candidates = [
            CandidateTrajectory(
                answer=answer_extractor(node) if answer_extractor is not None else node.state,
                reward=node.reward,
                depth=node.depth,
                trajectory=node.path_from_root,
            )
            for node in terminal_nodes
        ]

        return self.aggregate(candidates)
