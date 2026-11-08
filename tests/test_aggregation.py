"""Comprehensive tests for RAP-Aggregation (RAP Paper Section 3.4).

Tests cover:
- Weighted majority voting across rollouts and candidate trajectories.
- Trajectory depth-weighting vs unweighted accumulation.
- Filtering of low/negative rewards.
- Tree-based aggregation from MCTS terminal nodes.
- Edge cases: ties, empty inputs, single candidate, unanimous consensus, zero rewards.
"""

from __future__ import annotations
import math
from typing import Any
import pytest

from rap_demo.core.node import MCTSNode
from rap_demo.core.aggregation import (
    RAPAggregation,
    AggregationResult,
    CandidateTrajectory,
)


class TestRAPAggregationContract:
    """Verifies dataclass structures and basic initialization."""

    def test_candidate_trajectory_creation(self) -> None:
        """Verifies CandidateTrajectory fields and defaults."""
        cand = CandidateTrajectory(answer="42", reward=10.0, depth=3)
        assert cand.answer == "42"
        assert cand.reward == 10.0
        assert cand.depth == 3
        assert cand.trajectory == []

    def test_aggregation_result_creation(self) -> None:
        """Verifies AggregationResult structure."""
        result = AggregationResult(
            best_answer="42",
            confidence=0.85,
            score=12.5,
            votes={"42": 12.5, "17": 2.2},
            candidates_count=5,
        )
        assert result.best_answer == "42"
        assert result.confidence == 0.85
        assert result.score == 12.5
        assert result.votes["42"] == 12.5
        assert result.candidates_count == 5


class TestWeightedMajorityVoting:
    """Tests RAP-Aggregation voting mechanics and mathematical properties."""

    def test_weighted_majority_selects_highest_cumulative_weight(self) -> None:
        """Answer with the highest accumulated weight wins, even if minority count.

        Answer "A" has 2 paths with small weight: 1.0 + 1.0 = 2.0
        Answer "B" has 1 path with high weight:  5.0
        Winner must be "B" due to reward weighting.
        """
        aggregator = RAPAggregation(depth_weighting=False)
        candidates = [
            CandidateTrajectory(answer="A", reward=1.0, depth=1),
            CandidateTrajectory(answer="A", reward=1.0, depth=1),
            CandidateTrajectory(answer="B", reward=5.0, depth=1),
        ]

        result = aggregator.aggregate(candidates)

        assert result.best_answer == "B"
        assert result.score == 5.0
        assert result.votes["A"] == 2.0
        assert result.votes["B"] == 5.0
        assert pytest.approx(result.confidence, rel=1e-5) == 5.0 / 7.0
        assert result.candidates_count == 3

    def test_depth_weighting_favors_shorter_trajectories(self) -> None:
        """When depth_weighting is True, weight is reward / depth.

        Candidate 1: answer "X", reward 12.0, depth 2 -> weight = 12 / 2 = 6.0
        Candidate 2: answer "Y", reward 12.0, depth 4 -> weight = 12 / 4 = 3.0
        Winner must be "X".
        """
        aggregator = RAPAggregation(depth_weighting=True)
        candidates = [
            CandidateTrajectory(answer="X", reward=12.0, depth=2),
            CandidateTrajectory(answer="Y", reward=12.0, depth=4),
        ]

        result = aggregator.aggregate(candidates)

        assert result.best_answer == "X"
        assert pytest.approx(result.votes["X"], rel=1e-5) == 6.0
        assert pytest.approx(result.votes["Y"], rel=1e-5) == 3.0
        assert pytest.approx(result.confidence, rel=1e-5) == 6.0 / 9.0

    def test_unweighted_depth_accumulates_raw_rewards(self) -> None:
        """When depth_weighting is False, weight equals raw reward."""
        aggregator = RAPAggregation(depth_weighting=False)
        candidates = [
            CandidateTrajectory(answer="X", reward=10.0, depth=5),
            CandidateTrajectory(answer="X", reward=5.0, depth=2),
        ]

        result = aggregator.aggregate(candidates)

        assert result.best_answer == "X"
        assert pytest.approx(result.votes["X"], rel=1e-5) == 15.0
        assert result.confidence == 1.0

    def test_min_reward_filter_excludes_negative_and_sub_threshold(self) -> None:
        """Paths with reward below min_reward are excluded from voting."""
        aggregator = RAPAggregation(min_reward=0.0)
        candidates = [
            CandidateTrajectory(answer="GOOD", reward=5.0, depth=1),
            CandidateTrajectory(answer="BAD", reward=-10.0, depth=1),
            CandidateTrajectory(answer="ZERO", reward=0.0, depth=1),
        ]

        result = aggregator.aggregate(candidates)

        assert "BAD" not in result.votes
        assert result.best_answer == "GOOD"
        assert result.confidence == 1.0


class TestAggregationEdgeCases:
    """Verifies edge cases: empty lists, ties, single candidate, zero rewards."""

    def test_empty_candidates_handling(self) -> None:
        """Aggregating an empty candidate list returns empty result or raises ValueError."""
        aggregator = RAPAggregation()
        try:
            result = aggregator.aggregate([])
            assert result.best_answer is None
            assert result.confidence == 0.0
            assert result.candidates_count == 0
        except ValueError:
            pass  # Raising ValueError on empty input is also acceptable contract behavior

    def test_single_candidate(self) -> None:
        """A single valid candidate must yield 100% confidence."""
        aggregator = RAPAggregation()
        candidates = [CandidateTrajectory(answer="Solo", reward=8.0, depth=2)]

        result = aggregator.aggregate(candidates)

        assert result.best_answer == "Solo"
        assert result.confidence == 1.0
        assert result.candidates_count == 1

    def test_unanimous_agreement(self) -> None:
        """When all candidates propose the same answer, confidence is 1.0."""
        aggregator = RAPAggregation()
        candidates = [
            CandidateTrajectory(answer="Answer1", reward=2.0, depth=1),
            CandidateTrajectory(answer="Answer1", reward=4.0, depth=2),
            CandidateTrajectory(answer="Answer1", reward=8.0, depth=4),
        ]

        result = aggregator.aggregate(candidates)

        assert result.best_answer == "Answer1"
        assert result.confidence == 1.0
        assert len(result.votes) == 1

    def test_deterministic_tie_breaking(self) -> None:
        """When two candidates have identical weights, winner is deterministically chosen."""
        aggregator = RAPAggregation(depth_weighting=False)
        candidates = [
            CandidateTrajectory(answer="Alpha", reward=10.0, depth=2),
            CandidateTrajectory(answer="Beta", reward=10.0, depth=2),
        ]

        result1 = aggregator.aggregate(candidates)
        result2 = aggregator.aggregate(candidates)

        assert result1.best_answer in ["Alpha", "Beta"]
        assert result1.best_answer == result2.best_answer
        assert pytest.approx(result1.confidence, rel=1e-5) == 0.5

    def test_zero_total_reward_handling(self) -> None:
        """When all candidate rewards are 0.0, handles gracefully without ZeroDivisionError."""
        aggregator = RAPAggregation(depth_weighting=False, min_reward=0.0)
        candidates = [
            CandidateTrajectory(answer="Zero1", reward=0.0, depth=1),
            CandidateTrajectory(answer="Zero2", reward=0.0, depth=1),
        ]

        result = aggregator.aggregate(candidates)

        assert result.best_answer in ["Zero1", "Zero2"]
        assert 0.0 <= result.confidence <= 1.0


class TestTreeBasedAggregation:
    """Verifies extracting and aggregating solutions directly from an MCTS search tree."""

    @pytest.fixture
    def tree_with_multiple_solutions(self) -> MCTSNode:
        """Builds an MCTS tree with three terminal nodes:

        Root (N=10)
          ├── NodeA (depth 1)
          │     ├── Terminal 1: state="Answer: 42", reward=10.0, visited=True
          │     └── Terminal 2: state="Answer: 17", reward=4.0, visited=True
          └── NodeB (depth 1)
                └── Terminal 3: state="Answer: 42", reward=8.0, visited=True
        """
        root = MCTSNode(state="Question: What is 6 * 7?", depth=0)
        root.visited = True

        node_a = MCTSNode(state="Step A", depth=1, parent=root)
        node_a.visited = True

        term1 = MCTSNode(
            state="Answer: 42",
            depth=2,
            parent=node_a,
            reward=10.0,
            is_terminal=True,
        )
        term1.visited = True

        term2 = MCTSNode(
            state="Answer: 17",
            depth=2,
            parent=node_a,
            reward=4.0,
            is_terminal=True,
        )
        term2.visited = True

        node_a.children = [term1, term2]

        node_b = MCTSNode(state="Step B", depth=1, parent=root)
        node_b.visited = True

        term3 = MCTSNode(
            state="Answer: 42",
            depth=2,
            parent=node_b,
            reward=8.0,
            is_terminal=True,
        )
        term3.visited = True

        node_b.children = [term3]
        root.children = [node_a, node_b]

        return root

    def test_aggregate_from_tree_success(self, tree_with_multiple_solutions: MCTSNode) -> None:
        """Discovers terminal states, parses answers, and returns consensus result."""
        aggregator = RAPAggregation(depth_weighting=True)

        def extract_answer(node: MCTSNode) -> str:
            # Extracts the numeric answer from state string
            return node.state.replace("Answer: ", "").strip()

        result = aggregator.aggregate_from_tree(
            root=tree_with_multiple_solutions,
            answer_extractor=extract_answer,
        )

        # "42" has term1 (weight = 10/2 = 5) and term3 (weight = 8/2 = 4) => total 9.0
        # "17" has term2 (weight = 4/2 = 2) => total 2.0
        # Total weight: 11.0
        assert result.best_answer == "42"
        assert pytest.approx(result.votes["42"], rel=1e-5) == 9.0
        assert pytest.approx(result.votes["17"], rel=1e-5) == 2.0
        assert pytest.approx(result.confidence, rel=1e-5) == 9.0 / 11.0

    def test_aggregate_from_tree_no_terminals(self) -> None:
        """Tree without any visited terminal nodes handles safely."""
        root = MCTSNode(state="Root", depth=0)
        aggregator = RAPAggregation()

        try:
            result = aggregator.aggregate_from_tree(root=root)
            assert result.best_answer is None or result.candidates_count == 0
        except ValueError:
            pass  # Raising ValueError when tree has no terminals is also acceptable
