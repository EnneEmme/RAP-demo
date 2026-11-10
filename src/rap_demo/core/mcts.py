"""Monte Carlo Tree Search (MCTS) engine according to RAP (Reasoning via Planning)."""

from __future__ import annotations
import math
from typing import Any, Optional, Protocol, runtime_checkable
from rap_demo.core.node import MCTSNode
from rap_demo.core.events import EventBus, EventType


@runtime_checkable
class WorldModel(Protocol):
    """Protocol for state transition and environment simulation."""

    def step(self, state: Any, action: Any) -> tuple[Any, float, bool]:
        """Given current state and action, returns (next_state, immediate_reward, is_terminal)."""
        ...


@runtime_checkable
class ReasoningAgent(Protocol):
    """Protocol for generating candidate reasoning steps/actions."""

    def get_actions(self, state: Any) -> list[Any]:
        """Proposes candidate actions from the current state."""
        ...


def extract_best_trajectory(root: MCTSNode, criterion: str = "reward") -> list[MCTSNode]:
    """Extracts the best trajectory from root according to 'reward' or 'visits'.

    Following RAP Paper Section 3.3 and Appendix A:
    - For 'reward': evaluates complete paths to leaves/terminals, selecting the
      path with maximum cumulative reward (and terminal goal achievement).
    - For 'visits': selects the path with maximum exploration / visits.
    """
    if not root.children:
        return [root]

    all_leaves: list[MCTSNode] = []

    def collect_leaves(node: MCTSNode) -> None:
        if not node.children:
            all_leaves.append(node)
        else:
            for child in node.children:
                collect_leaves(child)

    collect_leaves(root)
    if not all_leaves:
        return [root]

    if criterion == "visits":
        best_leaf = max(
            all_leaves,
            key=lambda leaf: (
                sum(n.N for n in leaf.path_from_root),
                leaf.N,
                leaf.reward,
            ),
        )
    else:  # "reward"
        best_leaf = max(
            all_leaves,
            key=lambda leaf: (
                1 if leaf.is_terminal and leaf.reward >= 50.0 else 0,
                sum(n.reward for n in leaf.path_from_root),
                leaf.M if not math.isinf(leaf.M) else -1e9,
                leaf.reward,
                leaf.N,
            ),
        )

    return best_leaf.path_from_root


class MCTS:
    """Core MCTS planning engine implementing RAP Algorithm 1 & Section 3.3."""

    def __init__(
        self,
        w_exp: float = 1.0,
        discount: float = 1.0,
        aggr_reward: str = "sum",
        aggr_child: str = "max",
        max_depth: int = 10,
        world_model: Optional[WorldModel] = None,
        reasoning_agent: Optional[ReasoningAgent] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self.w_exp = w_exp
        self.discount = discount
        self.aggr_reward = aggr_reward
        self.aggr_child = aggr_child
        self.max_depth = max_depth
        self.world_model = world_model
        self.reasoning_agent = reasoning_agent
        self.event_bus = event_bus

    def search(self, root_state: Any, num_iterations: int) -> MCTSNode:
        """Runs MCTS search from the given root state for num_iterations cycles."""
        root = MCTSNode(state=root_state, depth=0)
        for iteration in range(num_iterations):
            self.step_iteration(root)
            if self.event_bus:
                self.event_bus.emit(
                    EventType.ON_ITERATION_COMPLETE,
                    {"iteration": iteration, "root": root},
                )
        return root

    def step_iteration(self, root: MCTSNode) -> list[MCTSNode]:
        """Executes a single MCTS iteration: Select -> Expand -> Simulate -> Backpropagate."""
        path = self.select(root)
        leaf = path[-1]

        rollout_reward = 0.0
        if not leaf.is_terminal:
            if leaf.is_leaf:
                self.expand(leaf)
            _, rollout_reward = self.simulate(leaf)

        self.backpropagate(path, rollout_reward)
        return path

    def select(self, root: MCTSNode) -> list[MCTSNode]:
        """Selects a leaf node starting from root using UCT formula."""
        node = root
        path = [node]

        while not node.is_terminal and not node.is_leaf:
            node = max(node.children, key=lambda c: self.uct_value(node, c))
            path.append(node)
            if node.N == 0:
                break

        if self.event_bus:
            self.event_bus.emit(EventType.ON_SELECT, {"node": node, "path": path})

        return path

    def uct_value(self, parent: MCTSNode, child: MCTSNode) -> float:
        """Computes the UCT score for a child node with respect to its parent."""
        if child.N == 0:
            return math.inf

        if self.aggr_child == "mean":
            exploitation = child.Q
        else:  # "max"
            exploitation = child.M

        exploration = self.w_exp * math.sqrt(math.log(max(parent.N, 1)) / child.N)
        return exploitation + exploration

    def expand(self, node: MCTSNode) -> list[MCTSNode]:
        """Expands the selected leaf node using ReasoningAgent and WorldModel."""
        if node.is_terminal or node.depth >= self.max_depth:
            return []

        if node.children:
            return node.children

        if self.reasoning_agent is None or self.world_model is None:
            return []

        actions = self.reasoning_agent.get_actions(node.state)
        for action in actions:
            next_state, step_reward, is_terminal = self.world_model.step(node.state, action)
            child = MCTSNode(
                state=next_state,
                action=action,
                parent=node,
                depth=node.depth + 1,
                reward=step_reward,
                is_terminal=is_terminal,
                thought=getattr(action, "thought", None),
            )
            node.children.append(child)

        if self.event_bus:
            self.event_bus.emit(EventType.ON_EXPAND, {"node": node, "children": node.children})

        return node.children

    def simulate(self, node: MCTSNode) -> tuple[list[MCTSNode], float]:
        """Performs rollout simulation until terminal state or max depth."""
        curr = node
        sim_path: list[MCTSNode] = [curr]
        rollout_reward = 0.0
        discount_power = 1.0

        while not curr.is_terminal and curr.depth < self.max_depth:
            if self.reasoning_agent is None or self.world_model is None:
                break
            actions = self.reasoning_agent.get_actions(curr.state)
            if not actions:
                break

            best_action = None
            best_next_state = None
            best_reward = -math.inf
            best_is_terminal = False

            for action in actions:
                next_state, reward, is_terminal = self.world_model.step(curr.state, action)
                if reward > best_reward:
                    best_reward = reward
                    best_action = action
                    best_next_state = next_state
                    best_is_terminal = is_terminal

            if best_action is None:
                break

            next_node = MCTSNode(
                state=best_next_state,
                action=best_action,
                parent=curr,
                depth=curr.depth + 1,
                reward=best_reward,
                is_terminal=best_is_terminal,
            )
            sim_path.append(next_node)
            rollout_reward += discount_power * best_reward
            discount_power *= self.discount
            curr = next_node

        if self.event_bus:
            self.event_bus.emit(
                EventType.ON_SIMULATE,
                {"node": node, "sim_path": sim_path, "rollout_reward": rollout_reward},
            )

        return sim_path, rollout_reward

    def backpropagate(self, path: list[MCTSNode], rollout_reward: float = 0.0) -> None:
        """Backpropagates rewards along the selected path back to the root."""
        reward = rollout_reward
        coeff = 0

        for node in reversed(path):
            reward = node.reward + self.discount * reward
            coeff = coeff * self.discount + 1

            if self.aggr_reward == "mean":
                c_reward = reward / coeff
            else:
                c_reward = reward

            node.N += 1
            node.Q += c_reward
            node.M = max(node.M, c_reward)
            node.visited = True

        if self.event_bus:
            self.event_bus.emit(
                EventType.ON_BACKPROP,
                {"path": path, "rollout_reward": rollout_reward},
            )

    def get_best_trajectory(self, root: MCTSNode, criterion: str = "reward") -> list[MCTSNode]:
        """Extracts the best trajectory from root according to 'reward' or 'visits'."""
        return extract_best_trajectory(root, criterion=criterion)
