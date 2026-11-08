"""Rich terminal visualizer for MCTS trees and planning progress."""

from __future__ import annotations

import math
from typing import Any, Optional
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from rap_demo.core.node import MCTSNode


class RichMCTSConsole:
    """Terminal visualizer for MCTS tree and search metrics using Rich."""

    def __init__(self, console: Optional[Console] = None) -> None:
        self.console: Console = console or Console()

    def render_tree(self, root: MCTSNode, max_depth: Optional[int] = None) -> Tree:
        """Renders MCTS tree hierarchy as a rich.tree.Tree instance.

        Args:
            root: Root node of the MCTS tree.
            max_depth: Maximum tree depth to render (truncates deeper nodes).

        Returns:
            Rich Tree component ready for rendering.
        """
        root_label = (
            f"[bold cyan]Root[/bold cyan] (N={root.N}, Q={root.Q:.2f}) "
            f"[dim]State: {root.state}[/dim]"
        )
        if root.is_terminal:
            root_label += " [bold magenta][TERMINAL][/bold magenta]"

        tree = Tree(root_label)

        def _add_children(tree_node: Tree, node: MCTSNode) -> None:
            if max_depth is not None and node.depth >= max_depth:
                return

            for child in node.children:
                action_str = str(child.action) if child.action is not None else "Root"
                label = (
                    f"[bold green]{action_str}[/bold green] "
                    f"(N={child.N}, Q={child.Q:.2f}, reward={child.reward:.1f})"
                )
                if child.is_terminal:
                    label += " [bold magenta][TERMINAL][/bold magenta]"

                child_tree = tree_node.add(label)
                if max_depth is None or child.depth < max_depth:
                    _add_children(child_tree, child)

        _add_children(tree, root)
        return tree

    def print_tree(self, root: MCTSNode, max_depth: Optional[int] = None) -> None:
        """Prints formatted MCTS tree directly to the rich console."""
        tree = self.render_tree(root, max_depth=max_depth)
        self.console.print(tree)

    def render_summary(
        self, best_path: list[MCTSNode], metrics: Optional[dict[str, Any]] = None
    ) -> Table:
        """Renders summary table of best trajectory and search metrics.

        Args:
            best_path: Path of nodes representing the best discovered trajectory.
            metrics: Optional dictionary of search metrics (e.g., iterations, best_reward).

        Returns:
            Rich Table displaying trajectory steps and metrics.
        """
        table = Table(title="MCTS Trajectory Summary", show_header=True, header_style="bold magenta")
        table.add_column("Step / Depth", style="cyan", justify="center")
        table.add_column("Action", style="green")
        table.add_column("Reward", style="yellow", justify="right")
        table.add_column("Visits (N)", style="blue", justify="right")
        table.add_column("Q-Value", style="blue", justify="right")
        table.add_column("State", style="dim")

        for node in best_path:
            action_desc = str(node.action) if node.action is not None else "[Initial State]"
            state_desc = str(node.state)
            if len(state_desc) > 40:
                state_desc = state_desc[:37] + "..."

            table.add_row(
                str(node.depth),
                action_desc,
                f"{node.reward:.2f}",
                str(node.N),
                f"{node.Q:.2f}",
                state_desc,
            )

        if metrics:
            caption_items = [f"{k}={v}" for k, v in metrics.items()]
            table.caption = f"[dim]Metrics: {', '.join(caption_items)}[/dim]"

        return table

    def print_summary(
        self, best_path: list[MCTSNode], metrics: Optional[dict[str, Any]] = None
    ) -> None:
        """Prints summary table of best trajectory and search metrics to console."""
        table = self.render_summary(best_path, metrics=metrics)
        self.console.print(table)
        if metrics:
            for key, val in metrics.items():
                self.console.print(f"[bold]{key}:[/bold] {val}")
