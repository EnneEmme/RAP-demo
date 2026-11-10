"""Command Line Interface for RAP Demonstration.

Provides CLI entry points for:
- `rap run`: terminal-based MCTS search execution with Rich visualization.
- `rap ui`: real-time web visualizer powered by FastAPI and Uvicorn.
- `rap compare`: side-by-side comparison of Hybrid vs Paper reasoning modes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from typing import Any, Sequence
from rich.console import Console
from rich.table import Table

from rap_demo.core.mcts import MCTS
from rap_demo.core.node import MCTSNode
from rap_demo.domains.blocksworld.domain import BlocksworldDomain
from rap_demo.domains.blocksworld.models import BlockState
from rap_demo.llm.base import BaseLLM, LLMError
from rap_demo.llm.factory import get_llm
from rap_demo.llm.litellm_client import _load_dotenv_file
from rap_demo.llm.mock import MockLLM
from rap_demo.visualizer.console import RichMCTSConsole
import rap_demo.visualizer.web.app
from rap_demo.visualizer.web.controller import SearchController

SUPPORTED_DOMAINS: set[str] = {"blocksworld"}

DEFAULT_INITIAL_STATE: BlockState = BlockState(
    clear={"red", "blue", "orange"},
    on_table={"yellow", "red", "blue"},
    on={"orange": "yellow"},
    holding=None,
)

DEFAULT_GOAL: list[str] = [
    "the red block is on top of the blue block",
    "the orange block is on top of the red block",
]


def _create_llm(
    backend: str,
    model: str | None = None,
    api_base: str | None = None,
    tpm: int | None = None,
    rpm: int | None = None,
) -> BaseLLM:
    """Instantiates and configures LLM backend for planning.

    Args:
        backend: Backend identifier ('mock', 'llama-server', or 'litellm').
        model: Optional model identifier forwarded to non-mock backends.
        api_base: Optional custom API base URL forwarded when provided.
        tpm: Optional tokens-per-minute rate limit for LiteLLM backend.
        rpm: Optional requests-per-minute rate limit for LiteLLM backend.

    Returns:
        Configured BaseLLM instance.
    """
    if backend == "mock":
        blocks = ["red", "blue", "orange", "yellow"]
        rules: list[tuple[str | re.Pattern[str], str]] = []

        # 1. World model transitions for paper mode
        for b in blocks:
            rules.append((
                f"Pick up the {b} block.",
                f"[CHANGE] the hand was empty and now is holding the {b} block, "
                f"the {b} block was on the table and now is in the hand, "
                f"and the {b} block is no longer clear.",
            ))
            rules.append((
                f"Put down the {b} block.",
                f"[CHANGE] hand is empty, the {b} block was in the hand and now is on the table, "
                f"and the {b} block is now clear.",
            ))
            for t in blocks:
                if b != t:
                    rules.append((
                        f"Stack the {b} block on top of the {t} block.",
                        f"[CHANGE] hand is empty, the {b} block was in the hand and now is on top of the {t} block, "
                        f"the {b} block is now clear, and the {t} block is no longer clear.",
                    ))
                    rules.append((
                        f"Unstack the {b} block from on top of the {t} block.",
                        f"[CHANGE] the hand was empty and now is holding the {b} block, "
                        f"the {b} block was on top of the {t} block and now is in the hand, "
                        f"the {b} block is no longer clear, and the {t} block is now clear.",
                    ))

        # 2. Hybrid action proposals
        rules.append((
            re.compile(r"holding the red block"),
            json.dumps({
                "thought": "Stack red onto blue",
                "action": "stack",
                "block": "red",
                "target": "blue",
            }),
        ))
        rules.append((
            re.compile(r"holding the orange block"),
            json.dumps({
                "thought": "Stack orange onto red",
                "action": "stack",
                "block": "orange",
                "target": "red",
            }),
        ))
        rules.append((
            re.compile(r"red block is on top of the blue block"),
            json.dumps({
                "thought": "Unstack orange from yellow",
                "action": "unstack",
                "block": "orange",
                "target": "yellow",
            }),
        ))
        rules.append((
            re.compile(r".*"),
            json.dumps({
                "thought": "Pick up red from table",
                "action": "pick_up",
                "block": "red",
                "target": None,
            }),
        ))

        default_resp = (
            "[CHANGE] the hand was empty and now is holding the red block, "
            "the red block was on the table and now is in the hand, "
            "and the red block is no longer clear."
        )
        return get_llm(backend, default_response=default_resp, response_rules=rules)

    _load_dotenv_file()

    normalized = backend.strip().lower().replace("-", "_")
    kwargs: dict[str, Any] = {}
    if model is not None:
        kwargs["model"] = model
    if api_base is not None:
        if normalized in ("llama", "llama_server"):
            kwargs["base_url"] = api_base
        else:
            kwargs["api_base"] = api_base
    if normalized in ("litellm", "lite_llm"):
        if tpm is not None:
            kwargs["tokens_per_minute"] = tpm
        if rpm is not None:
            kwargs["requests_per_minute"] = rpm
    return get_llm(backend, **kwargs)


def _create_llm_from_args(args: argparse.Namespace) -> BaseLLM:
    """Create a configured LLM backend from parsed CLI arguments."""
    return _create_llm(
        args.backend,
        model=getattr(args, "model", None),
        api_base=getattr(args, "api_base", None),
        tpm=getattr(args, "tpm", None),
        rpm=getattr(args, "rpm", None),
    )


def _add_llm_arguments(subparser: argparse.ArgumentParser) -> None:
    """Register shared LLM backend selection arguments on a CLI subparser."""
    subparser.add_argument(
        "--backend",
        choices=["mock", "llama-server", "litellm"],
        default="mock",
        help="LLM backend identifier (choices: mock, llama-server, litellm; default: mock)",
    )
    subparser.add_argument(
        "--model",
        default=None,
        help="Model identifier for external/local LLM backends (e.g. gpt-4o-mini, anthropic/claude-3-5-sonnet-20241022)",
    )
    subparser.add_argument(
        "--api-base",
        default=None,
        help="Optional custom API base URL for LLM backend requests",
    )
    subparser.add_argument(
        "--tpm",
        type=int,
        default=None,
        help="Max tokens per minute (TPM) rate limit",
    )
    subparser.add_argument(
        "--rpm",
        type=int,
        default=None,
        help="Max requests per minute (RPM) rate limit",
    )


def _print_backend_error_tip(err_console: Console, backend: str) -> None:
    """Print a backend-specific troubleshooting tip to stderr."""
    normalized = backend.strip().lower().replace("-", "_")
    if normalized in ("litellm", "lite_llm"):
        err_console.print(
            "\n[yellow]Tip:[/yellow] Make sure the appropriate API key environment variable is set "
            "(or defined in your [bold].env[/bold] file, e.g., [bold]OPENAI_API_KEY[/bold], "
            "[bold]ANTHROPIC_API_KEY[/bold], [bold]GEMINI_API_KEY[/bold]) and verify "
            "[bold]--model[/bold] / [bold]--api-base[/bold].\n"
        )
    elif normalized in ("llama", "llama_server"):
        err_console.print(
            "\n[yellow]Tip:[/yellow] Make sure llama-server is running with your Hugging Face model:"
        )
        err_console.print(
            "[bold green]llama-server -hf saidutta69/LFM3-VL-2B-heretic --port 8080 -ngl 99[/bold green]\n"
        )


def build_parser() -> argparse.ArgumentParser:
    """Builds and returns the top-level argument parser for the rap CLI.

    Configures subcommands:
    - `run`: Terminal-based MCTS search execution.
    - `ui`: Real-time web visualizer server.
    - `compare`: Side-by-side mode comparison.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="rap",
        description="Reasoning via Planning (RAP) Reimplementation with Real-Time Visualization",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. `rap run`
    run_parser = subparsers.add_parser("run", help="Run MCTS search in terminal")
    run_parser.add_argument(
        "--domain",
        default="blocksworld",
        help="Domain identifier (default: blocksworld)",
    )
    run_parser.add_argument(
        "--mode",
        choices=["hybrid", "paper"],
        default="hybrid",
        help="Reasoning mode (choices: hybrid, paper; default: hybrid)",
    )
    _add_llm_arguments(run_parser)
    run_parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="MCTS search iterations (default: 10)",
    )
    run_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print detailed tree structure in terminal",
    )

    # 2. `rap ui`
    ui_parser = subparsers.add_parser("ui", help="Launch real-time web visualizer server")
    ui_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address to bind server (default: 127.0.0.1)",
    )
    ui_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind server (default: 8000)",
    )
    ui_parser.add_argument(
        "--domain",
        default="blocksworld",
        help="Domain identifier (default: blocksworld)",
    )
    ui_parser.add_argument(
        "--mode",
        choices=["hybrid", "paper"],
        default="hybrid",
        help="Reasoning mode (choices: hybrid, paper; default: hybrid)",
    )
    _add_llm_arguments(ui_parser)
    ui_parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Maximum MCTS iterations (default: 50)",
    )

    # 3. `rap compare`
    compare_parser = subparsers.add_parser("compare", help="Side-by-side mode comparison")
    compare_parser.add_argument(
        "--domain",
        default="blocksworld",
        help="Domain identifier (default: blocksworld)",
    )
    _add_llm_arguments(compare_parser)
    compare_parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="MCTS search iterations per mode (default: 10)",
    )
    compare_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print detailed search trees for both modes",
    )

    return parser


def cmd_run(args: argparse.Namespace) -> int:
    """Executes terminal MCTS search according to parsed CLI arguments.

    Args:
        args: Parsed arguments namespace.

    Returns:
        Integer exit status code (0 for success, non-zero for failure).
    """
    console = Console()
    err_console = Console(stderr=True)

    if args.domain.lower() not in SUPPORTED_DOMAINS:
        err_console.print(
            f"[bold red]Error:[/bold red] Unsupported domain '{args.domain}'. "
            f"Supported domains: {', '.join(sorted(SUPPORTED_DOMAINS))}."
        )
        return 1

    if args.iterations <= 0:
        err_console.print(
            f"[bold red]Error:[/bold red] Iterations must be positive, got {args.iterations}."
        )
        return 1

    try:
        llm = _create_llm_from_args(args)
        domain = BlocksworldDomain(
            initial_state=DEFAULT_INITIAL_STATE,
            goal=DEFAULT_GOAL,
            llm=llm,
            mode=args.mode,
        )
        mcts = MCTS(
            world_model=domain.create_world_model(),
            reasoning_agent=domain.create_agent(),
            max_depth=6,
        )

        root = mcts.search(domain.get_initial_state(), num_iterations=args.iterations)

        visualizer = RichMCTSConsole(console=console)
        if args.verbose:
            visualizer.print_tree(root)

        best_path = mcts.get_best_trajectory(root)
        best_reward = best_path[-1].reward if best_path else 0.0
        metrics: dict[str, Any] = {
            "Iterations": args.iterations,
            "Best Reward": f"{best_reward:.2f}",
            "Trajectory Steps": max(0, len(best_path) - 1),
        }
        if hasattr(llm, "call_count"):
            metrics["LLM Calls"] = llm.call_count

        visualizer.print_summary(best_path, metrics=metrics)
        return 0
    except (ConnectionError, ValueError, LLMError) as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        _print_backend_error_tip(err_console, str(args.backend))
        return 1


def cmd_ui(args: argparse.Namespace) -> int:
    """Prepares and launches the real-time web visualizer server.

    Args:
        args: Parsed arguments namespace.

    Returns:
        Integer exit status code (0 for success, non-zero for failure).
    """
    console = Console()
    err_console = Console(stderr=True)

    if args.domain.lower() not in SUPPORTED_DOMAINS:
        err_console.print(
            f"[bold red]Error:[/bold red] Unsupported domain '{args.domain}'. "
            f"Supported domains: {', '.join(sorted(SUPPORTED_DOMAINS))}."
        )
        return 1

    if args.port < 1 or args.port > 65535:
        err_console.print(
            f"[bold red]Error:[/bold red] Port must be between 1 and 65535, got {args.port}."
        )
        return 1

    if args.iterations <= 0:
        err_console.print(
            f"[bold red]Error:[/bold red] Iterations must be positive, got {args.iterations}."
        )
        return 1

    try:
        llm = _create_llm_from_args(args)
        domain = BlocksworldDomain(
            initial_state=DEFAULT_INITIAL_STATE,
            goal=DEFAULT_GOAL,
            llm=llm,
            mode=args.mode,
        )
        mcts = MCTS(
            world_model=domain.create_world_model(),
            reasoning_agent=domain.create_agent(),
            max_depth=6,
        )
        controller = SearchController(
            mcts=mcts,
            initial_state=domain.get_initial_state(),
            max_iterations=args.iterations,
            goal=DEFAULT_GOAL,
            mode=args.mode,
        )

        create_app_fn = sys.modules["rap_demo.visualizer.web.app"].create_app
        app = create_app_fn(controller)
        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port)
        return 0
    except OSError as exc:
        err_console.print(f"[bold red]Server Bind Error:[/bold red] {exc}")
        return 1
    except (ConnectionError, ValueError, LLMError) as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        _print_backend_error_tip(err_console, str(args.backend))
        return 1


def cmd_compare(args: argparse.Namespace) -> int:
    """Executes side-by-side comparison of Hybrid vs Paper reasoning modes.

    Args:
        args: Parsed arguments namespace.

    Returns:
        Integer exit status code (0 for success, non-zero for failure).
    """
    console = Console()
    err_console = Console(stderr=True)

    if args.domain.lower() not in SUPPORTED_DOMAINS:
        err_console.print(
            f"[bold red]Error:[/bold red] Unsupported domain '{args.domain}'. "
            f"Supported domains: {', '.join(sorted(SUPPORTED_DOMAINS))}."
        )
        return 1

    if args.iterations <= 0:
        err_console.print(
            f"[bold red]Error:[/bold red] Iterations must be positive, got {args.iterations}."
        )
        return 1

    try:
        # 1. Execute Hybrid Mode
        llm_hybrid = _create_llm_from_args(args)
        domain_hybrid = BlocksworldDomain(
            initial_state=DEFAULT_INITIAL_STATE,
            goal=DEFAULT_GOAL,
            llm=llm_hybrid,
            mode="hybrid",
        )
        mcts_hybrid = MCTS(
            world_model=domain_hybrid.create_world_model(),
            reasoning_agent=domain_hybrid.create_agent(),
            max_depth=6,
        )
        t0_hybrid = time.perf_counter()
        root_hybrid = mcts_hybrid.search(domain_hybrid.get_initial_state(), num_iterations=args.iterations)
        time_hybrid = time.perf_counter() - t0_hybrid
        best_path_hybrid = mcts_hybrid.get_best_trajectory(root_hybrid)
        reward_hybrid = best_path_hybrid[-1].reward if best_path_hybrid else 0.0
        calls_hybrid = getattr(llm_hybrid, "call_count", 0)

        # 2. Execute Paper Mode
        llm_paper = _create_llm_from_args(args)
        domain_paper = BlocksworldDomain(
            initial_state=DEFAULT_INITIAL_STATE,
            goal=DEFAULT_GOAL,
            llm=llm_paper,
            mode="paper",
        )
        mcts_paper = MCTS(
            world_model=domain_paper.create_world_model(),
            reasoning_agent=domain_paper.create_agent(),
            max_depth=6,
        )
        t0_paper = time.perf_counter()
        root_paper = mcts_paper.search(domain_paper.get_initial_state(), num_iterations=args.iterations)
        time_paper = time.perf_counter() - t0_paper
        best_path_paper = mcts_paper.get_best_trajectory(root_paper)
        reward_paper = best_path_paper[-1].reward if best_path_paper else 0.0
        calls_paper = getattr(llm_paper, "call_count", 0)

        visualizer = RichMCTSConsole(console=console)
        if args.verbose:
            console.print("\n[bold cyan]=== Hybrid Mode Reasoning Tree ===[/bold cyan]")
            visualizer.print_tree(root_hybrid)
            console.print("\n[bold cyan]=== Paper Mode Reasoning Tree ===[/bold cyan]")
            visualizer.print_tree(root_paper)

        # 3. Format Comparative Summary Table
        table = Table(
            title="Reasoning Mode Comparison: Hybrid vs Paper",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Hybrid Mode", style="green", justify="right")
        table.add_column("Paper Mode", style="yellow", justify="right")

        table.add_row("Final reward / score", f"{reward_hybrid:.2f}", f"{reward_paper:.2f}")
        table.add_row("Trajectory Steps", str(max(0, len(best_path_hybrid) - 1)), str(max(0, len(best_path_paper) - 1)))
        table.add_row("MCTS Iterations / Visits", str(args.iterations), str(args.iterations))
        table.add_row("LLM calls count", str(calls_hybrid), str(calls_paper))
        table.add_row("Execution Time (s)", f"{time_hybrid:.4f}", f"{time_paper:.4f}")
        table.add_row("Thoughts Preserved", "Yes (Structured JSON)", "No (Raw Text)")

        console.print(table)
        return 0
    except (ConnectionError, ValueError, LLMError, RuntimeError) as exc:
        err_console.print(f"[bold red]Comparison Error:[/bold red] {exc}")
        if isinstance(exc, (ConnectionError, LLMError)):
            _print_backend_error_tip(err_console, str(args.backend))
        return 1


def main(args: Sequence[str] | None = None) -> int:
    """Main CLI entrypoint for the `rap` command.

    Args:
        args: Optional list of command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Integer exit status code (0 for success, non-zero for failure).
    """
    parser = build_parser()
    argv = list(args) if args is not None else sys.argv[1:]

    if not argv:
        parser.print_help(file=sys.stderr)
        return 1

    try:
        parsed_args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except Exception as exc:
        console = Console(stderr=True)
        console.print(f"[bold red]Parsing Error:[/bold red] {exc}")
        return 2

    command = getattr(parsed_args, "command", None)
    if command is None:
        parser.print_help(file=sys.stderr)
        return 1

    try:
        _load_dotenv_file()
        if command == "run":
            return cmd_run(parsed_args)
        elif command == "ui":
            return cmd_ui(parsed_args)
        elif command == "compare":
            return cmd_compare(parsed_args)
        else:
            parser.print_help(file=sys.stderr)
            return 1
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except Exception as exc:
        console = Console(stderr=True)
        console.print(f"[bold red]Error:[/bold red] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
