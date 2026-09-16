"""Main entry point and CLI runner for tau-game (Light-Duck Harness)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add package root to sys.path
_pkg_root = Path(__file__).resolve().parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from agent.agent import DuckAgent
from env.game import GridWorld, MiniArcGame
from llm.client import LLMClient

console = Console()


def create_mock_gridworld_solver():
    """Mock LLM response generator demonstrating Duck reasoning on GridWorld."""
    turn = [0]

    def mock_llm(messages: list[dict[str, str]]) -> str:
        turn[0] += 1
        t = turn[0]
        if t == 1:
            return (
                "Let's inspect the board to locate the player and the goal.\n"
                "World model:\n"
                "- 'B' represents the player at (1, 1).\n"
                "- 'G' is the target goal at (4, 4).\n"
                "- '#' are obstacle walls blocking the path.\n\n"
                "```python\n"
                "print('Board shape:', current_frame.shape)\n"
                "print('Valid actions:', valid_actions)\n"
                "action('DOWN')\n"
                "```"
            )
        elif t == 2:
            return (
                "We moved DOWN to (2, 1). Let's move DOWN again to bypass the wall at (2, 2).\n"
                "World model:\n"
                "- Walls are stationary obstacles.\n"
                "- Moving DOWN toward row 4 is clear.\n\n"
                "```python\n"
                "action(['DOWN', 'DOWN'])\n"
                "```"
            )
        elif t == 3:
            return (
                "Now at row 4. The goal G is at (4, 4). We can move RIGHT toward it.\n"
                "World model:\n"
                "- Player reached bottom corridor, moving toward G.\n\n"
                "```python\n"
                "action(['RIGHT', 'RIGHT', 'RIGHT'])\n"
                "```"
            )
        else:
            return "```python\nprint('Done!')\n```"

    return mock_llm


def create_mock_mcts_solver():
    """Mock LLM policy prior generator for MCTS tree search demonstration."""
    def mock_llm(messages: list[dict[str, str]]) -> str:
        return (
            '{"actions": [{"action": "DOWN", "prior": 0.7, "reason": "Move down along open corridor"}, '
            '{"action": "RIGHT", "prior": 0.3, "reason": "Lateral maneuver toward goal"}], '
            '"rationale": "Prioritize downward progress to avoid obstacles"}'
        )
    return mock_llm


@click.command()
@click.option("--game", type=click.Choice(["gridworld", "mini-arc"]), default="gridworld", help="Game to play.")
@click.option("--level", type=int, default=1, help="Level for multi-level games (mini-arc 1-7).")
@click.option("--provider", type=str, default=None, help="Tau LLM provider (openai, ollama, mlx, etc.).")
@click.option("--model", type=str, default=None, help="LLM model name.")
@click.option("--base-url", type=str, default=None, help="Custom OpenAI-compatible API base URL.")
@click.option("--max-turns", type=int, default=25, help="Maximum agent turns.")
@click.option("--max-tokens", type=int, default=512, help="Max tokens per LLM completion (keeps turns concise).")
@click.option("--demo", is_flag=True, help="Run an automated mock demo showing Duck Harness reasoning.")
@click.option("--show-thinking", is_flag=True, default=False, help="Stream live model thinking/reasoning process to console.")
@click.option("--mcts", type=int, default=0, help="MCTS Mode: 0=Off, 1=Fast Code-driven MCTS (0.02s, Cách 1), 2=LLM-Guided MCTS (Cách 2).")
@click.option("--mcts-sims", type=int, default=None, help="Number of MCTS simulations per turn (default: 50 for mode 1, 8 for mode 2).")
@click.option("--save-trace", type=click.Path(), default=None, help="Save execution trace to a JSON file.")
def main(
    game: str,
    level: int,
    provider: str | None,
    model: str | None,
    base_url: str | None,
    max_turns: int,
    max_tokens: int,
    demo: bool,
    show_thinking: bool,
    mcts: int,
    mcts_sims: int | None,
    save_trace: str | None,
) -> None:
    """Run the Duck Harness autonomous game agent."""
    console.print(
        Panel.fit(
            "[bold cyan]tau-game: Light-Duck Harness[/bold cyan]\n"
            "[dim]Autonomous Game-Solving Agent via Python REPL & World Modeling[/dim]",
            border_style="cyan",
        )
    )

    # 1. Initialize environment
    if game == "gridworld":
        env = GridWorld()
    else:
        env = MiniArcGame(level=level)

    from agent.progress import build_turn_panel, print_game_header

    print_game_header(game, level, console.print)

    # 2. Initialize LLM Client
    if demo:
        if mcts == 1:
            console.print("[yellow]Running in DEMO mode with Fast Code-driven MCTS (Cách 1, 0.02s).[/yellow]\n")
            llm = LLMClient(mock_fn=lambda msgs: "action('DOWN')", max_tokens=max_tokens)
        elif mcts == 2:
            console.print("[yellow]Running in DEMO mode with LLM-Guided MCTS policy generator (Cách 2).[/yellow]\n")
            llm = LLMClient(mock_fn=create_mock_mcts_solver(), max_tokens=max_tokens)
        else:
            console.print("[yellow]Running in DEMO mode with mock LLM reasoner.[/yellow]\n")
            llm = LLMClient(mock_fn=create_mock_gridworld_solver(), max_tokens=max_tokens)
    else:
        thinking_mode = "[bold magenta]ON[/bold magenta] (streaming live)" if show_thinking else "[dim]OFF (spinner only)[/dim]"
        console.print(
            f"[green]LLM Client:[/green] provider={provider or 'tau default'}, "
            f"model={model or 'tau default'}, max_tokens={max_tokens}, "
            f"show_thinking={thinking_mode}\n"
        )
        llm = LLMClient(provider=provider, model=model, base_url=base_url, max_tokens=max_tokens)

    # 3. Create DuckAgent with live streaming and status callbacks
    thinking_state = {"started": False, "turn": 0}
    token_counts = {"thinking": 0, "visible": 0}
    active_status = None

    def on_turn_start(turn_idx: int, frame):
        nonlocal active_status
        thinking_state["started"] = False
        thinking_state["turn"] = turn_idx
        token_counts["thinking"] = 0
        token_counts["visible"] = 0

        if not show_thinking:
            active_status = console.status(
                f"[bold cyan]Turn {turn_idx}:[/bold cyan] [dim]Observing board...[/dim]"
            )
            active_status.start()

    def on_token(turn_idx: int, delta: str, is_thinking: bool):
        nonlocal active_status
        if is_thinking:
            token_counts["thinking"] += 1
            if show_thinking:
                if not thinking_state["started"]:
                    thinking_state["started"] = True
                    console.print(f"\n[bold magenta]💭 Model Thinking (Turn {turn_idx}):[/bold magenta]")
                console.out(delta, style="dim italic", end="")
                sys.stdout.flush()
            else:
                if active_status and token_counts["thinking"] % 6 == 1:
                    active_status.update(
                        f"[bold cyan]Turn {turn_idx}:[/bold cyan] [magenta]💭 Thinking (~{token_counts['thinking']} tokens)...[/magenta]"
                    )
        else:
            token_counts["visible"] += 1
            if thinking_state["started"]:
                thinking_state["started"] = False
                if show_thinking:
                    console.print("\n")
            if not show_thinking and active_status and token_counts["visible"] % 6 == 1:
                active_status.update(
                    f"[bold cyan]Turn {turn_idx}:[/bold cyan] [green]💻 Writing code (~{token_counts['visible']} tokens)...[/green]"
                )

    def on_step(turn_idx: int, turn_record, curr_env):
        nonlocal active_status
        if active_status:
            active_status.stop()
            active_status = None
        if thinking_state["started"]:
            thinking_state["started"] = False
            if show_thinking:
                console.print("\n")
        panel = build_turn_panel(turn_idx, turn_record, curr_env, max_turns=max_turns)
        console.print(panel)

    agent = DuckAgent(
        env=env,
        llm=llm,
        max_turns=max_turns,
        verbose=False,
        on_step_callback=on_step,
        on_turn_start=on_turn_start,
        on_token=on_token,
        use_mcts=mcts,
        mcts_sims=mcts_sims,
    )

    # 4. Run loop
    result = agent.run()

    # 5. Display summary
    console.print("\n" + "=" * 50)
    if result.solved:
        console.print(f"[bold green]✓ PUZZLE SOLVED in {result.total_steps} action steps![/bold green]")
    else:
        console.print(f"[bold red]✗ Game ended without solving ({result.total_steps} steps).[/bold red]")

    console.print(f"[bold]Total Reward:[/bold] {result.total_reward}")
    console.print("\n[bold]Discovered World Model:[/bold]")
    console.print(f"[italic]{result.world_model or '(none)'}[/italic]")

    console.print("\n[bold]Final Board:[/bold]")
    console.print(result.final_frame.ascii)

    # 6. Save trace
    if save_trace:
        trace_data = {
            "solved": result.solved,
            "total_steps": result.total_steps,
            "total_reward": result.total_reward,
            "world_model": result.world_model,
            "turns": [
                {
                    "step": t.step,
                    "code": t.code_executed,
                    "output": t.repl_output,
                    "actions": t.actions_taken,
                }
                for t in result.turns
            ],
            "history": [t.to_dict() for t in result.history],
        }
        Path(save_trace).write_text(json.dumps(trace_data, indent=2), encoding="utf-8")
        console.print(f"\n[dim]Trace saved to {save_trace}[/dim]")


if __name__ == "__main__":
    main()
