"""Progress reporter and visualizer for tau-game CLI and extension."""

from __future__ import annotations

import re
from typing import Any, Callable

from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent.memory import TurnRecord
from env.environment import BaseEnvironment, Frame

SYMBOL_DESCRIPTIONS: dict[str, str] = {
    "B": "[bold cyan]B[/bold cyan]: Player",
    "G": "[bold green]G[/bold green]: Goal",
    "Y": "[bold yellow]Y[/bold yellow]: Key",
    "R": "[bold red]R[/bold red]: Red Key",
    "O": "[bold magenta]O[/bold magenta]: Door",
    "C": "[bold blue]C[/bold blue]: Push Block",
    "M": "[bold purple]M[/bold purple]: Target Pad",
    "W": "[bold white]W[/bold white]: Portal / Gate",
    "#": "[dim]#[/dim]: Wall",
    ".": "[dim].[/dim]: Floor",
}


def extract_reasoning_text(response: str) -> str:
    """Extract natural language reasoning from LLM response, stripping code blocks & world model."""
    text = re.sub(r"```(?:python)?.*?```", "", response, flags=re.DOTALL)
    text = re.sub(r"<\/?think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:World model|Discovered mechanics|Hypotheses|Findings)\s*:.*?(?=\n\n[A-Z]|\Z)", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned_lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Keep up to 5 lines of concise reasoning
    if len(cleaned_lines) > 5:
        cleaned_lines = cleaned_lines[:5] + ["..."]
    return "\n".join(cleaned_lines).strip() or "(Observing environment)"


def render_board_with_legend(frame: Frame) -> Table:
    """Render ASCII board side-by-side with legend and current status."""
    from perception.segmentation import COLOR_TO_CHAR

    table = Table(box=None, show_header=False, padding=(0, 2))
    table.add_column("Board", style="bold")
    table.add_column("Details", style="dim")

    # Board ASCII
    board_ascii = frame.ascii

    # Status details
    details: list[str] = [
        f"[bold]Level:[/bold] {frame.level}",
        f"[bold]Step:[/bold] {frame.step}",
        f"[bold]Shape:[/bold] {frame.shape[0]}x{frame.shape[1]}",
    ]

    info = frame.info or {}
    if "player" in info:
        details.append(f"[bold]Player Pos:[/bold] {info['player']}")
    if "has_key" in info and (info["has_key"] or any(4 in row for row in frame.grid)):
        key_status = "[bold green]YES 🔑[/bold green]" if info["has_key"] else "[dim]NO[/dim]"
        details.append(f"[bold]Yellow Key:[/bold] {key_status}")
    if "has_red_key" in info and (info["has_red_key"] or any(2 in row for row in frame.grid)):
        red_key_status = "[bold red]YES 🔑[/bold red]" if info["has_red_key"] else "[dim]NO[/dim]"
        details.append(f"[bold]Red Key:[/bold] {red_key_status}")
    if "space_toggle" in info and frame.level in (3, 5, 7):
        toggle_status = "[bold magenta]ACTIVE[/bold magenta]" if info["space_toggle"] else "[dim]OFF[/dim]"
        details.append(f"[bold]Space Phase:[/bold] {toggle_status}")

    # Dynamically show only symbols present on the current board
    present_values = {val for row in frame.grid for val in row}
    present_symbols = [COLOR_TO_CHAR.get(v, str(v)) for v in sorted(present_values)]
    active_legends = [SYMBOL_DESCRIPTIONS[s] for s in present_symbols if s in SYMBOL_DESCRIPTIONS]

    details.append("\n[bold]Legend (Current Board):[/bold]")
    details.append("  ".join(active_legends))

    table.add_row(board_ascii, "\n".join(details))
    return table


def build_turn_panel(
    turn_idx: int,
    turn_record: TurnRecord,
    env: BaseEnvironment,
    max_turns: int = 30,
) -> Panel:
    """Build a Rich Panel summarizing the turn progress, reasoning, code, and board state."""
    curr_frame = env.current_frame
    reasoning = extract_reasoning_text(turn_record.response)

    content_parts: list[Any] = []

    # 1. Board state & details table
    content_parts.append(render_board_with_legend(curr_frame))
    content_parts.append("")

    # 2. Agent Reasoning
    content_parts.append("[bold cyan]🧠 Reasoning & Hypothesis:[/bold cyan]")
    content_parts.append(f"[italic]{reasoning}[/italic]")
    content_parts.append("")

    # 3. Python Code Executed (if any)
    if turn_record.code_executed:
        content_parts.append("[bold green]💻 Python REPL Code:[/bold green]")
        content_parts.append(f"[dim]{turn_record.code_executed.strip()}[/dim]")
        content_parts.append("")

    # 4. Actions taken and REPL output
    if turn_record.actions_taken:
        action_badges = " ".join(f"[bold yellow]▶ {act}[/bold yellow]" for act in turn_record.actions_taken)
        content_parts.append(f"[bold]Actions Executed:[/bold] {action_badges}")
    else:
        content_parts.append("[dim]Actions Executed: (none - inspect only)[/dim]")

    if turn_record.repl_output and turn_record.repl_output.strip() and turn_record.repl_output.strip() != "(No output / actions)":
        content_parts.append(f"[dim yellow]REPL Output: {turn_record.repl_output.strip()}[/dim yellow]")

    # 5. Events / Milestone notices
    info = curr_frame.info or {}
    if info.get("has_key") and turn_record.previous_info and not turn_record.previous_info.get("has_key"):
        content_parts.append("\n[bold green]🔑 KEY COLLECTED! Door can now be opened.[/bold green]")
    if info.get("hit_wall"):
        content_parts.append("\n[bold red]⚠️ Hit a wall obstacle![/bold red]")
    if env.is_done():
        content_parts.append("\n[bold green]🏆 LEVEL GOAL REACHED! PUZZLE SOLVED![/bold green]")

    title = f"[bold magenta]🎮 Turn {turn_idx}/{max_turns} (Step {curr_frame.step})[/bold magenta]"
    border_style = "green" if env.is_done() else "cyan"

    # Combine text
    table_container = Table.grid(padding=(0, 0))
    table_container.add_column()
    for item in content_parts:
        table_container.add_row(item)

    return Panel(
        table_container,
        title=title,
        border_style=border_style,
        box=ROUNDED,
        padding=(1, 2),
    )


def print_game_header(game_name: str, level: int, print_fn: Callable[[Any], None]) -> None:
    """Print an attractive intro header when a game session begins."""
    intro_table = Table.grid(padding=(0, 1))
    intro_table.add_column(style="bold")
    intro_table.add_column()

    intro_table.add_row("Game:", f"[bold cyan]{game_name.upper()}[/bold cyan] (Level {level})")
    intro_table.add_row("Harness:", "[bold yellow]Duck Harness (Python REPL + World Model)[/bold yellow]")
    intro_table.add_row("Goal:", "Agent writes Python code to discover rules, inspect objects, and reach the exit.")

    panel = Panel(
        intro_table,
        title="[bold green]🎮 TAU-GAME: Autonomous Game Agent[/bold green]",
        border_style="green",
        box=ROUNDED,
        padding=(1, 2),
    )
    print_fn(panel)
