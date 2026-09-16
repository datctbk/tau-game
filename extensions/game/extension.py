"""tau-game extension — provides game solving agent capabilities to tau."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure package root is in sys.path
_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from tau.core.extension import Extension, ExtensionContext
from tau.core.types import (
    ExtensionManifest,
    SlashCommand,
    ToolDefinition,
    ToolParameter,
)

from agent.agent import DuckAgent
from env.game import GridWorld, MiniArcGame
from llm.client import LLMClient

logger = logging.getLogger(__name__)


class GameExtension(Extension):
    """Tau extension integrating the Duck Harness autonomous game agent."""

    manifest = ExtensionManifest(
        name="game",
        version="0.1.0",
        description=(
            "Autonomous game solving harness inspired by Tufa Labs' Duck Harness for ARC-AGI-3. "
            "Equips LLMs with a Python REPL workspace to explore environments, inspect objects "
            "via 4-connected segmentation, form hypotheses, and take actions."
        ),
        author="tau",
        system_prompt_fragment=(
            "You have a game solving tool (`game_run`) powered by the Duck Harness. "
            "It can autonomously explore grid environments and ARC puzzles using an interactive "
            "Python REPL and world model tracking."
        ),
    )

    def __init__(self) -> None:
        self._ctx: ExtensionContext | None = None
        self._is_running = False
        self._cancel_requested = False

    def on_load(self, context: ExtensionContext) -> None:
        self._ctx = context

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="game_run",
                description="Run the Duck Harness autonomous agent to solve a game environment.",
                parameters={
                    "game": ToolParameter(
                        type="string",
                        description="The game environment to solve: 'gridworld' or 'mini-arc'.",
                        required=False,
                    ),
                    "level": ToolParameter(
                        type="integer",
                        description="Level number for multi-level games (e.g., 1-7 for mini-arc).",
                        required=False,
                    ),
                    "max_turns": ToolParameter(
                        type="integer",
                        description="Maximum agent reasoning turns.",
                        required=False,
                    ),
                },
                handler=self._handle_game_run,
            )
        ]

    def _handle_game_run(
        self,
        game: str = "gridworld",
        level: int = 1,
        max_turns: int = 20,
        mcts: int | bool = 0,
        mcts_sims: int | None = None,
    ) -> str:
        game_name = game.lower().strip()
        level = int(level)
        max_turns = int(max_turns)

        if game_name == "gridworld":
            env = GridWorld()
        else:
            env = MiniArcGame(level=level)

        def on_step(turn_idx: int, turn_record, curr_env):
            if self._ctx:
                from agent.progress import build_turn_panel
                panel = build_turn_panel(turn_idx, turn_record, curr_env, max_turns=max_turns)
                self._ctx.print(panel)

        llm = LLMClient(extension_context=self._ctx)
        agent = DuckAgent(
            env=env,
            llm=llm,
            max_turns=max_turns,
            verbose=False,
            on_step_callback=on_step,
            use_mcts=mcts,
            mcts_sims=mcts_sims,
        )
        result = agent.run()

        return json.dumps(
            {
                "solved": result.solved,
                "steps": result.total_steps,
                "reward": result.total_reward,
                "world_model": result.world_model,
                "final_board_ascii": result.final_frame.ascii,
            },
            indent=2,
        )

    def slash_commands(self) -> list[SlashCommand]:
        return [
            SlashCommand(
                name="game",
                description="Play games via Duck Harness: /game [gridworld|mini-arc] [level] [--show-thinking] or /game stop",
            )
        ]

    def handle_slash(self, command: str, args: str, context: ExtensionContext) -> bool:
        if command != "game":
            return False

        import threading
        from rich.panel import Panel
        from agent.progress import build_turn_panel, print_game_header

        parts = args.strip().split()
        first_arg = parts[0].lower() if parts else "gridworld"

        if first_arg == "stop":
            if self._is_running:
                self._cancel_requested = True
                context.print("[yellow]Stopping current game session...[/yellow]")
            else:
                context.print("[dim]No game is currently running.[/dim]")
            return True

        if self._is_running:
            context.print(
                "[yellow]⚠ A game session is already in progress. "
                "Type [bold]/game stop[/bold] to cancel it first.[/yellow]"
            )
            return True

        game_name = first_arg
        level = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
        # Default show_thinking to True so user always sees thinking, unless explicitly disabled with --no-thinking
        show_thinking = not any(p in ("--no-thinking", "no-thinking") for p in parts)
        
        # Parse MCTS mode: --mcts 1 (Fast Code MCTS, 0.02s) vs --mcts 2 (LLM-Guided MCTS)
        use_mcts = 0
        mcts_sims = None
        for i, p in enumerate(parts):
            if p.startswith("--mcts="):
                val = p.split("=", 1)[1]
                if val in ("1", "2"):
                    use_mcts = int(val)
            elif p in ("--mcts", "mcts"):
                if i + 1 < len(parts) and parts[i + 1] in ("1", "2"):
                    use_mcts = int(parts[i + 1])
                else:
                    use_mcts = 1
            elif p in ("--mcts-sims", "--sims") and i + 1 < len(parts):
                try:
                    mcts_sims = int(parts[i + 1])
                except ValueError:
                    pass
            elif p.startswith("--mcts-sims="):
                try:
                    mcts_sims = int(p.split("=", 1)[1])
                except ValueError:
                    pass

        max_turns = 25

        if game_name == "mini-arc":
            env = MiniArcGame(level=level)
        else:
            env = GridWorld()

        self._is_running = True
        self._cancel_requested = False

        print_game_header(game_name, level, context.print)
        if use_mcts == 1:
            context.print("[bold yellow]⚡ Active: Fast Code-driven MCTS (Cách 1: Pure CPU search in ~0.02s)[/bold yellow]\n")
        elif use_mcts == 2:
            context.print("[bold magenta]🧠 Active: LLM-Guided MCTS (Cách 2: LLM policy priors with live thinking stream)[/bold magenta]\n")

        def _worker_thread():
            try:
                counts = {"thinking": 0, "visible": 0}
                thinking_announced = [False]
                think_buffer: list[str] = []

                def on_turn_start(turn_idx: int, frame):
                    counts["thinking"] = 0
                    counts["visible"] = 0
                    thinking_announced[0] = False
                    think_buffer.clear()
                    context.set_spinner(f"Turn {turn_idx}: Observing board...", key="game")

                def on_token(turn_idx: int, delta: str, is_thinking: bool):
                    if is_thinking:
                        counts["thinking"] += 1
                        if show_thinking:
                            if not thinking_announced[0]:
                                thinking_announced[0] = True
                                context.set_spinner("", key="game")
                                context.print(f"\n[bold magenta]💭 Model Thinking (Turn {turn_idx}):[/bold magenta]")
                            think_buffer.append(delta)
                            if len(think_buffer) >= 3 or "\n" in delta:
                                chunk = "".join(think_buffer)
                                think_buffer.clear()
                                context.print(f"[dim italic]{chunk}[/dim italic]", end="")
                        else:
                            if counts["thinking"] % 6 == 1:
                                context.set_spinner(
                                    f"Turn {turn_idx}: [Thinking] ~{counts['thinking']} tokens...",
                                    key="game",
                                )
                    else:
                        counts["visible"] += 1
                        if thinking_announced[0]:
                            if think_buffer:
                                chunk = "".join(think_buffer)
                                think_buffer.clear()
                                context.print(f"[dim italic]{chunk}[/dim italic]", end="")
                            thinking_announced[0] = False
                            if show_thinking:
                                context.print("\n")
                            context.set_spinner(f"Turn {turn_idx}: [Writing code] ~{counts['visible']} tokens...", key="game")
                        elif not show_thinking and counts["visible"] % 6 == 1:
                            context.set_spinner(
                                f"Turn {turn_idx}: [Writing code] ~{counts['visible']} tokens...",
                                key="game",
                            )

                def on_step(turn_idx: int, turn_record, curr_env):
                    if think_buffer:
                        chunk = "".join(think_buffer)
                        think_buffer.clear()
                        context.print(f"[dim italic]{chunk}[/dim italic]", end="")
                    if thinking_announced[0]:
                        thinking_announced[0] = False
                        context.print("\n")
                    context.set_spinner("", key="game")
                    panel = build_turn_panel(turn_idx, turn_record, curr_env, max_turns=max_turns)
                    context.print(panel)

                llm = LLMClient(extension_context=context, max_tokens=512)
                agent = DuckAgent(
                    env=env,
                    llm=llm,
                    max_turns=max_turns,
                    verbose=False,
                    on_step_callback=on_step,
                    on_turn_start=on_turn_start,
                    on_token=on_token,
                    cancel_check=lambda: self._cancel_requested,
                    use_mcts=use_mcts,
                    mcts_sims=mcts_sims,
                )
                result = agent.run()
                context.set_spinner("", key="game")

                if self._cancel_requested:
                    context.print("[yellow]Game session cancelled by user.[/yellow]")
                elif result.solved:
                    context.print(
                        Panel.fit(
                            f"[bold green]✓ LEVEL {level} SOLVED SUCCESSFULLY![/bold green]\n"
                            f"Total Action Steps: [bold]{result.total_steps}[/bold] | Total Reward: [bold]{result.total_reward}[/bold]\n\n"
                            f"[bold]Discovered World Model:[/bold]\n{result.world_model or '(none)'}",
                            title="[bold green]🏆 Victory[/bold green]",
                            border_style="green",
                        )
                    )
                else:
                    context.print(
                        Panel.fit(
                            f"[bold red]✗ Game ended without solving ({result.total_steps} steps).[/bold red]\n\n"
                            f"[bold]World Model Discoveries:[/bold]\n{result.world_model or '(none)'}",
                            title="[bold red]Game Over[/bold red]",
                            border_style="red",
                        )
                    )
            except Exception as exc:
                context.set_spinner("", key="game")
                context.print(f"[bold red]Error during game execution:[/bold red] {exc}")
            finally:
                self._is_running = False
                self._cancel_requested = False

        # Start worker thread so prompt_toolkit UI event loop remains active and prints live!
        thread = threading.Thread(target=_worker_thread, daemon=True)
        thread.start()
        return True


EXTENSION = GameExtension()
