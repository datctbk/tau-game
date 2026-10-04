"""Python REPL sandbox for the Duck Harness agent with invariant guards."""

from __future__ import annotations

import ast
import io
import logging
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.guards import (
    KnownDeathActionError,
    KnownNoOpActionError,
    NoopRepeatGuard,
    RepeatedActionInStateError,
    TerminalStateActionError,
    action_signature,
    state_hash,
)
from env.environment import BaseEnvironment, Frame, Transition

logger = logging.getLogger(__name__)

# Disallowed dangerous modules/builtins for safe local execution
BLOCKED_MODULES = {"os", "subprocess", "shutil", "socket", "sys", "urllib", "requests", "http"}


def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
    base = name.split(".")[0]
    if base in BLOCKED_MODULES:
        raise ImportError(f"Import of {name!r} is restricted in the Duck REPL sandbox.")
    return __import__(name, *args, **kwargs)


@dataclass
class REPLResult:
    """Result of running code in the REPL sandbox."""

    output: str
    actions_taken: list[str] = field(default_factory=list)
    transitions_log: list[str] = field(default_factory=list)
    last_frame: Frame | None = None
    error: str | None = None
    success: bool = True

    def __str__(self) -> str:
        parts: list[str] = []
        if self.output.strip():
            parts.append(self.output.strip())
        if self.transitions_log:
            parts.append("Step transitions:\n" + "\n".join(f"  - {t}" for t in self.transitions_log))
        elif self.actions_taken:
            parts.append(f"Actions executed: {self.actions_taken}")
        if self.error:
            parts.append(f"REPL Error: {self.error}")
        return "\n".join(parts) if parts else "(No output / actions)"


class PythonREPL:
    """Sandboxed Python execution environment exposing the game state to the LLM with invariant guards."""

    def __init__(
        self,
        env: BaseEnvironment,
        world_model_ref: Callable[[], str] | None = None,
        max_output_chars: int = 4000,
        enable_guards: bool = True,
    ) -> None:
        self.env = env
        self.world_model_ref = world_model_ref
        self.max_output_chars = max_output_chars
        self.enable_guards = enable_guards
        self.noop_guard = NoopRepeatGuard()
        self._custom_namespace: dict[str, Any] = {}

    def _diff_grids(self, before_grid: list[list[int]], after_grid: list[list[int]]) -> str:
        """Compute visual delta between two grid states without domain-specific knowledge."""
        from perception.segmentation import COLOR_TO_CHAR

        changes = []
        for r in range(len(before_grid)):
            for c in range(len(before_grid[0])):
                if before_grid[r][c] != after_grid[r][c]:
                    c_old = COLOR_TO_CHAR.get(before_grid[r][c], str(before_grid[r][c]))
                    c_new = COLOR_TO_CHAR.get(after_grid[r][c], str(after_grid[r][c]))
                    changes.append(f"({r},{c}) {c_old}->{c_new}")
        if not changes:
            return "blocked (no change)"
        return ", ".join(changes)

    def _build_namespace(
        self,
        actions_log: list[str],
        transitions_log: list[str],
        seen_in_snippet: set[tuple[str, str]],
        terminal_state: list[str],
    ) -> dict[str, Any]:
        """Expose game variables and action callable to the REPL."""
        curr = self.env.current_frame
        prev = self.env.previous_frame

        def action_fn(action: str | list[str], **kwargs: Any) -> Frame:
            """Execute an action or sequence of actions in the environment with empirical feedback."""
            act_list = list(action) if isinstance(action, (list, tuple)) else [action]
            last_res = self.env.current_frame

            for act in act_list:
                act_str = str(act).upper().strip()
                act_sig = action_signature(act_str)
                curr_grid = self.env.current_frame.grid
                curr_h = state_hash(curr_grid)

                # 1. Terminal state check
                if terminal_state:
                    raise TerminalStateActionError(
                        f"Action '{act_str}' refused: previous action reached terminal state ({terminal_state[0]})."
                    )

                if self.enable_guards:
                    # 2. In-snippet loop check (repeating same action from identical state in one snippet)
                    if (curr_h, act_sig) in seen_in_snippet:
                        raise RepeatedActionInStateError(
                            f"Action '{act_str}' has already been executed from this exact board state in this snippet. Infinite loop prevented."
                        )

                    # 3. Known no-op guard (overridable on immediate repeat)
                    if self.noop_guard.should_block(curr_h, act_sig):
                        raise KnownNoOpActionError(
                            f"Action '{act_str}' is a known no-op from this state (caused no change). "
                            "Execution refused to save action budget. To override, issue this exact action again."
                        )

                seen_in_snippet.add((curr_h, act_sig))
                actions_log.append(act_str)
                before_frame = self.env.current_frame

                last_res = self.env.step(act_str, **kwargs)
                diff = self._diff_grids(before_frame.grid, last_res.grid)
                gameplay_changed = (diff != "blocked (no change)")

                if self.enable_guards:
                    self.noop_guard.observe(curr_h, act_sig, gameplay_changed=gameplay_changed)
                    self.noop_guard.note_executed(curr_h, act_sig)

                outcome = f"{act_str}: {diff}"
                if self.env.is_done():
                    outcome += " [GOAL REACHED!]"
                    terminal_state.append("GOAL_REACHED")
                transitions_log.append(outcome)

                if self.env.is_done():
                    break

            return last_res

        safe_builtins = dict(__builtins__) if isinstance(__builtins__, dict) else vars(__builtins__).copy()
        safe_builtins["__import__"] = _safe_import

        world_model_text = self.world_model_ref() if self.world_model_ref else ""

        ns: dict[str, Any] = {
            "__builtins__": safe_builtins,
            # Core Duck runtime state variables
            "current_frame": curr,
            "previous_frame": prev,
            "history": self.env.history,
            "transitions": self.env.history,
            "valid_actions": self.env.valid_actions(),
            "world_model": world_model_text,
            # Direct action dispatcher
            "action": action_fn,
            "step": action_fn,
            # Expose guards and exceptions to sandbox
            "KnownNoOpActionError": KnownNoOpActionError,
            "RepeatedActionInStateError": RepeatedActionInStateError,
            "TerminalStateActionError": TerminalStateActionError,
            "KnownDeathActionError": KnownDeathActionError,
        }

        # Include persistent user variables across turns
        ns.update(self._custom_namespace)
        return ns

    def execute(self, code: str) -> REPLResult:
        """Execute Python code in the sandbox with guard checks and captured output."""
        actions_log: list[str] = []
        transitions_log: list[str] = []
        seen_in_snippet: set[tuple[str, str]] = set()
        terminal_state: list[str] = ["ALREADY_COMPLETED"] if self.env.is_done() else []

        self.noop_guard.note_level(self.env.current_frame.level)
        ns = self._build_namespace(actions_log, transitions_log, seen_in_snippet, terminal_state)

        stdout_capture = io.StringIO()
        old_stdout = sys.stdout

        error_msg: str | None = None
        success = True

        try:
            sys.stdout = stdout_capture
            code_text = str(code or "").strip()
            if code_text:
                tree = ast.parse(code_text, "<duck_repl>")
                last_expr = None
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    last_expr = ast.Expression(tree.body[-1].value)
                    ast.copy_location(last_expr, tree.body[-1])
                    tree.body = tree.body[:-1]

                if tree.body:
                    exec(compile(tree, "<duck_repl>", "exec"), ns)
                if last_expr is not None:
                    value = eval(compile(last_expr, "<duck_repl>", "eval"), ns)
                    if value is not None:
                        print(repr(value))
        except (
            KnownNoOpActionError,
            RepeatedActionInStateError,
            TerminalStateActionError,
            KnownDeathActionError,
            Exception,
        ) as exc:
            success = False
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.debug("REPL execution error:\n%s", traceback.format_exc())
        finally:
            sys.stdout = old_stdout

        output = stdout_capture.getvalue()
        if len(output) > self.max_output_chars:
            output = output[: self.max_output_chars] + f"\n... (truncated {len(output)} chars)"

        # Save any user-defined helper variables/functions (excluding internal state)
        for k, v in ns.items():
            if not k.startswith("_") and k not in {
                "current_frame",
                "previous_frame",
                "history",
                "transitions",
                "valid_actions",
                "world_model",
                "action",
                "step",
                "KnownNoOpActionError",
                "RepeatedActionInStateError",
                "TerminalStateActionError",
                "KnownDeathActionError",
            }:
                self._custom_namespace[k] = v

        return REPLResult(
            output=output,
            actions_taken=actions_log,
            transitions_log=transitions_log,
            last_frame=self.env.current_frame,
            error=error_msg,
            success=success,
        )

