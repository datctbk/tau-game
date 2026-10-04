"""Tests for Duck Harness invariant guards and REPL enhancements."""

import sys
from pathlib import Path

pkg_root = Path(__file__).resolve().parent.parent
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

import pytest
from agent.guards import (
    KnownNoOpActionError,
    NoopRepeatGuard,
    RepeatedActionInStateError,
    TerminalStateActionError,
    action_signature,
    state_hash,
)
from agent.repl import PythonREPL
from env.game import GridWorld


def test_action_signature_and_state_hash():
    assert action_signature("down") == "DOWN"
    assert action_signature({"action": "MOUSE", "row": 3, "col": 4}) == "MOUSE(3,4)"
    assert action_signature("MOUSE(row=3, col=4)") == "MOUSE(3,4)"

    grid_a = [[0, 1], [2, 3]]
    grid_b = [[0, 1], [2, 3]]
    grid_c = [[0, 1], [2, 4]]
    assert state_hash(grid_a) == state_hash(grid_b)
    assert state_hash(grid_a) != state_hash(grid_c)


def test_known_noop_guard_blocking_and_override():
    env = GridWorld(height=5, width=5, player_start=(0, 0), goal_pos=(4, 4), walls=[])
    repl = PythonREPL(env=env)

    # Move UP into wall/boundary at (0, 0) -> blocked
    res1 = repl.execute("action('UP')")
    assert res1.success is True
    assert "blocked (no change)" in res1.transitions_log[0]

    # Repeating the same action from the same state raises KnownNoOpActionError
    res2 = repl.execute("action('UP')")
    assert res2.success is False
    assert "KnownNoOpActionError" in res2.error

    # Insisting on the exact same action in the immediate next call overrides the guard!
    res3 = repl.execute("action('UP')")
    assert res3.success is True
    assert "blocked (no change)" in res3.transitions_log[0]


def test_repeated_action_in_snippet_loop_breaker():
    env = GridWorld(height=5, width=5, player_start=(0, 0), goal_pos=(4, 4), walls=[])
    repl = PythonREPL(env=env)

    # An infinite loop in snippet hitting the wall repeatedly
    code = """
for _ in range(10):
    action("UP")
"""
    res = repl.execute(code)
    assert res.success is False
    assert "RepeatedActionInStateError" in res.error
    assert len(res.actions_taken) == 1  # Blocked on second attempt from same state!


def test_base_exception_cannot_be_swallowed_by_except_exception():
    env = GridWorld(height=5, width=5, player_start=(0, 0), goal_pos=(4, 4), walls=[])
    repl = PythonREPL(env=env)

    # Turn 1: Block UP
    repl.execute("action('UP')")

    # Turn 2: LLM writes defensive code trying to swallow errors
    code = """
swallowed = False
try:
    action("UP")
except Exception:
    swallowed = True
"""
    res = repl.execute(code)
    # Because KnownNoOpActionError inherits from BaseException, except Exception does NOT swallow it
    assert res.success is False
    assert "KnownNoOpActionError" in res.error


def test_terminal_state_action_guard():
    # Player starts 1 step away from goal
    env = GridWorld(height=5, width=5, player_start=(3, 4), goal_pos=(4, 4), walls=[])
    repl = PythonREPL(env=env)

    # Action DOWN reaches goal
    res = repl.execute("action(['DOWN', 'RIGHT'])")
    assert res.success is True
    assert "GOAL REACHED!" in str(res)
    # RIGHT was skipped after DOWN reached goal
    assert res.actions_taken == ["DOWN"]

    # Calling action after game is done raises TerminalStateActionError
    res_after = repl.execute("action('LEFT')")
    assert res_after.success is False
    assert "TerminalStateActionError" in res_after.error


def test_auto_print_last_expression():
    env = GridWorld(height=5, width=5, player_start=(1, 1), goal_pos=(4, 4), walls=[])
    repl = PythonREPL(env=env)

    # Model writes variable expression without print()
    code = """
x = 42
current_frame.shape
"""
    res = repl.execute(code)
    assert res.success is True
    assert "(5, 5)" in res.output
