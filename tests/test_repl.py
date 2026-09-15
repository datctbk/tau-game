"""Tests for agent/repl.py execution sandbox."""

import sys
from pathlib import Path

pkg_root = Path(__file__).resolve().parent.parent
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

from agent.repl import PythonREPL
from env.game import GridWorld


def test_repl_variable_access_and_action_dispatch():
    env = GridWorld(height=5, width=5, player_start=(1, 1), goal_pos=(3, 3), walls=[])
    repl = PythonREPL(env=env)

    code = """
print("Current step:", current_frame.step)
print("Shape:", current_frame.shape)
action("DOWN")
action(["RIGHT", "DOWN"])
"""
    result = repl.execute(code)
    assert result.success is True
    assert result.error is None
    assert "Current step: 0" in result.output
    assert "Shape: (5, 5)" in result.output
    assert result.actions_taken == ["DOWN", "RIGHT", "DOWN"]
    assert env.current_frame.info["player"] == (3, 2)


def test_repl_sandbox_security():
    env = GridWorld()
    repl = PythonREPL(env=env)

    # Attempting to import restricted os module
    code = """
import os
os.system("echo hacked")
"""
    result = repl.execute(code)
    assert result.success is False
    assert "ImportError" in result.error
