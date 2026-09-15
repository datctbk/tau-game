"""Tests for DuckAgent orchestrator loop."""

import sys
from pathlib import Path

pkg_root = Path(__file__).resolve().parent.parent
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

from agent.agent import DuckAgent
from env.game import GridWorld
from llm.client import LLMClient
from main import create_mock_gridworld_solver


def test_duck_agent_solves_gridworld_end_to_end():
    env = GridWorld(
        height=6,
        width=6,
        player_start=(1, 1),
        goal_pos=(4, 4),
        walls=[(2, 2), (2, 3), (3, 2)],
    )

    llm = LLMClient(mock_fn=create_mock_gridworld_solver())
    agent = DuckAgent(env=env, llm=llm, max_turns=10, verbose=False)

    result = agent.run()

    assert result.solved is True
    assert result.total_steps > 0
    assert result.total_reward == 1.0
    assert "moving toward g" in result.world_model.lower()
    assert len(result.turns) == 3
    assert env.is_done()
