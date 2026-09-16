"""Unit and integration tests for LLM-Guided MCTS (agent/mcts.py)."""

from __future__ import annotations

import pytest

from agent.agent import DuckAgent
from agent.mcts import (
    LLMGuidedMCTS,
    MCTSNode,
    estimate_state_heuristic,
    is_corner_deadlock,
)
from env.game import GridWorld, MiniArcGame
from llm.client import LLMClient


def test_is_corner_deadlock():
    """Verify corner deadlock detection for pushable crates."""
    # Free crate in open space
    grid_free = [
        [5, 5, 5, 5, 5],
        [5, 0, 0, 0, 5],
        [5, 0, 8, 0, 5],  # 8 is crate C
        [5, 0, 0, 0, 5],
        [5, 5, 5, 5, 5],
    ]
    assert not is_corner_deadlock(grid_free)

    # Crate trapped in top-left corner
    grid_trapped_top_left = [
        [5, 5, 5, 5, 5],
        [5, 8, 0, 0, 5],  # Crate at (1,1) bordered by wall at (0,1) and (1,0)
        [5, 0, 0, 0, 5],
        [5, 0, 0, 0, 5],
        [5, 5, 5, 5, 5],
    ]
    assert is_corner_deadlock(grid_trapped_top_left)

    # Crate trapped in bottom-right corner
    grid_trapped_bottom_right = [
        [5, 5, 5, 5, 5],
        [5, 0, 0, 0, 5],
        [5, 0, 0, 0, 5],
        [5, 0, 0, 8, 5],  # Crate at (3,3) bordered by wall at (4,3) and (3,4)
        [5, 5, 5, 5, 5],
    ]
    assert is_corner_deadlock(grid_trapped_bottom_right)


def test_estimate_state_heuristic():
    """Verify heuristic scores closer states higher and deadlocks as -1.0."""
    grid_deadlock = [
        [5, 5, 5, 5],
        [5, 8, 0, 5],
        [5, 0, 6, 5],
        [5, 5, 5, 5],
    ]
    assert estimate_state_heuristic(grid_deadlock) == -1.0

    # Closer crate to target has higher heuristic
    grid_far = [
        [5, 5, 5, 5, 5, 5],
        [5, 0, 8, 0, 0, 5],  # Crate at (1,2)
        [5, 0, 0, 0, 0, 5],
        [5, 0, 0, 0, 6, 5],  # Target at (3,4), dist = (3-1) + (4-2) = 4
        [5, 5, 5, 5, 5, 5],
    ]
    grid_near = [
        [5, 5, 5, 5, 5, 5],
        [5, 0, 0, 0, 0, 5],
        [5, 0, 0, 8, 0, 5],  # Crate at (2,3)
        [5, 0, 0, 0, 6, 5],  # Target at (3,4), dist = (3-2) + (4-3) = 2
        [5, 5, 5, 5, 5, 5],
    ]
    score_far = estimate_state_heuristic(grid_far)
    score_near = estimate_state_heuristic(grid_near)
    assert score_near > score_far


def test_mcts_node_puct_selection():
    """Verify PUCT formula balances Q-value and prior exploration."""
    env = GridWorld()
    root = MCTSNode(state=env, frame=env.current_frame)

    child_a = MCTSNode(state=env, frame=env.current_frame, parent=root, action_from_parent="UP", prior=0.7)
    child_b = MCTSNode(state=env, frame=env.current_frame, parent=root, action_from_parent="RIGHT", prior=0.3)
    root.children = {"UP": child_a, "RIGHT": child_b}

    # At 0 parent visits, child with higher prior has higher PUCT
    score_a = child_a.puct_score(parent_visits=1, c_puct=1.414)
    score_b = child_b.puct_score(parent_visits=1, c_puct=1.414)
    assert score_a > score_b

    # If child_b gets high reward Q, it can overcome lower prior
    child_b.visit_count = 2
    child_b.total_value = 2.0  # Q = 1.0
    child_a.visit_count = 2
    child_a.total_value = -1.0  # Q = -0.5

    assert child_b.puct_score(parent_visits=5) > child_a.puct_score(parent_visits=5)


def test_llm_guided_mcts_mock_prior_and_search():
    """Verify LLMGuidedMCTS can expand and find valid actions with mock LLM."""
    env = GridWorld()

    def mock_prior_generator(messages):
        return (
            '{"actions": [{"action": "DOWN", "prior": 0.8, "reason": "Move down toward goal"}, '
            '{"action": "RIGHT", "prior": 0.2, "reason": "Alternative lateral move"}], '
            '"rationale": "Clear corridor moving down"}'
        )

    mock_llm = LLMClient(mock_fn=mock_prior_generator)
    mcts = LLMGuidedMCTS(llm=mock_llm, num_simulations=8, max_depth=4)

    res = mcts.search(env)
    assert res.best_action in env.valid_actions()
    assert res.tree_size >= 2
    assert res.root_visits >= 1


def test_duck_agent_with_mcts_solves_gridworld():
    """Verify DuckAgent operates seamlessly when use_mcts=True."""
    env = GridWorld(height=5, width=5, player_start=(1, 1), goal_pos=(3, 3), walls=[(2, 2)])

    # Heuristic MCTS with no LLM (uses internal simulation & heuristic)
    agent = DuckAgent(
        env=env,
        llm=LLMClient(mock_fn=lambda msgs: "action('DOWN')"),
        max_turns=15,
        use_mcts=True,
        mcts_sims=12,
        verbose=False,
    )

    result = agent.run()
    assert result.solved
    assert result.total_steps > 0
    assert len(result.turns) > 0


def test_mcts_sokoban_level2_avoids_deadlock():
    """Verify MCTS on Sokoban Level 2 selects viable moves without crashing."""
    env = MiniArcGame(level=2)

    mcts = LLMGuidedMCTS(num_simulations=10, max_depth=5)
    result = mcts.search(env)

    assert result.best_action in env.valid_actions()
    assert result.best_action != "RESET"
    assert result.tree_size > 1


def test_mcts_streams_thinking_tokens():
    """Verify LLMGuidedMCTS correctly pipes reasoning/thinking tokens to on_token."""
    env = GridWorld()
    captured_tokens: list[tuple[str, bool]] = []

    def mock_thinking_llm(messages, on_token=None):
        if on_token:
            on_token("Thinking about best moves...", True)
            on_token("Done thinking.", True)
            on_token('{"actions": [{"action": "DOWN", "prior": 1.0}]}', False)
        return '{"actions": [{"action": "DOWN", "prior": 1.0}]}'

    mock_llm = LLMClient(mock_fn=mock_thinking_llm)
    mcts = LLMGuidedMCTS(
        llm=mock_llm,
        num_simulations=4,
        on_token=lambda d, is_t: captured_tokens.append((d, is_t)),
    )

    res = mcts.search(env)
    assert len(captured_tokens) > 0
    thinking_tokens = [t for t, is_t in captured_tokens if is_t]
    assert len(thinking_tokens) >= 2
    assert "Thinking about best moves..." in thinking_tokens

