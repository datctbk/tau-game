"""Tests for agent/memory.py: history, world model tracking, and context eviction."""

import sys
from pathlib import Path

pkg_root = Path(__file__).resolve().parent.parent
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

from agent.memory import GameMemory


def test_memory_world_model_extraction():
    memory = GameMemory()

    response = """
I noticed moving UP caused no position change because of a wall.
World model:
- Color 5 represents solid walls.
- Player starts at (1, 1).
- Green 3 is the goal.

```python
action("DOWN")
```
"""
    memory.record_turn(
        step=1,
        observation_summary="step 1",
        response=response,
        code='action("DOWN")',
        repl_output="Executed DOWN",
        actions=["DOWN"],
    )

    assert "Color 5 represents solid walls" in memory.world_model
    assert "Green 3 is the goal" in memory.world_model


def test_memory_context_eviction():
    # Set small turn budget to test eviction
    memory = GameMemory(max_turns_in_context=2)

    messages = [
        {"role": "system", "content": "You are a game agent."},
        {"role": "user", "content": "Turn 1 obs"},
        {"role": "assistant", "content": "Turn 1 reply"},
        {"role": "user", "content": "Turn 2 obs"},
        {"role": "assistant", "content": "Turn 2 reply"},
        {"role": "user", "content": "Turn 3 obs"},
        {"role": "assistant", "content": "Turn 3 reply"},
    ]

    trimmed = memory.trim_messages(messages)

    # System prompt MUST always be preserved at index 0
    assert trimmed[0]["role"] == "system"
    assert trimmed[0]["content"] == "You are a game agent."

    # Only 2 most recent user/assistant turns should remain (4 messages + system = 5 total)
    assert len(trimmed) == 5
    assert trimmed[1]["content"] == "Turn 2 obs"
    assert trimmed[-1]["content"] == "Turn 3 reply"
