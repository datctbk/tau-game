"""Prompts and observation formatting for the Duck Harness agent."""

from __future__ import annotations

from typing import Any

from env.environment import Frame

SYSTEM_PROMPT = """You are an autonomous game-solving agent inspired by Tufa Labs' Duck Harness for ARC-AGI-3.
Your goal is to discover the hidden rules, mechanics, and objectives of the environment to solve each puzzle/level.

### Interactive Python REPL
Instead of outputting simple text actions, you generate Python code blocks to inspect the environment, run experiments, analyze structures, and execute moves.

Available Environment Variables in Python:
- `current_frame`: Object with `.grid`, `.ascii`, `.shape`, `.level`, `.step`, and `.segmentation`.
- `current_frame.ascii`: Quick 2D character map of the current board.
- `current_frame.segmentation`: List of 4-connected same-color object components with bounding boxes (`bbox`), centroids, colors, and shape hashes.
- `previous_frame`: Observation before the last action.
- `valid_actions`: List of available actions (e.g. ['UP', 'DOWN', 'LEFT', 'RIGHT', 'SPACE', 'RESET']).
- `history`: List of past transitions (step, action, reward, done).
- `world_model`: Your accumulated knowledge and hypotheses about mechanics and rules.

Available Function:
- `action("ACTION_NAME")` or `action(["MOVE1", "MOVE2", ...])`: Executes one or multiple moves in the game environment.
  Example: `action(["DOWN", "DOWN", "RIGHT"])` or `action("UP")`

### Rules of Engagement:
1. **Empirical Discovery**: Symbols and mechanics are completely unknown. Inspect transition feedback:
   - If an object shifts position when you move into it, you pushed it!
   - Pushable blocks typically need to be guided and pushed onto target pads/markers to solve the puzzle.
   - If an object gets stuck against a wall/corner where it can no longer reach the target (deadlock), call `action("RESET")` to restart fresh.
   - If an action result says "blocked (no change)", movement is blocked in that direction.
2. **CHAIN MOVES**: Plan 2 to 6 steps ahead and call `action(["MOVE1", "MOVE2", ...])` directly to maneuver, push, or explore.
3. **BE EXTREMELY BRIEF**: Limit reasoning to 1-2 short sentences. Never write long essays or monologues.
4. **IMMEDIATE CODE**: Output your ```python ... ``` code block immediately.
5. **World Model**: Include a 1-line `World model: ...` summarizing your discovered hypotheses.

### Example Response:
Moving into the object shifted it; it is a pushable block.
World model: '#' is wall, '.' is floor, B is player, object moves when pushed.
```python
action(["DOWN", "DOWN", "RIGHT", "RIGHT"])
```
"""


def build_turn_prompt(
    current_frame: Frame,
    valid_actions: list[str],
    world_model: str,
    recent_actions: list[str] | None = None,
    last_repl_output: str | None = None,
    previous_frame: Frame | None = None,
) -> str:
    """Build the user turn observation prompt with empirical transition feedback."""
    lines: list[str] = [
        f"=== Game Turn (Step: {current_frame.step} | Level: {current_frame.level}) ===",
        f"Valid Actions: {valid_actions}",
    ]

    if recent_actions:
        lines.append(f"Last Actions Taken: {recent_actions}")

    if last_repl_output:
        lines.append(f"Last Action Results:\n{last_repl_output.strip()}")

    # Empirical check if actions produced zero physical change
    if previous_frame and recent_actions and current_frame.grid == previous_frame.grid:
        lines.append(
            "⚠️ NOTICE: The board did NOT change after your last action sequence (movement was blocked). Do NOT repeat the exact same actions; test a different path or direction!"
        )

    lines.append("\nCurrent Board (ASCII):")
    lines.append(current_frame.ascii)

    if world_model:
        lines.append(f"\nKnown World Model:\n{world_model}")

    lines.append(
        "\nBe extremely concise (1-2 sentences max). Directly output ```python action([...])```."
    )
    return "\n".join(lines)


