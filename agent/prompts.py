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
- Coordinates are strictly `(row, col)`, 0-indexed from top-left (`UP` = row-1, `DOWN` = row+1, `LEFT` = col-1, `RIGHT` = col+1).
- Key object positions are already listed below the ASCII board every turn — do NOT waste turns running `print(current_frame.segmentation)` without calling `action(...)`.

### Execution Safety & Guards:
- **Known No-Op Guard**: An action already proven to change nothing (`blocked (no change)`) from the exact same board state is refused before execution (`KnownNoOpActionError`), saving your move budget. If you have reason to believe it will behave differently now, re-issuing that exact action again immediately will override the guard.
- **Loop Breaker Guard**: Repeating the same action from the exact same board state within a single Python snippet is refused (`RepeatedActionInStateError`) to prevent runaway loops.
- **Terminal State Guard**: Once the puzzle or level is completed, subsequent actions in the snippet are safely halted (`TerminalStateActionError`).

### Rules of Engagement:
1. **Empirical Discovery (Learn from History)**: Symbols and mechanics are completely unknown. Inspect transition feedback:
   - If an object shifts position when you move into it, you pushed it!
   - Pushable blocks typically need to be guided and pushed onto target pads/markers to solve the puzzle.
   - Pushing requires open space: If an object has a wall or boundary behind it in the push direction, pushing into it is BLOCKED. Maneuver to another side to push it into an open tile ('.') or target pad.
   - Deadlock & RESET: Never push a block onto an outer wall/corner unless a target pad is on that same wall. If a block is trapped against a wall/corner with no way to reach a target pad, call `action("RESET")` immediately instead of overthinking.
   - Collectibles (keys, items): Collected by stepping directly ONTO their tile (`.->B`), not from a distance.
   - The SPACE action: `SPACE` acts as a remote switch or toggle for phase barriers/mechanisms (e.g., toggling a barrier `# <-> .`). It does NOT collect or interact with adjacent items! Inspect the transition output: if only a remote coordinate changed, `SPACE` only toggles that remote coordinate. Never spam `SPACE` expecting adjacent items to be collected.
   - If an action result says "blocked (no change)", movement is blocked in that direction; you CANNOT pass through it. Pick an unblocked direction.
2. **FORWARD-LOOKING PRINCIPLE (No Retrospective Overthinking)**:
   - Use history ONLY for binary checks: did your last move get blocked? Did you trigger a barrier/pickup?
   - NEVER re-analyze or write paragraphs dissecting past move sequences.
   - Your current position on the ASCII board is your sole ground truth. Look at where B is right now, find unvisited symbols or unblocked paths, and navigate forward!
3. **CHAIN MOVES**: Plan 2 to 6 steps ahead and call `action(["MOVE1", "MOVE2", ...])` directly to maneuver, push, or explore.
4. **REASONING & WORLD MODEL**:
   - Limit reasoning text to 1 short sentence before code.
   - You MAY list multiple concise bullet points under `World model:` to track all discovered mechanics (symbols, triggers, doors, goals). Retain past mechanics and append new ones!
5. **IMMEDIATE CODE**: Output your ```python action(["MOVE1", "MOVE2", ...]) ``` code block with actual moves immediately. Do NOT output a literal `[...]` placeholder.

### Example Response:
Moving right entered the portal and teleported.
World model:
- '#' is wall, '.' is floor, 'B' is player
- Trigger at (1,2) opened barrier at (1,3)
- 'W' acts as teleport portal
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
    empirical_facts: list[str] | None = None,
) -> str:
    """Build the user turn observation prompt with empirical transition feedback."""
    lines: list[str] = [
        f"=== Game Turn (Step: {current_frame.step} | Level: {current_frame.level}) ===",
        f"Valid Actions: {valid_actions}",
    ]

    if recent_actions:
        lines.append(f"Last Actions Taken: {recent_actions}")

    if last_repl_output:
        trimmed_repl = last_repl_output.strip()
        if len(trimmed_repl) > 1200:
            trimmed_repl = trimmed_repl[:1200] + "\n... (truncated)"
        lines.append(f"Last Action Results:\n{trimmed_repl}")

    # Empirical check if actions produced zero physical change
    if previous_frame and recent_actions and current_frame.grid == previous_frame.grid:
        lines.append(
            "⚠️ NOTICE: The board did NOT change after your last action sequence (movement was blocked). Do NOT repeat the exact same actions; test a different path or direction!"
        )

    if empirical_facts:
        lines.append("\nVerified Empirical Observations (Physical Facts):")
        for fact in empirical_facts[-6:]:
            lines.append(f"- {fact}")

    lines.append("\nCurrent Board (ASCII, coords=(row, col), 0-indexed from top-left):")
    lines.append(current_frame.ascii)

    # Compact object coordinate summary so the model doesn't waste turns inspecting segmentation
    from perception.segmentation import COLOR_TO_CHAR

    symbol_positions: dict[str, list[tuple[int, int]]] = {}
    for r, row in enumerate(current_frame.grid):
        for c, val in enumerate(row):
            if val in (0, 5):  # skip floor (.) and wall (#)
                continue
            ch = COLOR_TO_CHAR.get(val, str(val))
            symbol_positions.setdefault(ch, []).append((r, c))
    if symbol_positions:
        summary_parts = [f"{ch}={coords[0] if len(coords) == 1 else coords}" for ch, coords in sorted(symbol_positions.items())]
        lines.append(f"Key Positions (row, col): {', '.join(summary_parts)}")

    if current_frame.info:
        covered = current_frame.info.get("covered_targets")
        if covered:
            lines.append(f"Covered Targets (box already on target pad — do NOT push off): {covered}")
        status_flags = [f"{k}={v}" for k, v in current_frame.info.items() if v is True and k != "space_toggle"]
        if status_flags:
            lines.append(f"\nPlayer Status / Inventory: {status_flags}")

    if world_model:
        lines.append(f"\nKnown World Model:\n{world_model}")

    lines.append(
        "\nCRITICAL: Keep reasoning under 20 words. Look at B's current position and plan forward toward remaining symbols. "
        "Directly output concrete moves, e.g. ```python action([\"DOWN\", \"RIGHT\"])```. Do NOT use '...' or '[...]' placeholders."
    )
    return "\n".join(lines)


