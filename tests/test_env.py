"""Tests for tau-game environments (GridWorld and MiniArcGame)."""

import sys
from pathlib import Path

# Add package root to sys.path
pkg_root = Path(__file__).resolve().parent.parent
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

from env.game import GridWorld, MiniArcGame


def test_gridworld_navigation_and_walls():
    env = GridWorld(
        height=5,
        width=5,
        player_start=(1, 1),
        goal_pos=(3, 3),
        walls=[(1, 2)],
    )
    frame = env.reset()
    assert frame.shape == (5, 5)
    assert not env.is_done()
    assert "UP" in env.valid_actions()

    # Move into wall at (1, 2)
    new_frame = env.step("RIGHT")
    assert new_frame.info["player"] == (1, 1)  # blocked by wall

    # Move DOWN to (2, 1)
    new_frame = env.step("DOWN")
    assert new_frame.info["player"] == (2, 1)

    # Move RIGHT to (2, 2)
    new_frame = env.step("RIGHT")
    assert new_frame.info["player"] == (2, 2)

    # Move RIGHT to (2, 3)
    new_frame = env.step("RIGHT")
    assert new_frame.info["player"] == (2, 3)

    # Move DOWN to (3, 3) -> Goal!
    final_frame = env.step("DOWN")
    assert final_frame.info["player"] == (3, 3)
    assert env.is_done()
    assert len(env.history) == 5
    assert env.history[-1].reward == 1.0


def test_mini_arc_game_level1_key_door():
    env = MiniArcGame(level=1)
    env.reset()
    assert not env.is_done()

    # 1. Try moving straight toward door without key -> blocked!
    env.step("DOWN")  # (2, 1)
    env.step("DOWN")  # (3, 1)
    env.step("RIGHT")  # (3, 2)
    blocked_frame = env.step("RIGHT")  # Door is at (3, 3), blocked!
    assert not env.is_done()
    assert blocked_frame.info["player"] == (3, 2)
    assert not blocked_frame.info["has_key"]

    # 2. Move down to pick up Yellow Key at (5, 1)
    env.step("LEFT")  # (3, 1)
    env.step("DOWN")  # (4, 1)
    key_frame = env.step("DOWN")  # (5, 1)
    assert key_frame.info["player"] == (5, 1)
    assert key_frame.info["has_key"] is True

    # 3. Return to Door at (3, 3) and unlock it
    env.step("UP")    # (4, 1)
    env.step("UP")    # (3, 1)
    env.step("RIGHT") # (3, 2)
    door_frame = env.step("RIGHT") # (3, 3) - Unlocked!
    assert door_frame.info["player"] == (3, 3)

    # 4. Move into right room and reach Goal at (5, 5)
    env.step("RIGHT") # (3, 4)
    env.step("DOWN")  # (4, 4)
    env.step("DOWN")  # (5, 4)
    goal_frame = env.step("RIGHT") # (5, 5)
    assert goal_frame.info["player"] == (5, 5)
    assert env.is_done()
    assert env.history[-1].reward == 1.0


def test_mini_arc_game_level3_space_toggle():
    env = MiniArcGame(level=3)
    frame = env.reset()
    assert not env.is_done()

    # Press SPACE to toggle phase barrier
    new_frame = env.step("SPACE")
    assert new_frame.info["space_toggle"] is True
    # Press SPACE again
    toggled_back = env.step("SPACE")
    assert toggled_back.info["space_toggle"] is False


def test_mini_arc_game_level4_dual_box_sokoban():
    env = MiniArcGame(level=4)
    frame = env.reset()
    assert not env.is_done()
    assert frame.shape == (7, 7)

    # Move to box 1 and push to target pad (4, 2)
    moves = ["RIGHT", "DOWN", "DOWN"]
    for m in moves:
        env.step(m)
    assert env.current_frame.grid[4][2] == 8  # Box 1 on target pad
    assert not env.is_done()  # Need both boxes

    # Maneuver to box 2 and push to target pad (4, 4)
    moves2 = ["UP", "UP", "RIGHT", "RIGHT", "DOWN", "DOWN"]
    for m in moves2:
        env.step(m)

    assert env.current_frame.grid[4][4] == 8  # Box 2 on target pad
    assert env.is_done()
    assert env.history[-1].reward == 2.0


def test_mini_arc_game_level5_dual_key_phase_barrier():
    env = MiniArcGame(level=5)
    env.reset()
    assert not env.is_done()

    # 1. Grab Yellow Key at (2, 1)
    env.step("DOWN")
    assert env.current_frame.info["has_key"] is True

    # 2. Unlock Door at (2, 3)
    env.step("RIGHT")
    env.step("RIGHT")
    assert env.current_frame.info["player"] == (2, 3)

    # 3. Navigate to Phase Barrier at (4, 5)
    env.step("RIGHT")  # (2, 4)
    env.step("DOWN")   # (3, 4)
    env.step("RIGHT")  # (3, 5)

    # Phase Barrier at (4, 5) blocks DOWN move
    blocked = env.step("DOWN")
    assert blocked.info["player"] == (3, 5)

    # Press SPACE to drop Phase Barrier
    toggled = env.step("SPACE")
    assert toggled.info["space_toggle"] is True
    assert env.current_frame.grid[4][5] == 0

    # 4. Enter lower chamber and collect Red Key at (6, 6)
    env.step("DOWN")   # (4, 5)
    env.step("DOWN")   # (5, 5)
    env.step("RIGHT")  # (5, 6)
    env.step("DOWN")   # (6, 6)
    assert env.current_frame.info["has_red_key"] is True

    # 5. Move to Red Gate at (5, 3) and unlock it
    env.step("UP")     # (5, 6)
    env.step("LEFT")   # (5, 5)
    env.step("LEFT")   # (5, 4)
    gate_frame = env.step("LEFT")  # (5, 3)
    assert gate_frame.info["player"] == (5, 3)

    # 6. Reach Green Goal at (6, 1)
    env.step("LEFT")   # (5, 2)
    env.step("DOWN")   # (6, 2)
    goal_frame = env.step("LEFT")  # (6, 1)
    assert goal_frame.info["player"] == (6, 1)
    assert env.is_done()
    assert env.history[-1].reward == 1.0


def test_mini_arc_game_level6_wormhole_portal():
    env = MiniArcGame(level=6)
    env.reset()
    assert not env.is_done()

    # 1. Door at (1, 2) blocks goal without key
    blocked = env.step("RIGHT")
    assert blocked.info["player"] == (1, 1)

    # 2. Run down West Wing to Portal A at (7, 1)
    for _ in range(6):
        env.step("DOWN")

    # Stepped into (7, 1) -> Teleported to Portal B at (1, 7)!
    assert env.current_frame.info["player"] == (1, 7)
    assert env.current_frame.grid[7][1] == 9  # Portal A remains active

    # 3. Move down East Wing to (4, 7)
    for _ in range(3):
        env.step("DOWN")
    assert env.current_frame.info["player"] == (4, 7)

    # 4. Flank pushable block at (4, 6) by moving LEFT to push it to (4, 5)
    env.step("LEFT")
    assert env.current_frame.info["player"] == (4, 6)
    assert env.current_frame.grid[4][5] == 8  # Block moved

    # 5. Move down to collect Yellow Key at (7, 6)
    for _ in range(3):
        env.step("DOWN")
    assert env.current_frame.info["player"] == (7, 6)
    assert env.current_frame.info["has_key"] is True

    # 6. Backtrack up East Wing to Portal B at (1, 7)
    for _ in range(3):
        env.step("UP")
    env.step("RIGHT")  # (4, 7)
    for _ in range(3):
        env.step("UP")

    # Teleported back to Portal A at (7, 1)!
    assert env.current_frame.info["player"] == (7, 1)
    assert env.current_frame.grid[1][7] == 9  # Portal B remains active

    # 7. Run up West Wing to Door at (1, 2)
    for _ in range(6):
        env.step("UP")
    assert env.current_frame.info["player"] == (1, 1)

    # 8. Unlock Door and reach Goal at (1, 3)
    env.step("RIGHT")  # (1, 2) Door unlocked!
    goal_frame = env.step("RIGHT")  # (1, 3) Goal!
    assert goal_frame.info["player"] == (1, 3)
    assert env.is_done()
    assert env.history[-1].reward == 1.0


def test_mini_arc_game_level7_quantum_citadel():
    env = MiniArcGame(level=7)
    env.reset()
    assert not env.is_done()

    # 1. Phase wall at (1, 3) blocks passage
    env.step("RIGHT")  # (1, 2)
    blocked = env.step("RIGHT")
    assert blocked.info["player"] == (1, 2)

    # Press SPACE to open phase wall
    env.step("SPACE")
    assert env.current_frame.grid[1][3] == 0

    # 2. Walk to Portal A at (1, 7)
    for _ in range(5):
        env.step("RIGHT")

    # Teleported to Portal B at (7, 1)!
    assert env.current_frame.info["player"] == (7, 1)

    # 3. Enter row 7 at (7, 2) and push box twice to the right onto target pad (7, 5)
    env.step("UP")     # (6, 1)
    env.step("RIGHT")  # (6, 2)
    env.step("DOWN")   # (7, 2)
    env.step("RIGHT")  # push Box from (7, 3) to (7, 4), player at (7, 3)
    env.step("RIGHT")  # push Box from (7, 4) to (7, 5) (Target Pad!), player at (7, 4)
    assert env.current_frame.grid[7][5] == 8

    # 4. Collect Yellow Key at (7, 7)
    env.step("UP")     # (6, 4)
    env.step("RIGHT")  # (6, 5)
    env.step("RIGHT")  # (6, 6)
    env.step("RIGHT")  # (6, 7)
    env.step("DOWN")   # (7, 7) Key collected!
    assert env.current_frame.info["has_key"] is True

    # 5. Return to Portal B at (7, 1)
    env.step("UP")     # (6, 7)
    for _ in range(6):
        env.step("LEFT")  # to (6, 1)
    env.step("DOWN")   # (7, 1) Portal B -> Teleported to Portal A at (1, 7)!
    assert env.current_frame.info["player"] == (1, 7)

    # 6. Walk back through open Phase Wall to vault entrance
    for _ in range(5):
        env.step("LEFT")  # to (1, 2)
    env.step("DOWN")   # (2, 2)
    door_frame = env.step("DOWN")  # (3, 2) Door unlocked!
    assert door_frame.info["player"] == (3, 2)

    # 7. Reach Goal at (4, 2)
    goal_frame = env.step("DOWN")  # (4, 2)
    assert goal_frame.info["player"] == (4, 2)
    assert env.is_done()
    assert env.history[-1].reward == 1.0

