"""Concrete game environments: GridWorld, MiniArcGame, and Arc3Adapter."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from env.environment import BaseEnvironment, Frame, Transition


class GridWorld(BaseEnvironment):
    """A 2D navigation grid world with obstacles and a goal.

    Colors:
      0: Empty (.)
      1: Player (B)
      3: Goal (G)
      5: Wall (#)
    """

    def __init__(
        self,
        height: int = 6,
        width: int = 6,
        player_start: tuple[int, int] = (1, 1),
        goal_pos: tuple[int, int] = (4, 4),
        walls: list[tuple[int, int]] | None = None,
    ) -> None:
        self.height = height
        self.width = width
        self.player_start = player_start
        self.goal_pos = goal_pos
        self.walls = set(walls) if walls is not None else {(2, 2), (2, 3), (3, 2)}

        self._player = player_start
        self._step_count = 0
        self._done = False
        self._history: list[Transition] = []
        self._current_frame: Frame | None = None
        self._previous_frame: Frame | None = None

        self.reset()

    def _render_grid(self) -> list[list[int]]:
        grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
        for wr, wc in self.walls:
            if 0 <= wr < self.height and 0 <= wc < self.width:
                grid[wr][wc] = 5  # Wall

        gr, gc = self.goal_pos
        if 0 <= gr < self.height and 0 <= gc < self.width:
            grid[gr][gc] = 3  # Goal

        pr, pc = self._player
        if 0 <= pr < self.height and 0 <= pc < self.width:
            grid[pr][pc] = 1  # Player

        return grid

    def reset(self) -> Frame:
        self._player = self.player_start
        self._step_count = 0
        self._done = False
        self._history.clear()
        self._previous_frame = None
        self._current_frame = Frame(
            grid=self._render_grid(),
            step=0,
            level=1,
            info={"player": self._player, "goal": self.goal_pos},
        )
        return self._current_frame

    def valid_actions(self) -> list[str]:
        if self._done:
            return ["RESET"]
        return ["UP", "DOWN", "LEFT", "RIGHT", "RESET"]

    def is_done(self) -> bool:
        return self._done

    @property
    def current_frame(self) -> Frame:
        if self._current_frame is None:
            return self.reset()
        return self._current_frame

    @property
    def previous_frame(self) -> Frame | None:
        return self._previous_frame

    @property
    def history(self) -> list[Transition]:
        return list(self._history)

    def step(self, action: str, **kwargs: Any) -> Frame:
        action = action.upper().strip()
        if action == "RESET":
            return self.reset()

        if self._done:
            return self.current_frame

        self._previous_frame = self.current_frame
        r, c = self._player

        dr, dc = 0, 0
        if action == "UP":
            dr = -1
        elif action == "DOWN":
            dr = 1
        elif action == "LEFT":
            dc = -1
        elif action == "RIGHT":
            dc = 1

        nr, nc = r + dr, c + dc

        # Check bounds and walls
        if 0 <= nr < self.height and 0 <= nc < self.width and (nr, nc) not in self.walls:
            self._player = (nr, nc)

        self._step_count += 1
        reward = 0.0

        if self._player == self.goal_pos:
            self._done = True
            reward = 1.0

        new_frame = Frame(
            grid=self._render_grid(),
            step=self._step_count,
            level=1,
            info={"player": self._player, "goal": self.goal_pos, "hit_wall": (nr, nc) in self.walls},
        )
        self._current_frame = new_frame

        transition = Transition(
            step=self._step_count,
            action=action,
            action_args=kwargs or None,
            previous_frame=self._previous_frame,
            current_frame=new_frame,
            reward=reward,
            done=self._done,
            info=new_frame.info,
        )
        self._history.append(transition)
        return new_frame


class MiniArcGame(BaseEnvironment):
    """ARC-style multi-mechanic puzzle environment.

    Includes key-door unlock, pushable objects, and mystery transformation.
    Colors:
      0: Background (.)
      1: Player (B)
      2: Red Key (R)
      3: Goal (G)
      4: Yellow Key (Y)
      5: Wall (#)
      6: Target Pad (M)
      7: Orange Door (O)
      8: Pushable Block (C)
      9: White/Maroon Portal or Gate (W)
    """

    def __init__(self, level: int = 1) -> None:
        self.initial_level = level
        self.level = level
        self._step_count = 0
        self._done = False
        self._history: list[Transition] = []
        self._current_frame: Frame | None = None
        self._previous_frame: Frame | None = None

        # Level state
        self._grid: list[list[int]] = []
        self._player: tuple[int, int] = (0, 0)
        self._has_key = False
        self._has_red_key = False
        self._block_pos: tuple[int, int] | None = None
        self._target_pos: tuple[int, int] | None = None
        self._target_pads: set[tuple[int, int]] = set()
        self._portals: dict[tuple[int, int], tuple[int, int]] = {}
        self._space_toggle = False

        self.reset()

    def _setup_level(self, level: int) -> None:
        self.level = level
        self._has_key = False
        self._has_red_key = False
        self._space_toggle = False
        self._block_pos = None
        self._target_pos = None
        self._target_pads.clear()
        self._portals.clear()

        if level == 1:
            # Level 1: Find Yellow Key (4) in left room to open Orange Door (7) and reach Green Goal (3)
            # Layout (7x7):
            self._grid = [
                [5, 5, 5, 5, 5, 5, 5],
                [5, 1, 0, 5, 0, 0, 5],
                [5, 0, 0, 5, 0, 0, 5],
                [5, 0, 0, 7, 0, 0, 5],
                [5, 0, 0, 5, 0, 0, 5],
                [5, 4, 0, 5, 0, 3, 5],
                [5, 5, 5, 5, 5, 5, 5],
            ]
            self._player = (1, 1)

        elif level == 2:
            # Level 2: Sokoban push Cyan Block (8) onto Magenta Target (6)
            # Layout (6x6):
            self._grid = [
                [5, 5, 5, 5, 5, 5],
                [5, 1, 0, 0, 0, 5],
                [5, 0, 8, 0, 0, 5],
                [5, 0, 0, 0, 0, 5],
                [5, 0, 0, 0, 6, 5],
                [5, 5, 5, 5, 5, 5],
            ]
            self._player = (1, 1)
            self._block_pos = (2, 2)
            self._target_pos = (4, 4)
            self._target_pads = {(4, 4)}

        elif level == 3:
            # Level 3: Press SPACE to toggle phase barrier and reach Green Goal (3)
            self._grid = [
                [5, 5, 5, 5, 5, 5],
                [5, 1, 0, 5, 0, 5],
                [5, 0, 0, 5, 0, 5],
                [5, 0, 0, 5, 0, 5],
                [5, 0, 0, 5, 3, 5],
                [5, 5, 5, 5, 5, 5],
            ]
            self._player = (1, 1)

        elif level == 4:
            # Level 4: Dual-Box Sokoban (7x7)
            # Push two Cyan blocks (8) onto two Magenta target pads (6)
            self._grid = [
                [5, 5, 5, 5, 5, 5, 5],
                [5, 1, 0, 0, 0, 0, 5],
                [5, 0, 8, 0, 8, 0, 5],
                [5, 0, 0, 5, 0, 0, 5],
                [5, 0, 6, 0, 6, 0, 5],
                [5, 0, 0, 0, 0, 0, 5],
                [5, 5, 5, 5, 5, 5, 5],
            ]
            self._player = (1, 1)
            self._target_pads = {(4, 2), (4, 4)}

        elif level == 5:
            # Level 5: Dual-Key Labyrinth & Phase Barrier (8x8)
            # Yellow Key (4) -> Door (7) -> Space Toggle Phase Barrier (4, 5) -> Red Key (2) -> Gate (9) -> Goal (3)
            self._grid = [
                [5, 5, 5, 5, 5, 5, 5, 5],
                [5, 1, 0, 5, 0, 0, 0, 5],
                [5, 4, 0, 7, 0, 0, 0, 5],
                [5, 0, 0, 5, 0, 0, 0, 5],
                [5, 5, 5, 5, 5, 5, 5, 5],
                [5, 0, 0, 9, 0, 0, 0, 5],
                [5, 3, 0, 5, 0, 0, 2, 5],
                [5, 5, 5, 5, 5, 5, 5, 5],
            ]
            self._player = (1, 1)

        elif level == 6:
            # Level 6: Wormhole Relay (9x9)
            # Portal A at (7, 1) <-> Portal B at (1, 7). Box (4, 6) blocking Key (7, 6). Goal at (1, 3).
            self._grid = [
                [5, 5, 5, 5, 5, 5, 5, 5, 5],
                [5, 1, 7, 3, 5, 0, 0, 9, 5],
                [5, 0, 5, 5, 5, 0, 0, 0, 5],
                [5, 0, 0, 0, 5, 0, 5, 0, 5],
                [5, 0, 5, 0, 5, 0, 8, 0, 5],
                [5, 0, 0, 0, 5, 0, 0, 0, 5],
                [5, 0, 0, 0, 5, 5, 0, 5, 5],
                [5, 9, 0, 0, 5, 5, 4, 5, 5],
                [5, 5, 5, 5, 5, 5, 5, 5, 5],
            ]
            self._player = (1, 1)
            self._portals = {(7, 1): (1, 7), (1, 7): (7, 1)}

        else:
            # Level 7: The Quantum Citadel (9x9)
            # Phase barrier (1, 3) + Portal A (1, 7) <-> Portal B (7, 1) + Box (7, 3) onto Target (7, 5) + Key (7, 7)
            # Goal (4, 2) in vault sealed by Door (3, 2)
            self._grid = [
                [5, 5, 5, 5, 5, 5, 5, 5, 5],
                [5, 1, 0, 5, 0, 0, 0, 9, 5],
                [5, 0, 0, 5, 0, 0, 0, 0, 5],
                [5, 5, 7, 5, 0, 0, 0, 0, 5],
                [5, 5, 3, 5, 0, 0, 0, 0, 5],
                [5, 5, 5, 5, 0, 0, 0, 0, 5],
                [5, 0, 0, 0, 0, 0, 0, 0, 5],
                [5, 9, 0, 8, 0, 6, 0, 4, 5],
                [5, 5, 5, 5, 5, 5, 5, 5, 5],
            ]
            self._player = (1, 1)
            self._portals = {(1, 7): (7, 1), (7, 1): (1, 7)}
            self._target_pads = {(7, 5)}

    def _restore_cell(self, r: int, c: int) -> None:
        """Restore cell when player or entity vacates."""
        if (r, c) in self._target_pads:
            self._grid[r][c] = 6
        elif (r, c) in self._portals:
            self._grid[r][c] = 9
        else:
            self._grid[r][c] = 0

    def reset(self) -> Frame:
        self._step_count = 0
        self._done = False
        self._history.clear()
        self._previous_frame = None
        self._setup_level(self.initial_level)

        self._current_frame = Frame(
            grid=deepcopy(self._grid),
            step=0,
            level=self.level,
            info={
                "has_key": self._has_key,
                "has_red_key": self._has_red_key,
                "player": self._player,
                "space_toggle": self._space_toggle,
                "done": self._done,
            },
        )
        return self._current_frame

    def valid_actions(self) -> list[str]:
        if self._done:
            return ["RESET"]
        return ["UP", "DOWN", "LEFT", "RIGHT", "SPACE", "RESET"]

    def is_done(self) -> bool:
        return self._done

    @property
    def current_frame(self) -> Frame:
        if self._current_frame is None:
            return self.reset()
        return self._current_frame

    @property
    def previous_frame(self) -> Frame | None:
        return self._previous_frame

    @property
    def history(self) -> list[Transition]:
        return list(self._history)

    def step(self, action: str, **kwargs: Any) -> Frame:
        action = action.upper().strip()
        if action == "RESET":
            return self.reset()

        if self._done:
            return self.current_frame

        self._previous_frame = self.current_frame
        self._step_count += 1
        reward = 0.0

        r, c = self._player
        dr, dc = 0, 0

        if action == "UP":
            dr = -1
        elif action == "DOWN":
            dr = 1
        elif action == "LEFT":
            dc = -1
        elif action == "RIGHT":
            dc = 1
        elif action == "SPACE":
            # Space toggle mechanic
            self._space_toggle = not self._space_toggle
            if self.level == 3:
                self._grid[3][3] = 0 if self._space_toggle else 5
            elif self.level == 5:
                if self._player != (4, 5):
                    self._grid[4][5] = 0 if self._space_toggle else 5
            elif self.level == 7:
                if self._player != (1, 3):
                    self._grid[1][3] = 0 if self._space_toggle else 5

        if dr != 0 or dc != 0:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(self._grid) and 0 <= nc < len(self._grid[0]):
                target_cell = self._grid[nr][nc]

                if target_cell in (0, 6):  # empty space or target pad
                    self._restore_cell(r, c)
                    self._player = (nr, nc)
                    self._grid[nr][nc] = 1

                elif target_cell == 4:  # collect Yellow Key
                    self._has_key = True
                    self._restore_cell(r, c)
                    self._player = (nr, nc)
                    self._grid[nr][nc] = 1

                elif target_cell == 2:  # collect Red Key
                    self._has_red_key = True
                    self._restore_cell(r, c)
                    self._player = (nr, nc)
                    self._grid[nr][nc] = 1

                elif target_cell == 7:  # Orange Door
                    if self._has_key:
                        self._restore_cell(r, c)
                        self._player = (nr, nc)
                        self._grid[nr][nc] = 1

                elif target_cell == 9:  # Portal or Red Gate
                    if (nr, nc) in self._portals:
                        # Wormhole teleportation!
                        dest_r, dest_c = self._portals[(nr, nc)]
                        if self._grid[dest_r][dest_c] in (0, 6, 9):
                            self._restore_cell(r, c)
                            self._grid[nr][nc] = 9  # Entry portal remains active
                            self._player = (dest_r, dest_c)
                            self._grid[dest_r][dest_c] = 1
                    elif self._has_red_key:  # Red Gate unlocked
                        self._restore_cell(r, c)
                        self._player = (nr, nc)
                        self._grid[nr][nc] = 1

                elif target_cell == 8:  # Pushable block
                    nnr, nnc = nr + dr, nc + dc
                    if 0 <= nnr < len(self._grid) and 0 <= nnc < len(self._grid[0]):
                        if self._grid[nnr][nnc] in (0, 6):  # empty or target pad
                            self._grid[nnr][nnc] = 8
                            self._block_pos = (nnr, nnc)
                            self._restore_cell(r, c)
                            self._player = (nr, nc)
                            self._grid[nr][nc] = 1

                            # Check target pads win condition
                            if self._target_pads and all(self._grid[tr][tc] == 8 for tr, tc in self._target_pads):
                                has_goal = any(3 in row for row in self._grid)
                                if not has_goal:
                                    self._done = True
                                    reward = 2.0

                elif target_cell == 3:  # Green Goal!
                    self._restore_cell(r, c)
                    self._player = (nr, nc)
                    self._grid[nr][nc] = 1
                    self._done = True
                    reward = 1.0

        new_frame = Frame(
            grid=deepcopy(self._grid),
            step=self._step_count,
            level=self.level,
            info={
                "player": self._player,
                "has_key": self._has_key,
                "has_red_key": self._has_red_key,
                "space_toggle": self._space_toggle,
                "done": self._done,
            },
        )
        self._current_frame = new_frame

        transition = Transition(
            step=self._step_count,
            action=action,
            action_args=kwargs or None,
            previous_frame=self._previous_frame,
            current_frame=new_frame,
            reward=reward,
            done=self._done,
            info=new_frame.info,
        )
        self._history.append(transition)
        return new_frame


class Arc3Adapter(BaseEnvironment):
    """Generic adapter for external ARC-AGI-3 competition APIs or TAAF game servers."""

    def __init__(self, raw_env: Any = None) -> None:
        self.raw_env = raw_env
        self._history: list[Transition] = []
        self._current_frame: Frame | None = None
        self._previous_frame: Frame | None = None
        self._step_count = 0
        self._done = False

    def reset(self) -> Frame:
        self._step_count = 0
        self._done = False
        self._history.clear()
        self._previous_frame = None

        if self.raw_env and hasattr(self.raw_env, "reset"):
            payload = self.raw_env.reset()
            grid = payload.get("grid", [[0]]) if isinstance(payload, dict) else payload
            level = payload.get("level", 1) if isinstance(payload, dict) else 1
        else:
            grid = [[0]]
            level = 1

        self._current_frame = Frame(grid=grid, step=0, level=level)
        return self._current_frame

    def valid_actions(self) -> list[str]:
        if self.raw_env and hasattr(self.raw_env, "valid_actions"):
            return list(self.raw_env.valid_actions())
        return ["UP", "DOWN", "LEFT", "RIGHT", "SPACE", "RESET"]

    def is_done(self) -> bool:
        if self.raw_env and hasattr(self.raw_env, "is_done"):
            return bool(self.raw_env.is_done())
        return self._done

    @property
    def current_frame(self) -> Frame:
        if self._current_frame is None:
            return self.reset()
        return self._current_frame

    @property
    def previous_frame(self) -> Frame | None:
        return self._previous_frame

    @property
    def history(self) -> list[Transition]:
        return list(self._history)

    def step(self, action: str, **kwargs: Any) -> Frame:
        self._previous_frame = self.current_frame
        self._step_count += 1

        if self.raw_env and hasattr(self.raw_env, "step"):
            obs = self.raw_env.step(action, **kwargs)
            grid = obs.get("grid", [[0]]) if isinstance(obs, dict) else obs
            self._done = bool(obs.get("done", False)) if isinstance(obs, dict) else False
            reward = float(obs.get("reward", 0.0)) if isinstance(obs, dict) else 0.0
        else:
            grid = deepcopy(self.current_frame.grid)
            reward = 0.0

        self._current_frame = Frame(grid=grid, step=self._step_count, level=self.current_frame.level)
        self._history.append(
            Transition(
                step=self._step_count,
                action=action,
                action_args=kwargs or None,
                previous_frame=self._previous_frame,
                current_frame=self._current_frame,
                reward=reward,
                done=self._done,
            )
        )
        return self._current_frame
