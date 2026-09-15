"""Base environment definitions and core types for tau-game."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Frame:
    """Represents a single observation frame in a game."""

    grid: list[list[int]]
    step: int = 0
    level: int = 1
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        """Return (height, width) of the grid."""
        if not self.grid or not self.grid[0]:
            return (0, 0)
        return (len(self.grid), len(self.grid[0]))

    @property
    def ascii(self) -> str:
        """Convenient ASCII representation of the grid."""
        from perception.segmentation import format_grid_ascii

        return format_grid_ascii(self.grid)

    @property
    def segmentation(self) -> list[dict[str, Any]]:
        """4-connected components segmentation of the current grid."""
        from perception.segmentation import connected_components

        return [node.to_dict() for node in connected_components(self.grid)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid": self.grid,
            "step": self.step,
            "level": self.level,
            "shape": self.shape,
            "info": self.info,
        }


@dataclass
class Transition:
    """Record of an action transition between states."""

    step: int
    action: str
    action_args: dict[str, Any] | None
    previous_frame: Frame
    current_frame: Frame
    reward: float = 0.0
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action,
            "reward": self.reward,
            "done": self.done,
            "info": self.info,
        }


class BaseEnvironment(ABC):
    """Abstract base class for all game environments in tau-game."""

    @abstractmethod
    def reset(self) -> Frame:
        """Reset the environment to initial state and return the first frame."""
        pass

    @abstractmethod
    def step(self, action: str, **kwargs: Any) -> Frame:
        """Take an action in the environment and return the new frame."""
        pass

    @abstractmethod
    def valid_actions(self) -> list[str]:
        """Return the list of currently valid action names."""
        pass

    @abstractmethod
    def is_done(self) -> bool:
        """Return whether the current game / level is completed or terminated."""
        pass

    @property
    @abstractmethod
    def current_frame(self) -> Frame:
        """Return the current frame observation."""
        pass

    @property
    @abstractmethod
    def previous_frame(self) -> Frame | None:
        """Return the frame prior to the last action."""
        pass

    @property
    @abstractmethod
    def history(self) -> list[Transition]:
        """Return the full history of transitions in this episode."""
        pass
