"""Guards and invariants for Duck Harness / ARC-AGI-3 puzzle execution.

Includes:
- KnownNoOpActionError (BaseException): Refuses actions already proven to change nothing.
  Overridable by immediate repeat.
- RepeatedActionInStateError (BaseException): Loop breaker refusing repeated execution
  from the identical board state inside a single snippet.
- TerminalStateActionError (BaseException): Prevents executing further actions once the
  game or level has already reached a terminal state.
- KnownDeathActionError (BaseException): Prevents replaying fatal actions.
- NoopRepeatGuard: Tracks (level, state_hash, action_signature) and enforces the overridable
  single-slot contract.
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from typing import Any


class KnownNoOpActionError(BaseException):
    """Raised instead of executing an action already proven inert from this exact state.

    Inherits from BaseException so defensive `try: action(...) except Exception:`
    blocks written by LLMs cannot silently swallow the refusal.
    """


class KnownDeathActionError(BaseException):
    """Raised instead of executing an action that ended a prior attempt fatally."""


class RepeatedActionInStateError(BaseException):
    """Raised when a snippet attempts to execute an action from the identical state it has

    already visited in the same execution run, breaking infinite loops.
    """


class TerminalStateActionError(BaseException):
    """Raised when an action is called after a terminal condition (goal or game over) has been met."""


_DISPLAY_COORDS_RE = re.compile(
    r"^([A-Z0-9_]+)\(\s*(?:ROW\s*=\s*)?(-?\d+)\s*,\s*(?:COL\s*=\s*)?(-?\d+)\s*\)$"
)


def action_signature(action: Any) -> str:
    """Stable signature for an action.

    Normalizes strings, dicts, or tuples into a consistent uppercase representation.
    """
    if isinstance(action, dict):
        name = str(action.get("action") or action.get("name") or "").strip().upper()
        row = action.get("row")
        col = action.get("col")
        if row is not None or col is not None:
            return f"{name}({row},{col})"
        return name
    text = " ".join(str(action or "").split()).upper()
    match = _DISPLAY_COORDS_RE.match(text)
    if match:
        return f"{match.group(1)}({match.group(2)},{match.group(3)})"
    return text


def state_hash(grid: list[list[int]], border: int = 0) -> str:
    """Compute a stable hash identifying whether two boards are identical.

    If border > 0, crops outer edge cells (useful if border contains a timer or HUD bar).
    """
    rows = len(grid)
    cols = max((len(r) for r in grid), default=0)
    if border > 0 and rows > 2 * border and cols > 2 * border:
        core = [tuple(row[border : cols - border]) for row in grid[border : rows - border]]
    else:
        core = [tuple(row) for row in grid]
    return hashlib.sha1(repr(core).encode("utf-8")).hexdigest()


class NoopRepeatGuard:
    """Tracks known-inert actions per level with single-slot override support.

    Design choices:
    - Overridable: A blocked action arms a single slot. If the model insists by
      re-issuing that exact action again as its next move, it executes.
    - Level scoped: Clears states when the level changes.
    """

    def __init__(self, *, max_states: int = 512, max_actions_per_state: int = 16) -> None:
        self._max_states = max(1, int(max_states))
        self._max_actions = max(1, int(max_actions_per_state))
        self._states: OrderedDict[str, OrderedDict[str, None]] = OrderedDict()
        self._level: Any = None
        self._armed_noop: tuple[str, str] | None = None

    def note_level(self, level: Any) -> None:
        """Clear recorded states when level layout changes."""
        if level != self._level:
            self._level = level
            self._states.clear()
            self._armed_noop = None

    def observe(self, state_h: str, action_sig: str, *, gameplay_changed: bool) -> None:
        """Record whether an action changed the board."""
        if not state_h or not action_sig:
            return
        if not gameplay_changed:
            entry = self._states.get(state_h)
            if entry is None:
                entry = OrderedDict()
                self._states[state_h] = entry
            while len(self._states) > self._max_states:
                self._states.popitem(last=False)
            entry[action_sig] = None
            while len(entry) > self._max_actions:
                entry.popitem(last=False)
        else:
            # Action actually changed the board - remove it from no-op store if present
            entry = self._states.get(state_h)
            if entry is not None and action_sig in entry:
                del entry[action_sig]
                if not entry:
                    del self._states[state_h]

    def should_block(self, state_h: str, action_sig: str) -> bool:
        """Return True if this combination is a known no-op AND has not been insisted.

        Arms the single override slot when it blocks.
        """
        if not state_h or not action_sig:
            return False
        key = (state_h, action_sig)
        if self._armed_noop == key:
            # The model saw the refusal and insisted: allow it through once
            self._armed_noop = None
            return False

        entry = self._states.get(state_h)
        if not entry or action_sig not in entry:
            self._armed_noop = None
            return False

        self._armed_noop = key
        return True

    def note_executed(self, state_h: str, action_sig: str) -> None:
        """Clear armed override if an action other than the armed one executes."""
        key = (state_h, action_sig)
        if self._armed_noop and self._armed_noop != key:
            self._armed_noop = None

    def is_known_noop(self, state_h: str, action_sig: str) -> bool:
        """Check if an action is recorded as a known no-op."""
        entry = self._states.get(state_h)
        return bool(entry and action_sig in entry)
