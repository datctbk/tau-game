"""Memory management: Short-term history, World Model tracking, and Context Eviction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TurnRecord:
    """Record of a single agent interaction turn."""

    step: int
    observation_summary: str
    response: str
    code_executed: str | None
    repl_output: str | None
    actions_taken: list[str] = field(default_factory=list)
    previous_info: dict[str, Any] | None = None
    current_info: dict[str, Any] | None = None


class GameMemory:
    """Stores agent discoveries, hypotheses, and manages context eviction."""

    def __init__(self, max_turns_in_context: int = 15, max_chars_in_context: int = 25000) -> None:
        self.max_turns_in_context = max_turns_in_context
        self.max_chars_in_context = max_chars_in_context
        self.world_model: str = ""
        self.turns: list[TurnRecord] = []

    def update_world_model(self, text: str) -> None:
        """Extract and update world model insights from LLM response or user override."""
        # Check for explicit "World model:" or "Discovered mechanics:" block
        pattern = r"(?:World model|Discovered mechanics|Hypotheses|Findings)\s*:\s*(.*?)(?=(?:```|\Z|\n\n[A-Z]))"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            # Clean out any accidental code blocks from the extracted line
            extracted = re.sub(r"```.*?```", "", extracted, flags=re.DOTALL).strip()
            if len(extracted) > 5 and not extracted.startswith(("action(", "print(")):
                self.world_model = extracted
                return

        # Do NOT overwrite with raw code blocks or thought dumps if no explicit hypothesis was stated
        # This keeps the world model clean across turns.

    def record_turn(
        self,
        step: int,
        observation_summary: str,
        response: str,
        code: str | None,
        repl_output: str | None,
        actions: list[str],
        previous_info: dict[str, Any] | None = None,
        current_info: dict[str, Any] | None = None,
    ) -> None:
        """Append a completed turn record."""
        turn = TurnRecord(
            step=step,
            observation_summary=observation_summary,
            response=response,
            code_executed=code,
            repl_output=repl_output,
            actions_taken=actions,
            previous_info=previous_info,
            current_info=current_info,
        )
        self.turns.append(turn)
        self.update_world_model(response)

    def trim_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Perform context eviction.

        Guarantees:
        1. System prompt (messages[0]) is ALWAYS preserved.
        2. World model is maintained.
        3. Oldest interaction turns are pruned when exceeding turn limit or char limit.
        """
        if len(messages) <= 2:
            return list(messages)

        system_msg = messages[0]
        recent = messages[1:]

        # Turn count pruning (keep last max_turns * 2 messages: user + assistant pairs)
        max_msgs = self.max_turns_in_context * 2
        if len(recent) > max_msgs:
            recent = recent[-max_msgs:]

        # Character count pruning
        def total_chars(msgs: list[dict[str, str]]) -> int:
            return sum(len(m.get("content", "")) for m in msgs)

        while len(recent) > 2 and total_chars(recent) > self.max_chars_in_context:
            # Drop oldest user+assistant pair
            recent = recent[2:]

        return [system_msg] + recent
