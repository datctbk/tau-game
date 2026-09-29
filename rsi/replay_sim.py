"""Puzzle Replay Simulator: Turning historical discovery trees into empirical world models.

As formalized in Dream-RSI Section 3:
- Full recorded tree T_i remains fixed.
- Only the revealed portion T_revealed evolves as the policy makes decisions.
- Probing a batch of nodes deterministically reveals their recorded children without executing LLM calls.
"""

from __future__ import annotations

from typing import Any

from agent.discovery_tree import DiscoveryNode, DiscoveryTree


class PuzzleReplaySimulator:
    """Replay environment over an existing DiscoveryTree."""

    def __init__(
        self,
        tree: DiscoveryTree,
        baseline_score: float = 0.0,
        max_parallelism: int = 3,
        max_rounds: int = 20,
    ) -> None:
        self.full_tree = tree
        self.baseline_score = baseline_score
        self.max_parallelism = max_parallelism
        self.max_rounds = max_rounds

        self.revealed_nodes: dict[str, DiscoveryNode] = {}
        self.revealed_children: dict[str, set[str]] = {}
        self.current_round: int = 0
        self.total_probes: int = 0

        self.reset()

    def reset(self) -> dict[str, DiscoveryNode]:
        """Reset replay state to only the root node."""
        self.revealed_nodes.clear()
        self.revealed_children.clear()
        self.current_round = 0
        self.total_probes = 0

        root = self.full_tree.nodes[self.full_tree.root_id]
        self.revealed_nodes[root.node_id] = root
        self.revealed_children[root.node_id] = set()

        return dict(self.revealed_nodes)

    def observed(self) -> dict[str, DiscoveryNode]:
        """Prefix-only view of revealed nodes."""
        return dict(self.revealed_nodes)

    def legal_actions(self) -> list[str]:
        """Return revealed nodes that have unrevealed children in the full tree."""
        legal: list[str] = []
        for nid, node in self.revealed_nodes.items():
            full_node = self.full_tree.nodes.get(nid)
            if not full_node:
                continue
            # Check if there are children in full_tree that haven't been revealed yet
            already_revealed = self.revealed_children.get(nid, set())
            unrevealed = [cid for cid in full_node.children_ids if cid not in already_revealed]
            if unrevealed:
                legal.append(nid)
        return legal

    def probe_batch(self, parent_ids: list[str]) -> list[DiscoveryNode]:
        """Reveal one unrevealed recorded child for each parent in the batch."""
        if not parent_ids:
            return []

        # Enforce parallelism cap
        batch = parent_ids[: self.max_parallelism]
        newly_revealed: list[DiscoveryNode] = []

        for pid in batch:
            full_node = self.full_tree.nodes.get(pid)
            if not full_node:
                continue

            already_revealed = self.revealed_children.setdefault(pid, set())
            unrevealed = [cid for cid in full_node.children_ids if cid not in already_revealed]

            if unrevealed:
                # Reveal the earliest unrevealed child
                child_id = unrevealed[0]
                child_node = self.full_tree.nodes[child_id]
                already_revealed.add(child_id)
                self.revealed_nodes[child_id] = child_node
                self.revealed_children.setdefault(child_id, set())
                newly_revealed.append(child_node)
                self.total_probes += 1

        self.current_round += 1
        return newly_revealed

    def best_revealed_score(self) -> float:
        """Best score achieved across all revealed nodes."""
        if not self.revealed_nodes:
            return self.baseline_score
        return max(n.score for n in self.revealed_nodes.values())

    def is_finished(self) -> bool:
        """True if round limit reached or no legal actions remain."""
        if self.current_round >= self.max_rounds:
            return True
        return len(self.legal_actions()) == 0
