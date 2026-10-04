"""Puzzle Replay Simulator: Turning historical discovery trees into empirical world models.

As formalized in Dream-RSI Section 3:
- Full recorded tree T_i remains fixed.
- Only the revealed portion T_revealed evolves as the policy makes decisions.
- Probing a batch of nodes deterministically reveals their recorded children without executing LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agent.discovery_tree import DiscoveryNode, DiscoveryTree


@dataclass
class CellMeta:
    """Structural metadata for a discovery node/cell in the tree."""

    cell_id: str
    branch: int
    attempt: int
    parent_id: str | None
    seq: int
    tags: list[str] = field(default_factory=list)


@dataclass
class Observation:
    """Prefix-observable attempt outcome matching Dream-RSI Listing 2 specification."""

    cell_id: str
    branch: int
    attempt: int
    score: float
    evaluated: bool
    valid: bool
    fail_class: str
    error: str | None
    delta_vs_baseline: float
    delta_vs_parent: float
    n_valid: int
    n_total: int
    node: DiscoveryNode

    @property
    def node_id(self) -> str:
        return self.cell_id

    @property
    def parent_id(self) -> str | None:
        return self.node.parent_id

    @property
    def depth(self) -> int:
        return self.node.depth

    @property
    def children_ids(self) -> list[str]:
        return self.node.children_ids

    @property
    def is_successful(self) -> bool:
        return self.error is None and self.fail_class == "ok"

    @property
    def is_repairable(self) -> bool:
        return self.fail_class == "repairable_code"


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

        # Precompute branch mappings for all nodes
        self._node_branch_map: dict[str, int] = {}
        self._node_attempt_map: dict[str, int] = {}
        self._compute_branch_structure()

        self.reset()

    def _compute_branch_structure(self) -> None:
        """Map every node to its root branch index and attempt index."""
        root = self.full_tree.nodes.get(self.full_tree.root_id)
        if not root:
            return

        self._node_branch_map[root.node_id] = 0
        self._node_attempt_map[root.node_id] = 0

        for branch_idx, child_id in enumerate(root.children_ids):
            # Traverse branch down
            curr_id = child_id
            attempt_idx = 0
            while curr_id and curr_id in self.full_tree.nodes:
                self._node_branch_map[curr_id] = branch_idx
                self._node_attempt_map[curr_id] = attempt_idx
                node = self.full_tree.nodes[curr_id]
                curr_id = node.children_ids[0] if node.children_ids else None
                attempt_idx += 1

    def meta(self, cell_id: str) -> CellMeta:
        """Return structural metadata for cell_id (Dream-RSI API)."""
        node = self.full_tree.nodes.get(cell_id)
        branch = self._node_branch_map.get(cell_id, 0)
        attempt = self._node_attempt_map.get(cell_id, 0)
        parent_id = node.parent_id if node else None
        seq = node.step if node else 0
        tags = [node.fail_class] if node else []
        return CellMeta(cell_id=cell_id, branch=branch, attempt=attempt, parent_id=parent_id, seq=seq, tags=tags)

    def _make_observation(self, node: DiscoveryNode) -> Observation:
        """Wrap DiscoveryNode into Dream-RSI Observation."""
        branch = self._node_branch_map.get(node.node_id, 0)
        attempt = self._node_attempt_map.get(node.node_id, 0)
        is_val = node.error is None and node.score > self.baseline_score
        return Observation(
            cell_id=node.node_id,
            branch=branch,
            attempt=attempt,
            score=node.score,
            evaluated=True,
            valid=is_val,
            fail_class=node.fail_class,
            error=node.error,
            delta_vs_baseline=node.score - self.baseline_score,
            delta_vs_parent=node.delta_vs_parent,
            n_valid=1 if is_val else 0,
            n_total=1,
            node=node,
        )

    def legal_roots(self) -> list[str]:
        """Return unopened branch roots from the root node (Dream-RSI API)."""
        root = self.full_tree.nodes.get(self.full_tree.root_id)
        if not root:
            return []
        already_revealed = self.revealed_children.get(root.node_id, set())
        unopened = [cid for cid in root.children_ids if cid not in already_revealed]
        return [root.node_id] if unopened else []

    def opened_branches(self) -> list[int]:
        """Return list of branch IDs that have been opened (Dream-RSI API)."""
        root = self.full_tree.nodes.get(self.full_tree.root_id)
        if not root:
            return []
        already_revealed = self.revealed_children.get(root.node_id, set())
        return [self._node_branch_map[cid] for cid in already_revealed if cid in self._node_branch_map]

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

    def observed_observations(self) -> dict[str, Observation]:
        """Prefix-only view returning structured Observation objects (Dream-RSI API)."""
        return {nid: self._make_observation(node) for nid, node in self.revealed_nodes.items()}

    def probe_batch(
        self, parent_ids: list[str], on_reveal: Callable[[Observation], None] | None = None
    ) -> list[DiscoveryNode]:
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

                if on_reveal:
                    obs = self._make_observation(child_node)
                    try:
                        on_reveal(obs)
                    except Exception:
                        pass

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
