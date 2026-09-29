"""Discovery Tree: Structured search graph recording puzzle exploration history.

In Dream-RSI, the discovery tree preserves the full exploration DAG of actions,
REPL executions, intermediate frames, scores, and failure classifications.
Once recorded, it serves as a reusable replay simulator (World Model).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from env.environment import Frame


@dataclass
class DiscoveryNode:
    """A single decision/attempt node in the discovery tree."""

    node_id: str
    parent_id: str | None
    depth: int
    step: int
    actions: list[str]
    code: str | None = None
    repl_output: str | None = None
    frame_grid: list[list[int]] = field(default_factory=list)
    score: float = 0.0
    delta_vs_parent: float = 0.0
    done: bool = False
    fail_class: str = "ok"  # "ok" | "repairable_code" | "blocked_mechanic" | "dead_end"
    error: str | None = None
    children_ids: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def is_successful(self) -> bool:
        """Dream-RSI success semantics: error is None and fail_class == 'ok'."""
        return self.error is None and self.fail_class == "ok"

    @property
    def is_repairable(self) -> bool:
        """Node failed due to execution/code error but parent direction may be viable."""
        return self.fail_class == "repairable_code"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiscoveryNode:
        return cls(**data)


class DiscoveryTree:
    """Manages the full exploration tree of a game/puzzle episode."""

    def __init__(self, root_id: str = "node_0", root_frame: Frame | None = None) -> None:
        self.root_id = root_id
        self.nodes: dict[str, DiscoveryNode] = {}
        self._next_id = 0

        if root_frame is not None:
            self.create_root(root_frame)

    def generate_node_id(self) -> str:
        uid = f"node_{self._next_id}"
        self._next_id += 1
        return uid

    def create_root(self, frame: Frame, score: float = 0.0) -> DiscoveryNode:
        """Initialize the root node of the puzzle."""
        root = DiscoveryNode(
            node_id=self.root_id,
            parent_id=None,
            depth=0,
            step=0,
            actions=[],
            frame_grid=frame.grid,
            score=score,
            delta_vs_parent=0.0,
            done=False,
            fail_class="ok",
            error=None,
            info=dict(frame.info or {}),
        )
        self.nodes[self.root_id] = root
        self._next_id = 1
        return root

    def add_attempt(
        self,
        parent_id: str,
        actions: list[str],
        frame: Frame,
        code: str | None = None,
        repl_output: str | None = None,
        score: float = 0.0,
        done: bool = False,
        error: str | None = None,
        fail_class: str | None = None,
    ) -> DiscoveryNode:
        """Add an exploration attempt branching from parent_id."""
        if parent_id not in self.nodes:
            raise ValueError(f"Parent node '{parent_id}' does not exist in discovery tree.")

        parent = self.nodes[parent_id]
        node_id = self.generate_node_id()

        # Automatic failure classification if not explicitly provided
        if fail_class is None:
            if error is not None:
                if any(err in str(error) for err in ["SyntaxError", "NameError", "TypeError", "IndexError"]):
                    fail_class = "repairable_code"
                else:
                    fail_class = "dead_end"
            elif frame.info.get("hit_wall", False):
                fail_class = "blocked_mechanic"
            elif done and score <= 0.0:
                fail_class = "dead_end"
            else:
                fail_class = "ok"

        delta = score - parent.score

        node = DiscoveryNode(
            node_id=node_id,
            parent_id=parent_id,
            depth=parent.depth + 1,
            step=parent.step + len(actions),
            actions=actions,
            code=code,
            repl_output=repl_output,
            frame_grid=frame.grid,
            score=score,
            delta_vs_parent=delta,
            done=done,
            fail_class=fail_class,
            error=error,
            info=dict(frame.info or {}),
        )

        parent.children_ids.append(node_id)
        self.nodes[node_id] = node
        return node

    def get_leaves(self) -> list[str]:
        """Return all leaf node IDs in the tree."""
        return [node_id for node_id, node in self.nodes.items() if not node.children_ids]

    def get_trajectory(self, node_id: str) -> list[DiscoveryNode]:
        """Reconstruct the path from root to the given node."""
        path: list[DiscoveryNode] = []
        curr: str | None = node_id
        while curr is not None and curr in self.nodes:
            node = self.nodes[curr]
            path.append(node)
            curr = node.parent_id
        path.reverse()
        return path

    def best_node(self) -> DiscoveryNode:
        """Return the highest-scoring node found so far."""
        if not self.nodes:
            raise ValueError("Tree has no nodes.")
        return max(self.nodes.values(), key=lambda n: n.score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "next_id": self._next_id,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiscoveryTree:
        tree = cls(root_id=data["root_id"])
        tree._next_id = data.get("next_id", len(data["nodes"]))
        tree.nodes = {nid: DiscoveryNode.from_dict(ndata) for nid, ndata in data["nodes"].items()}
        return tree

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> DiscoveryTree:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
