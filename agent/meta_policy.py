"""Prefix-only meta-exploration policy for Dream-RSI.

Adheres strictly to Dream-RSI constraints:
- Prefix-only: Decides solely using revealed observations and prefix trajectories.
- Dynamic portfolio batching: Exploitation, Exploration, and Recovery.
- Single scalar beta: Configured once per run, routing patience and pruning through a schedule.
"""

from __future__ import annotations

import math
from typing import Any

from agent.discovery_tree import DiscoveryNode


class ObservationSignal:
    """Helper signals derived from prefix-revealed nodes."""

    @staticmethod
    def branch_promising(node: DiscoveryNode, baseline_score: float = 0.0) -> bool:
        """True if the node improved score over baseline or shows positive trend."""
        return node.score > baseline_score or node.delta_vs_parent > 0.0

    @staticmethod
    def branch_failed_hard(node: DiscoveryNode) -> bool:
        """True if the node has an unrecoverable failure or terminal loss."""
        return node.fail_class == "dead_end" or (node.done and node.score <= 0.0)

    @staticmethod
    def is_repairable(node: DiscoveryNode) -> bool:
        """True if failure is an implementation/code error (Syntax, Index, Name error)."""
        return node.fail_class == "repairable_code"


class OptimalPolicy:
    """Prefix-only exploration policy orchestrator implementing Dream-RSI principles."""

    NAME = "OptimalPolicy"

    def __init__(self, beta: float = 0.5, max_parallelism: int = 3) -> None:
        self.beta = float(max(0.01, min(1.0, beta)))
        self.max_parallelism = max(1, max_parallelism)
        self.schedule = self._schedule(self.beta)

    def _schedule(self, beta: float) -> dict[str, Any]:
        """Map single scalar beta into behavioral thresholds.
        
        High beta (e.g. 0.8): More width, deeper patience, weaker pruning, high exploration.
        Low beta (e.g. 0.2): Selective exploitation, strict pruning, earlier stop.
        """
        return {
            "depth_patience": max(2, int(2 + 8 * beta)),
            "exploration_ratio": 0.2 + 0.5 * beta,
            "min_delta_to_continue": 0.05 * (1.0 - beta),
            "allow_recovery": beta >= 0.2,
            "max_stagnation_rounds": max(2, int(2 + 6 * beta)),
        }

    def rank_candidates(
        self,
        observed: dict[str, DiscoveryNode],
        legal_nodes: list[str],
        baseline_score: float = 0.0,
    ) -> tuple[list[str], list[str], list[str]]:
        """Rank legal nodes into Exploitation, Exploration, and Recovery queues."""
        exploitation: list[tuple[float, str]] = []
        exploration: list[tuple[float, str]] = []
        recovery: list[tuple[float, str]] = []

        for nid in legal_nodes:
            node = observed.get(nid)
            if not node:
                continue

            # Hard dead-ends are skipped
            if ObservationSignal.branch_failed_hard(node):
                continue

            # Check for repairable code errors
            if ObservationSignal.is_repairable(node):
                if self.schedule["allow_recovery"]:
                    recovery_priority = node.score - (0.1 * node.depth)
                    recovery.append((recovery_priority, nid))
                continue

            # If node is promising or improved vs parent
            if ObservationSignal.branch_promising(node, baseline_score):
                # Priority: high score + positive delta - slight depth penalty
                prio = node.score + 0.5 * max(0.0, node.delta_vs_parent) - 0.05 * node.depth
                exploitation.append((prio, nid))
            else:
                # Unopened root or underexplored frontier
                prio = 1.0 / (1.0 + node.depth)
                exploration.append((prio, nid))

        # Sort descending by priority
        exploitation.sort(key=lambda x: x[0], reverse=True)
        exploration.sort(key=lambda x: x[0], reverse=True)
        recovery.sort(key=lambda x: x[0], reverse=True)

        return (
            [nid for _, nid in exploitation],
            [nid for _, nid in exploration],
            [nid for _, nid in recovery],
        )

    def select_batch(
        self,
        observed: dict[str, DiscoveryNode],
        legal_nodes: list[str],
        max_parallelism: int | None = None,
        baseline_score: float = 0.0,
    ) -> list[str]:
        """Build dynamic portfolio batch up to max_parallelism slots."""
        cap = max_parallelism or self.max_parallelism
        if not legal_nodes or cap <= 0:
            return []

        exploit_queue, explore_queue, recovery_queue = self.rank_candidates(
            observed, legal_nodes, baseline_score
        )

        batch: list[str] = []
        seen: set[str] = set()

        def add_candidate(nid: str) -> bool:
            if nid not in seen and len(batch) < cap:
                batch.append(nid)
                seen.add(nid)
                return True
            return False

        # 1. Recovery allocation: at most 1 slot reserved for repairable error
        if recovery_queue and self.schedule["allow_recovery"]:
            add_candidate(recovery_queue[0])

        # 2. Exploitation: feed the top promising refinement
        if exploit_queue:
            add_candidate(exploit_queue[0])

        # 3. Exploration: ensure diversity if exploration slots available
        if explore_queue and len(batch) < cap:
            add_candidate(explore_queue[0])

        # 4. Fill remaining slots with highest priority available from exploitation or exploration
        combined_remaining = exploit_queue[1:] + explore_queue[1:]
        for nid in combined_remaining:
            if len(batch) >= cap:
                break
            add_candidate(nid)

        return batch
