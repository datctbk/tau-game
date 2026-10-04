"""Prefix-only meta-exploration policy for Dream-RSI.

Adheres strictly to Dream-RSI constraints:
- Prefix-only: Decides solely using revealed observations and prefix trajectories.
- Dynamic portfolio batching: Exploitation, Exploration, and Recovery.
- Single scalar beta: Configured once per run, routing patience and pruning through a schedule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.discovery_tree import DiscoveryNode


@dataclass
class GridPlan:
    """Next-cycle exploration grid plan (width vs depth) decided from history."""

    branch_count: int
    refine_count: int
    reason: str = ""


@dataclass
class GridPlanningContext:
    """Historical context provided to plan_grid before creating a new live grid."""

    history: list[dict[str, Any]] = field(default_factory=list)
    hard_max_branch_count: int = 10
    hard_max_refine_count: int = 20
    fallback_branch_count: int = 3
    fallback_refine_count: int = 6
    worker_cap: int = 3


@dataclass
class BranchTrajectory:
    """Reconstructed prefix trajectory for an opened branch in the discovery tree."""

    branch_id: str
    nodes: list[DiscoveryNode]
    successful_anchor: DiscoveryNode | None
    score_trend: float
    regressions: int
    repair_sequence: list[str]
    explored_depth: int


def adapt_default_beta(
    prior_beta: float,
    live_best_improving: bool,
    plateaued: bool,
    sweep_results: list[dict[str, Any]] | None = None,
) -> float:
    """Cross-cycle default-beta adaptation rule from Dream-RSI Section B.2.
    
    Rules:
    1. Live best is still improving: keep prior default beta unless sweep clearly favors a nearby one.
    2. Live best has plateaued, and higher beta reaches higher attainment: raise default by small step (0.1-0.2).
    3. High default beta tried through plateau and adds work without higher attainment: lower by small step.
    4. Insufficient history or conflicting evidence: use moderately exploratory default (0.6).
    """
    if sweep_results is None or len(sweep_results) == 0:
        return 0.6

    if live_best_improving:
        return prior_beta

    if plateaued:
        # Check if higher beta achieves strictly higher attainment in sweep
        higher_betas = [r for r in sweep_results if r.get("beta", 0) > prior_beta]
        prior_res = next((r for r in sweep_results if abs(r.get("beta", 0) - prior_beta) < 0.05), None)
        prior_score = prior_res.get("mean_score", 0.0) if prior_res else 0.0

        higher_improved = any(r.get("mean_score", 0.0) > prior_score + 1e-4 for r in higher_betas)
        if higher_improved and prior_beta < 0.9:
            return round(min(1.0, prior_beta + 0.15), 2)
        elif prior_beta >= 0.7:
            return round(max(0.2, prior_beta - 0.15), 2)

    return prior_beta


class ObservationSignal:
    """Helper signals derived from prefix-revealed nodes."""

    @staticmethod
    def branch_promising(node: Any, baseline_score: float = 0.0) -> bool:
        """True if the node improved score over baseline or shows positive trend."""
        return getattr(node, "score", 0.0) > baseline_score or getattr(node, "delta_vs_parent", 0.0) > 0.0

    @staticmethod
    def branch_failed_hard(node: Any) -> bool:
        """True if the node has an unrecoverable failure or terminal loss."""
        fail_class = getattr(node, "fail_class", "ok")
        done = getattr(node, "done", False)
        score = getattr(node, "score", 0.0)
        return fail_class == "dead_end" or (done and score <= 0.0)

    @staticmethod
    def is_repairable(node: Any) -> bool:
        """True if failure is an implementation/code error (Syntax, Index, Name error)."""
        return getattr(node, "fail_class", "ok") == "repairable_code"


class OptimalPolicy:
    """Prefix-only exploration policy orchestrator implementing Dream-RSI principles."""

    NAME = "OptimalPolicy"

    def __init__(self, beta: float = 0.5, max_parallelism: int = 3, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        configured_beta = self.config.get("beta", beta)
        self.beta = float(max(0.01, min(1.0, configured_beta)))
        self.max_parallelism = max(1, max_parallelism)
        self.schedule = self._schedule(self.beta)

    def plan_grid(self, context: GridPlanningContext) -> GridPlan:
        """Required next-cycle grid planning (Dream-RSI Listing 2, lines 190-243).
        
        Chooses width (branch_count) vs depth (refine_count) from historical evidence:
        - Roots improve early while deeper refinements stall -> widen, reduce depth.
        - High gains arrive late on small set of directions -> narrow, increase depth.
        - Plateaued directions after sufficient depth -> widen.
        - Repeated hard failures -> reduce width and depth conservatively.
        - Insufficient history -> conservative bootstrap plan.
        """
        history = context.history
        if not history:
            return GridPlan(
                branch_count=context.fallback_branch_count,
                refine_count=context.fallback_refine_count,
                reason="Bootstrap plan: insufficient history to infer depth vs width preference.",
            )

        # Inspect latest cycle outcomes
        latest = history[-1]
        solved = latest.get("online_solved", False)
        steps = latest.get("online_steps", 0)
        discovered = latest.get("discovered_nodes", 0)

        # Evidence: Quick solve with few steps -> High depth efficiency, maintain balanced plan
        if solved and steps <= context.fallback_refine_count:
            return GridPlan(
                branch_count=min(context.hard_max_branch_count, context.fallback_branch_count + 1),
                refine_count=context.fallback_refine_count,
                reason="Policy solved quickly; expanding width for diversity while holding depth.",
            )

        # Evidence: Many attempts explored but plateaued -> expand width
        if discovered > 15 and not solved:
            return GridPlan(
                branch_count=min(context.hard_max_branch_count, context.fallback_branch_count + 2),
                refine_count=max(2, context.fallback_refine_count - 1),
                reason="Exploration plateaued on current directions; widening search to unblock new mechanics.",
            )

        # Default evidence-based conservative allocation
        return GridPlan(
            branch_count=context.fallback_branch_count,
            refine_count=context.fallback_refine_count,
            reason="Balanced plan maintained according to recent cycle progression.",
        )

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
