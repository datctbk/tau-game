"""Dream-RSI Closed Loop Orchestrator for tau-game.

Alternates between:
1. Online Exploration: Guides DuckAgent with current policy on game levels, collecting DiscoveryTrees.
2. Simulator Construction: Appends new discovery trees to the replay world pool.
3. Offline Dreaming: Evaluates candidate policies over historical trees at zero cost,
   refines exploration parameters/policy, and redeploys for the next round.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agent.discovery_tree import DiscoveryTree
from agent.meta_policy import OptimalPolicy
from rsi.dreaming import DreamingOptimizer, DreamingResult
from rsi.replay_sim import PuzzleReplaySimulator

logger = logging.getLogger(__name__)


@dataclass
class RSICycleResult:
    """Outcome of one complete Dream-RSI online + offline cycle."""

    iteration: int
    online_solved: bool
    online_steps: int
    online_reward: float
    tree_size: int
    discovered_nodes: int
    dreaming_result: DreamingResult | None = None
    policy_beta: float = 0.5


class DreamRSIOrchestrator:
    """Manages the recursive self-improvement discovery loop for game puzzles."""

    def __init__(
        self,
        env_factory: Callable[[], Any],
        initial_beta: float = 0.5,
        max_parallelism: int = 3,
        artifact_dir: str | Path = "traces/dream_rsi",
    ) -> None:
        self.env_factory = env_factory
        self.current_policy = OptimalPolicy(beta=initial_beta, max_parallelism=max_parallelism)
        self.history_pool: list[DiscoveryTree] = []
        self.cycles: list[RSICycleResult] = []
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def run_online_round(self, agent_runner: Callable[[Any, OptimalPolicy], DiscoveryTree]) -> DiscoveryTree:
        """Run one online episode guided by current_policy and return its DiscoveryTree."""
        env = self.env_factory()
        tree = agent_runner(env, self.current_policy)
        self.history_pool.append(tree)

        # Save tree artifact
        tree_idx = len(self.history_pool)
        save_path = self.artifact_dir / f"tree_round_{tree_idx:03d}.json"
        tree.save(save_path)
        logger.info("Saved discovery tree with %d nodes to %s", len(tree.nodes), save_path)

        return tree

    def run_offline_dreaming(
        self, candidate_betas: list[float] | None = None
    ) -> DreamingResult:
        """Run offline dreaming across all accumulated history worlds."""
        if not self.history_pool:
            raise ValueError("Cannot dream with an empty history pool.")

        optimizer = DreamingOptimizer(
            trees=self.history_pool,
            max_parallelism=self.current_policy.max_parallelism,
        )
        dream_res = optimizer.optimize(candidate_betas=candidate_betas)

        # Deploy winning policy
        self.current_policy = dream_res.best_policy
        logger.info(
            "Offline Dreaming complete! Selected beta: %.2f (Pareto reward: %.4f)",
            dream_res.best_beta,
            dream_res.mean_pareto_reward,
        )
        return dream_res

    def run_cycle(
        self,
        agent_runner: Callable[[Any, OptimalPolicy], DiscoveryTree],
        candidate_betas: list[float] | None = None,
    ) -> RSICycleResult:
        """Execute one complete online-rollout -> offline-dreaming loop."""
        iter_num = len(self.cycles) + 1
        logger.info("=== Starting Dream-RSI Cycle %d ===", iter_num)

        # 1. Online Rollout
        tree = self.run_online_round(agent_runner)
        best_node = tree.best_node()

        # 2. Offline Dreaming
        dream_res = self.run_offline_dreaming(candidate_betas=candidate_betas)

        cycle_res = RSICycleResult(
            iteration=iter_num,
            online_solved=best_node.done and best_node.score > 0.0,
            online_steps=best_node.step,
            online_reward=best_node.score,
            tree_size=len(tree.nodes),
            discovered_nodes=len(tree.nodes),
            dreaming_result=dream_res,
            policy_beta=self.current_policy.beta,
        )
        self.cycles.append(cycle_res)
        return cycle_res
