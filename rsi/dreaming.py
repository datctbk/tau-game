"""Dreaming Optimizer: Offline policy optimization across historical discovery trees.

Iterates over historical puzzle worlds, evaluates candidate exploration policies at zero cost,
and selects the upgraded policy for the next online run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from agent.discovery_tree import DiscoveryTree
from agent.meta_policy import OptimalPolicy
from rsi.evaluator import ParetoEvaluator, ReplayMetrics
from rsi.replay_sim import PuzzleReplaySimulator

logger = logging.getLogger(__name__)


@dataclass
class DreamingResult:
    """Summary outcome of offline dreaming phase."""

    best_beta: float
    best_policy: OptimalPolicy
    mean_pareto_reward: float
    mean_score: float
    total_simulated_probes: int
    evaluations_summary: list[dict[str, Any]]

    def summary(self) -> str:
        return (
            f"=== Dream-RSI Dreaming Summary ===\n"
            f"Optimal Beta: {self.best_beta}\n"
            f"Mean Pareto Reward: {self.mean_pareto_reward:.4f}\n"
            f"Mean Puzzle Score: {self.mean_score:.3f}\n"
            f"Total Simulated Probes: {self.total_simulated_probes}\n"
        )


class DreamingOptimizer:
    """Orchestrates offline dreaming across a pool of historical discovery trees."""

    def __init__(
        self,
        trees: list[DiscoveryTree],
        beta1: float = 0.05,
        beta2: float = 0.1,
        max_parallelism: int = 3,
    ) -> None:
        self.trees = trees
        self.evaluator = ParetoEvaluator(beta1=beta1, beta2=beta2)
        self.max_parallelism = max_parallelism

    def optimize(
        self,
        candidate_betas: list[float] | None = None,
        on_progress: Callable[[int, int, float, list[ReplayMetrics]], None] | None = None,
    ) -> DreamingResult:
        """Evaluate candidate beta values across all replay trees and pick the best."""
        if not self.trees:
            raise ValueError("No historical trees available for dreaming.")

        if candidate_betas is None:
            candidate_betas = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

        beta_scores: dict[float, list[ReplayMetrics]] = {b: [] for b in candidate_betas}
        total_b = len(candidate_betas)

        for idx, b in enumerate(candidate_betas, start=1):
            for tree in self.trees:
                sim = PuzzleReplaySimulator(tree=tree, max_parallelism=self.max_parallelism)
                policy = OptimalPolicy(beta=b, max_parallelism=self.max_parallelism)
                metrics = self.evaluator.evaluate(policy, sim)
                beta_scores[b].append(metrics)

            if on_progress:
                try:
                    on_progress(idx, total_b, b, beta_scores[b])
                except Exception:
                    pass

        # Average metrics across all replay trees
        best_b = candidate_betas[0]
        best_mean_reward = -float("inf")
        summary_list: list[dict[str, Any]] = []

        for b, metrics_list in beta_scores.items():
            mean_reward = sum(m.pareto_reward for m in metrics_list) / len(metrics_list)
            mean_score = sum(m.best_score for m in metrics_list) / len(metrics_list)
            mean_probes = sum(m.total_probes for m in metrics_list) / len(metrics_list)

            summary_list.append(
                {
                    "beta": b,
                    "mean_pareto_reward": mean_reward,
                    "mean_score": mean_score,
                    "mean_probes": mean_probes,
                }
            )

            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                best_b = b

        best_metrics = beta_scores[best_b]
        mean_best_score = sum(m.best_score for m in best_metrics) / len(best_metrics)
        total_probes = sum(sum(m.total_probes for m in mlist) for mlist in beta_scores.values())

        return DreamingResult(
            best_beta=best_b,
            best_policy=OptimalPolicy(beta=best_b, max_parallelism=self.max_parallelism),
            mean_pareto_reward=best_mean_reward,
            mean_score=mean_best_score,
            total_simulated_probes=total_probes,
            evaluations_summary=summary_list,
        )
