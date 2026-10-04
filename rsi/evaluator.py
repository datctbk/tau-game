"""Dream-RSI Pareto Evaluator and Metrics.

Calculates the Pareto objective balancing solution quality, probe cost, and parallelism:
    V = max(score) - beta1 * total_probes + beta2 * (total_probes / rounds)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.meta_policy import OptimalPolicy
from rsi.replay_sim import PuzzleReplaySimulator


import math


@dataclass
class ReplayMetrics:
    """Metrics achieved by an exploration policy on a replay world."""

    best_score: float
    total_probes: int
    decision_rounds: int
    parallel_ratio: float  # total_probes / decision_rounds
    pareto_reward: float
    effective_sequential_rounds: int = 1
    parallel_penalty: float = 1.0

    def summary(self) -> str:
        return (
            f"Best Score: {self.best_score:.3f} | "
            f"Probes: {self.total_probes} | "
            f"Rounds: {self.decision_rounds} | "
            f"Parallelism: {self.parallel_ratio:.2f} | "
            f"Parallel Penalty: {self.parallel_penalty:.2f} | "
            f"Pareto Reward: {self.pareto_reward:.4f}"
        )


class ParetoEvaluator:
    """Evaluates policies on historical replay worlds using Dream-RSI Pareto objective."""

    def __init__(self, beta1: float = 0.05, beta2: float = 0.1) -> None:
        self.beta1 = beta1  # probe cost penalty
        self.beta2 = beta2  # parallelism bonus

    def evaluate(self, policy: OptimalPolicy, sim: PuzzleReplaySimulator) -> ReplayMetrics:
        """Run policy rollout on the replay simulator."""
        sim.reset()
        effective_seq_rounds = 0

        while not sim.is_finished():
            observed = sim.observed()
            legal = sim.legal_actions()
            if not legal:
                break

            batch = policy.select_batch(
                observed=observed,
                legal_nodes=legal,
                max_parallelism=sim.max_parallelism,
                baseline_score=sim.baseline_score,
            )

            if not batch:
                # Policy chose to stop
                break

            # Effective sequential rounds per Dream-RSI: ceil(k / W)
            effective_seq_rounds += math.ceil(len(batch) / max(1, sim.max_parallelism))
            sim.probe_batch(batch)

        best_score = sim.best_revealed_score()
        total_probes = sim.total_probes
        rounds = max(1, sim.current_round)
        parallel_ratio = total_probes / rounds
        effective_seq_rounds = max(1, effective_seq_rounds)
        parallel_penalty = effective_seq_rounds / max(1, total_probes)

        # Dream-RSI Pareto reward
        pareto_reward = best_score - (self.beta1 * total_probes) + (self.beta2 * parallel_ratio)

        return ReplayMetrics(
            best_score=best_score,
            total_probes=total_probes,
            decision_rounds=rounds,
            parallel_ratio=parallel_ratio,
            pareto_reward=pareto_reward,
            effective_sequential_rounds=effective_seq_rounds,
            parallel_penalty=parallel_penalty,
        )

    def sweep_beta(
        self,
        sim: PuzzleReplaySimulator,
        beta_grid: list[float] | None = None,
        max_parallelism: int = 3,
    ) -> dict[float, ReplayMetrics]:
        """Sweep beta across [0.1 .. 0.9] to measure attainment vs work trade-offs."""
        if beta_grid is None:
            beta_grid = [0.1, 0.3, 0.5, 0.7, 0.9]

        results: dict[float, ReplayMetrics] = {}
        for b in beta_grid:
            policy = OptimalPolicy(beta=b, max_parallelism=max_parallelism)
            metrics = self.evaluate(policy, sim)
            results[b] = metrics

        return results
