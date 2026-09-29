"""Comprehensive tests for Dream-RSI integration in tau-game."""

import sys
from pathlib import Path

pkg_root = Path(__file__).resolve().parent.parent
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

from agent.discovery_tree import DiscoveryNode, DiscoveryTree
from agent.meta_policy import ObservationSignal, OptimalPolicy
from env.environment import Frame
from env.game import GridWorld
from rsi.dreaming import DreamingOptimizer
from rsi.evaluator import ParetoEvaluator
from rsi.orchestrator import DreamRSIOrchestrator
from rsi.replay_sim import PuzzleReplaySimulator


def test_discovery_tree_creation_and_serialization(tmp_path):
    root_frame = Frame(grid=[[0, 1], [0, 3]], step=0, level=1)
    tree = DiscoveryTree(root_frame=root_frame)

    assert tree.root_id == "node_0"
    assert len(tree.nodes) == 1

    # Add successful attempt
    node1 = tree.add_attempt(
        parent_id=tree.root_id,
        actions=["RIGHT"],
        frame=Frame(grid=[[0, 0], [1, 3]], step=1, level=1),
        score=0.5,
    )

    # Add repairable error attempt
    node2 = tree.add_attempt(
        parent_id=node1.node_id,
        actions=["UP"],
        frame=Frame(grid=[[0, 0], [1, 3]], step=2, level=1),
        score=0.5,
        error="IndexError: grid index out of range",
    )

    assert node2.fail_class == "repairable_code"
    assert node2.is_repairable

    # Save and reload
    filepath = tmp_path / "test_tree.json"
    tree.save(filepath)
    loaded_tree = DiscoveryTree.load(filepath)

    assert len(loaded_tree.nodes) == 3
    assert loaded_tree.nodes[node2.node_id].fail_class == "repairable_code"
    assert loaded_tree.best_node().score == 0.5


def test_optimal_policy_dynamic_portfolio():
    root_frame = Frame(grid=[[0, 1], [0, 3]], step=0, level=1)
    tree = DiscoveryTree(root_frame=root_frame)

    # Branch 1: High scoring exploitation candidate
    n1 = tree.add_attempt(tree.root_id, ["DOWN"], root_frame, score=0.8)
    # Branch 2: Repairable code error candidate
    n2 = tree.add_attempt(tree.root_id, ["RIGHT"], root_frame, score=0.4, error="NameError: name 'x' is not defined")
    # Branch 3: Hard dead end
    n3 = tree.add_attempt(tree.root_id, ["UP"], root_frame, score=-1.0, fail_class="dead_end")

    policy = OptimalPolicy(beta=0.6, max_parallelism=2)
    observed = tree.nodes
    legal = [n1.node_id, n2.node_id, n3.node_id]

    batch = policy.select_batch(observed, legal, max_parallelism=2)

    # Must contain repairable candidate (recovery) and exploitation candidate (n1)
    assert len(batch) == 2
    assert n1.node_id in batch
    assert n2.node_id in batch
    # Hard dead-end should never be selected
    assert n3.node_id not in batch


def test_puzzle_replay_simulator_and_pareto_evaluation():
    root_frame = Frame(grid=[[0, 1], [0, 3]], step=0, level=1)
    tree = DiscoveryTree(root_frame=root_frame)

    # Create multi-depth discovery tree
    c1 = tree.add_attempt(tree.root_id, ["DOWN"], root_frame, score=0.3)
    c2 = tree.add_attempt(c1.node_id, ["RIGHT"], root_frame, score=0.7)
    c3 = tree.add_attempt(c2.node_id, ["UP"], root_frame, score=1.0, done=True)

    sim = PuzzleReplaySimulator(tree=tree, max_parallelism=2)
    assert len(sim.observed()) == 1  # Only root revealed initially
    assert sim.legal_actions() == [tree.root_id]

    evaluator = ParetoEvaluator(beta1=0.05, beta2=0.1)
    policy = OptimalPolicy(beta=0.7, max_parallelism=2)

    metrics = evaluator.evaluate(policy, sim)

    # Should traverse and reach final score
    assert metrics.best_score == 1.0
    assert metrics.total_probes > 0
    assert metrics.decision_rounds > 0
    assert metrics.pareto_reward > 0


def test_dreaming_optimizer():
    root_frame = Frame(grid=[[0, 1], [0, 3]], step=0, level=1)
    tree1 = DiscoveryTree(root_frame=root_frame)
    a1 = tree1.add_attempt(tree1.root_id, ["RIGHT"], root_frame, score=0.5)
    tree1.add_attempt(a1.node_id, ["DOWN"], root_frame, score=1.0, done=True)

    optimizer = DreamingOptimizer(trees=[tree1], max_parallelism=2)
    result = optimizer.optimize(candidate_betas=[0.2, 0.5, 0.8])

    assert result.best_beta in [0.2, 0.5, 0.8]
    assert result.mean_score == 1.0
    assert result.total_simulated_probes > 0
    assert "Dream-RSI Dreaming Summary" in result.summary()


def test_dream_rsi_orchestrator_cycle(tmp_path):
    def env_factory():
        return GridWorld(height=4, width=4, player_start=(0, 0), goal_pos=(1, 1))

    orchestrator = DreamRSIOrchestrator(
        env_factory=env_factory,
        initial_beta=0.5,
        max_parallelism=2,
        artifact_dir=tmp_path / "rsi_artifacts",
    )

    def dummy_agent_runner(env, policy):
        # Generate a discovery tree mimicking an online episode
        tree = DiscoveryTree(root_frame=env.reset())
        n1 = tree.add_attempt(tree.root_id, ["RIGHT"], env.current_frame, score=0.5)
        tree.add_attempt(n1.node_id, ["DOWN"], env.current_frame, score=1.0, done=True)
        return tree

    cycle = orchestrator.run_cycle(dummy_agent_runner, candidate_betas=[0.3, 0.7])

    assert cycle.iteration == 1
    assert cycle.online_solved is True
    assert cycle.online_reward == 1.0
    assert cycle.tree_size == 3
    assert len(orchestrator.history_pool) == 1
    assert (tmp_path / "rsi_artifacts" / "tree_round_001.json").exists()
