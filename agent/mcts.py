"""LLM-Guided Monte Carlo Tree Search (Cách 2: LLM-Guided MCTS) for tau-game.

Integrates Monte Carlo Tree Search with LLM policy proposals, sandbox state cloning,
PUCT exploration, and deadlock detection.
"""

from __future__ import annotations

import json
import logging
import math
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

from env.environment import BaseEnvironment, Frame
from llm.client import LLMClient

logger = logging.getLogger(__name__)


def is_corner_deadlock(grid: list[list[int]]) -> bool:
    """Detect if any pushable box (color 8 'C') is trapped in a non-target corner.

    A box at (r, c) is deadlocked if it is flanked by walls (color 5 '#') on two
    perpendicular sides (e.g., Up & Left, Up & Right, Down & Left, Down & Right),
    unless that position is already a target pad (color 6 'M').
    """
    if not grid or not grid[0]:
        return False

    h = len(grid)
    w = len(grid[0])

    for r in range(h):
        for c in range(w):
            if grid[r][c] == 8:  # Pushable box 'C'
                # Check walls in 4 orthogonal directions
                wall_up = (r == 0) or (grid[r - 1][c] == 5)
                wall_down = (r == h - 1) or (grid[r + 1][c] == 5)
                wall_left = (c == 0) or (grid[r][c - 1] == 5)
                wall_right = (c == w - 1) or (grid[r][c + 1] == 5)

                # Check if in any corner
                if (
                    (wall_up and wall_left)
                    or (wall_up and wall_right)
                    or (wall_down and wall_left)
                    or (wall_down and wall_right)
                ):
                    return True
    return False


def estimate_state_heuristic(grid: list[list[int]]) -> float:
    """Calculate a normalized heuristic score in [-1.0, 1.0] for a grid state.

    Higher score = closer boxes to targets or player to goal.
    Deadlock = -1.0.
    """
    if is_corner_deadlock(grid):
        return -1.0

    h = len(grid)
    w = len(grid[0])

    player_pos: tuple[int, int] | None = None
    goals: list[tuple[int, int]] = []
    boxes: list[tuple[int, int]] = []
    targets: list[tuple[int, int]] = []

    for r in range(h):
        for c in range(w):
            val = grid[r][c]
            if val == 1:
                player_pos = (r, c)
            elif val == 3:
                goals.append((r, c))
            elif val == 8:
                boxes.append((r, c))
            elif val == 6:
                targets.append((r, c))

    # Case 1: Sokoban (boxes and targets)
    if boxes and targets:
        total_dist = 0
        for b in boxes:
            min_d = min(abs(b[0] - t[0]) + abs(b[1] - t[1]) for t in targets)
            total_dist += min_d

        max_possible = (h + w) * len(boxes)
        if max_possible == 0:
            return 0.5
        normalized = 1.0 - (total_dist / max_possible)
        return max(-0.8, min(0.9, (normalized * 2.0) - 1.0))

    # Case 2: Standard Goal navigation (player and goal)
    if player_pos and goals:
        min_g = min(abs(player_pos[0] - g[0]) + abs(player_pos[1] - g[1]) for g in goals)
        max_dist = h + w
        normalized = 1.0 - (min_g / max(1, max_dist))
        return max(-0.8, min(0.9, (normalized * 2.0) - 1.0))

    return 0.0


@dataclass
class MCTSNode:
    """Represents a node in the LLM-Guided Monte Carlo Search Tree."""

    state: BaseEnvironment
    frame: Frame
    parent: MCTSNode | None = None
    action_from_parent: str | list[str] | None = None
    children: dict[str, MCTSNode] = field(default_factory=dict)
    visit_count: int = 0
    total_value: float = 0.0
    prior: float = 1.0
    depth: int = 0
    is_terminal: bool = False
    reasoning: str = ""

    @property
    def q_value(self) -> float:
        """Average action value Q(s, a)."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def puct_score(self, parent_visits: int, c_puct: float = 1.414) -> float:
        """Calculate Upper Confidence Bound for Trees (PUCT)."""
        exploration = c_puct * self.prior * (math.sqrt(parent_visits) / (1 + self.visit_count))
        return self.q_value + exploration


@dataclass
class MCTSResult:
    """Outcome of an MCTS search episode."""

    best_action: str | list[str]
    root_visits: int
    tree_size: int
    winning_path: list[str] | None = None
    candidate_actions: dict[str, float] = field(default_factory=dict)
    llm_rationale: str = ""


class LLMGuidedMCTS:
    """Monte Carlo Tree Search guided by LLM policy priors and value estimation."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        num_simulations: int = 10,
        c_puct: float = 1.414,
        max_depth: int = 8,
        rollout_steps: int = 3,
        verbose: bool = False,
        on_token: Callable[[str, bool], None] | None = None,
    ) -> None:
        self.llm = llm
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.rollout_steps = rollout_steps
        self.verbose = verbose
        self.on_token = on_token

    def search(
        self,
        env: BaseEnvironment,
        world_model: str = "",
        candidate_filter: Callable[[list[str]], list[str]] | None = None,
    ) -> MCTSResult:
        """Run MCTS simulations from the current environment state."""
        # 1. Initialize root node with a clone of the environment
        root_env = deepcopy(env)
        root_frame = root_env.current_frame
        root = MCTSNode(
            state=root_env,
            frame=root_frame,
            depth=0,
            is_terminal=root_env.is_done(),
        )

        if root.is_terminal:
            return MCTSResult(best_action="RESET", root_visits=0, tree_size=1)

        # 2. Expand root node immediately
        self._expand(root, world_model=world_model, candidate_filter=candidate_filter)

        if not root.children:
            valid = root.state.valid_actions()
            fallback = valid[0] if valid else "DOWN"
            return MCTSResult(best_action=fallback, root_visits=0, tree_size=1)

        # Check if any immediate child solved the level
        for act, child in root.children.items():
            if child.is_terminal and child.state.is_done():
                return MCTSResult(
                    best_action=act,
                    root_visits=1,
                    tree_size=len(root.children) + 1,
                    winning_path=[act],
                    candidate_actions={act: 1.0},
                    llm_rationale=child.reasoning or "Immediate winning action",
                )

        # 3. Run MCTS iterations
        tree_size = 1 + len(root.children)
        winning_sequence: list[str] | None = None

        for sim in range(self.num_simulations):
            # Phase A: Select
            node = self._select(root)

            # Phase B: Expand if not terminal and depth within limit
            if not node.is_terminal and node.depth < self.max_depth and not node.children:
                new_nodes = self._expand(node, world_model=world_model, candidate_filter=candidate_filter)
                tree_size += new_nodes
                if node.children:
                    # Pick best child of newly expanded node
                    node = self._select(node)

            # Phase C: Evaluate
            val = self._evaluate(node)

            if node.is_terminal and node.state.is_done():
                # Reconstruct winning path
                path: list[str] = []
                curr: MCTSNode | None = node
                while curr and curr.parent:
                    if isinstance(curr.action_from_parent, str):
                        path.append(curr.action_from_parent)
                    elif isinstance(curr.action_from_parent, list):
                        path.extend(reversed(curr.action_from_parent))
                    curr = curr.parent
                winning_sequence = list(reversed(path))

            # Phase D: Backpropagate
            self._backpropagate(node, val)

        # 4. Pick best action: prioritize winning sequence, then highest visit count among viable actions
        action_scores: dict[str, float] = {}
        for act, child in root.children.items():
            action_scores[act] = float(child.visit_count)

        # Exclude blocked actions or dead ends from best_act selection
        viable_actions = [
            act for act, child in root.children.items()
            if not (child.is_terminal and not child.state.is_done()) and child.q_value > -0.9
        ]
        if not viable_actions:
            viable_actions = list(root.children.keys())

        if winning_sequence:
            best_act = winning_sequence[0]
        else:
            best_act = max(
                viable_actions,
                key=lambda a: (root.children[a].visit_count, root.children[a].q_value),
            )

        best_child = root.children[best_act]
        return MCTSResult(
            best_action=best_act,
            root_visits=sum(c.visit_count for c in root.children.values()),
            tree_size=tree_size,
            winning_path=winning_sequence,
            candidate_actions=action_scores,
            llm_rationale=best_child.reasoning or root.reasoning,
        )

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select child node prioritizing unvisited children first, then maximum PUCT score."""
        curr = node
        while curr.children and not curr.is_terminal:
            # Always explore unvisited children at least once before PUCT exploitation
            unvisited = [c for c in curr.children.values() if c.visit_count == 0]
            if unvisited:
                return unvisited[0]

            best_score = -float("inf")
            best_child = None

            for child in curr.children.values():
                score = child.puct_score(parent_visits=curr.visit_count, c_puct=self.c_puct)
                if score > best_score:
                    best_score = score
                    best_child = child

            if best_child is None:
                break
            curr = best_child

        return curr

    def _expand(
        self,
        node: MCTSNode,
        world_model: str = "",
        candidate_filter: Callable[[list[str]], list[str]] | None = None,
    ) -> int:
        """Expand node by proposing candidate actions via LLM policy prior."""
        valid_actions = node.state.valid_actions()
        if not valid_actions or node.is_terminal:
            return 0

        # In forward search, filter out RESET unless it is the only action
        forward_actions = [a for a in valid_actions if a != "RESET"] or valid_actions
        if candidate_filter:
            forward_actions = candidate_filter(forward_actions) or forward_actions

        # In Mode 2: Query LLM only at the Root Node (depth == 0) to get policy priors for candidate actions.
        # Deeper nodes use fast heuristic priors (10x faster, exactly 1 LLM call per turn!).
        if node.depth == 0 and self.llm is not None:
            priors, reasoning = self._get_action_priors(node, forward_actions, world_model)
        else:
            prob = 1.0 / len(forward_actions) if forward_actions else 1.0
            priors, reasoning = {a: prob for a in forward_actions}, "Tree expansion heuristic prior"
        node.reasoning = reasoning

        count = 0
        for act, prior_prob in priors.items():
            if act in node.children:
                continue

            # Clone state into sandbox
            child_env = deepcopy(node.state)
            child_frame = child_env.step(act)

            # Check if action was blocked (produced no state change)
            is_blocked = child_frame.grid == node.frame.grid
            if is_blocked:
                child_node = MCTSNode(
                    state=child_env,
                    frame=child_frame,
                    parent=node,
                    action_from_parent=act,
                    prior=0.01,
                    depth=node.depth + 1,
                    is_terminal=True,
                    total_value=-1.0,
                    visit_count=1,
                    reasoning="Blocked action (no state change)",
                )
            else:
                child_node = MCTSNode(
                    state=child_env,
                    frame=child_frame,
                    parent=node,
                    action_from_parent=act,
                    prior=prior_prob,
                    depth=node.depth + 1,
                    is_terminal=child_env.is_done(),
                    reasoning=reasoning,
                )
            node.children[act] = child_node
            count += 1

        return count

    def _get_action_priors(
        self,
        node: MCTSNode,
        valid_actions: list[str],
        world_model: str,
    ) -> tuple[dict[str, float], str]:
        """Query LLM to generate candidate action priors P(a|s)."""
        if not self.llm or not valid_actions:
            # Uniform prior fallback
            prob = 1.0 / len(valid_actions)
            return {a: prob for a in valid_actions}, "Uniform heuristic prior"

        prompt = (
            f"You are the Policy Guide in an MCTS tree search for a 2D puzzle game.\n"
            f"Current Grid:\n{node.frame.ascii}\n"
            f"Valid actions: {valid_actions}\n"
            f"World model knowledge:\n{world_model or '(none)'}\n\n"
            f"Select top 2 to 3 most promising candidate moves to explore.\n"
            f"Output strictly valid JSON with action names, confidence priors (0.0 to 1.0 summing to ~1.0), and 1-sentence rationale.\n"
            f'Format: {{"actions": [{{"action": "UP", "prior": 0.6, "reason": "Move toward target"}}, ...], "rationale": "..."}}\n'
        )

        try:
            messages = [{"role": "user", "content": prompt}]
            resp = self.llm.generate(messages, on_token=self.on_token)

            # Try to parse JSON from response
            match = re.search(r"\{.*\}", resp, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                priors: dict[str, float] = {}
                actions_list = data.get("actions", [])
                for item in actions_list:
                    act = str(item.get("action", "")).upper()
                    if act in valid_actions:
                        p = float(item.get("prior", 0.5))
                        priors[act] = max(0.05, min(1.0, p))

                if priors:
                    # Normalize
                    total = sum(priors.values())
                    norm_priors = {k: v / total for k, v in priors.items()}
                    rationale = str(data.get("rationale", ""))
                    return norm_priors, rationale
        except Exception as err:
            logger.debug("LLM prior generation error: %s", err)

        # Fallback 1: extract actions mentioned in plain text/thought if JSON failed
        try:
            mentioned = [a for a in valid_actions if re.search(rf"\b{a}\b", resp.upper())]
            if mentioned:
                priors = {
                    a: (0.8 / len(mentioned) if a in mentioned else 0.2 / max(1, len(valid_actions) - len(mentioned)))
                    for a in valid_actions
                }
                total = sum(priors.values())
                return {k: v / total for k, v in priors.items()}, "Extracted actions from LLM reasoning"
        except Exception:
            pass

        # Fallback 2: uniform prior over valid actions
        prob = 1.0 / len(valid_actions)
        return {a: prob for a in valid_actions}, "Fallback heuristic prior"

    def _evaluate(self, node: MCTSNode) -> float:
        """Evaluate state value V(s) in [-1.0, 1.0]."""
        # Terminal win
        if node.is_terminal and node.state.is_done():
            return 1.0

        # Blocked move: taking an action resulted in zero state change
        if node.parent and node.frame.grid == node.parent.frame.grid:
            return -1.0

        grid = node.frame.grid

        # Deadlock check
        if is_corner_deadlock(grid):
            return -1.0

        # Heuristic score from current grid
        base_score = estimate_state_heuristic(grid)
        if node.frame.info.get("hit_wall"):
            base_score -= 0.5

        # Quick simulated rollout (up to rollout_steps) with step discounting
        if self.rollout_steps > 0 and not node.is_terminal:
            sim_env = deepcopy(node.state)
            for step_i in range(1, self.rollout_steps + 1):
                if sim_env.is_done():
                    return 1.0 * (0.95 ** step_i)
                valid = sim_env.valid_actions()
                if not valid:
                    break

                # Greedily pick action that maximizes progress
                best_next_score = -float("inf")
                best_next_act = valid[0]
                for a in valid:
                    test_env = deepcopy(sim_env)
                    test_env.step(a)
                    if test_env.is_done():
                        return 1.0 * (0.95 ** step_i)

                    if test_env.current_frame.grid == sim_env.current_frame.grid:
                        sc = -1.0
                    else:
                        sc = estimate_state_heuristic(test_env.current_frame.grid)
                        if test_env.current_frame.info.get("hit_wall"):
                            sc -= 0.5
                    if sc > best_next_score:
                        best_next_score = sc
                        best_next_act = a

                sim_env.step(best_next_act)
                if is_corner_deadlock(sim_env.current_frame.grid):
                    return -0.9

            rollout_score = estimate_state_heuristic(sim_env.current_frame.grid) * (0.95 ** self.rollout_steps)
            return 0.5 * base_score + 0.5 * rollout_score

        return base_score

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagate evaluation value V up to root."""
        curr: MCTSNode | None = node
        while curr is not None:
            curr.visit_count += 1
            curr.total_value += value
            curr = curr.parent
