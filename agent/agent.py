"""DuckAgent orchestrator: Connects LLM, REPL sandbox, Perception, and Environment."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.memory import GameMemory, TurnRecord
from agent.mcts import LLMGuidedMCTS, MCTSResult
from agent.prompts import SYSTEM_PROMPT, build_turn_prompt
from agent.repl import PythonREPL, REPLResult
from env.environment import BaseEnvironment, Frame, Transition
from llm.client import LLMClient, extract_python_code

logger = logging.getLogger(__name__)


@dataclass
class GameRunResult:
    """Summary outcome of an agent's game-solving run."""

    solved: bool
    total_steps: int
    total_reward: float
    final_frame: Frame
    world_model: str
    turns: list[TurnRecord] = field(default_factory=list)
    history: list[Transition] = field(default_factory=list)

    def summary(self) -> str:
        status = "SOLVED ✓" if self.solved else "UNSOLVED ✗"
        return (
            f"=== DuckAgent Run Result ===\n"
            f"Status: {status}\n"
            f"Steps taken: {self.total_steps}\n"
            f"Total Reward: {self.total_reward}\n"
            f"Discovered World Model:\n{self.world_model or '(none)'}\n"
        )


class DuckAgent:
    """Autonomous game-solving agent using Python REPL exploration and World Model tracking."""

    def __init__(
        self,
        env: BaseEnvironment,
        llm: LLMClient,
        max_turns: int = 30,
        max_idle_turns: int = 5,
        verbose: bool = True,
        on_step_callback: Callable[..., None] | None = None,
        on_turn_start: Callable[[int, Frame], None] | None = None,
        on_token: Callable[[int, str, bool], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        use_mcts: bool = False,
        mcts_sims: int = 10,
    ) -> None:
        self.env = env
        self.llm = llm
        self.max_turns = max_turns
        self.max_idle_turns = max_idle_turns
        self.verbose = verbose
        self.on_step_callback = on_step_callback
        self.on_turn_start = on_turn_start
        self.on_token = on_token
        self.cancel_check = cancel_check
        self.use_mcts = use_mcts
        self.mcts_sims = mcts_sims

        self.memory = GameMemory()
        self.repl = PythonREPL(env=self.env, world_model_ref=lambda: self.memory.world_model)
        self.messages: list[dict[str, str]] = []

    def run(self) -> GameRunResult:
        """Run the main observe-reason-code-action loop."""
        initial_frame = self.env.reset()
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        last_repl_output: str | None = None
        last_actions: list[str] | None = None
        idle_count = 0
        total_reward = 0.0

        if self.verbose:
            logger.info("Starting DuckAgent run on level %d", initial_frame.level)

        for turn_idx in range(1, self.max_turns + 1):
            if self.cancel_check and self.cancel_check():
                logger.info("DuckAgent run cancelled by user.")
                break

            curr_frame = self.env.current_frame
            if self.env.is_done():
                break

            if self.on_turn_start:
                try:
                    self.on_turn_start(turn_idx, curr_frame)
                except Exception:
                    pass

            prev_info = dict(curr_frame.info or {})

            # 1. Build observation prompt
            user_prompt = build_turn_prompt(
                current_frame=curr_frame,
                valid_actions=self.env.valid_actions(),
                world_model=self.memory.world_model,
                recent_actions=last_actions,
                last_repl_output=last_repl_output,
                previous_frame=self.env.previous_frame,
            )
            self.messages.append({"role": "user", "content": user_prompt})

            # 2. Context eviction / trimming
            self.messages = self.memory.trim_messages(self.messages)

            # 3. Decision Making: LLM-Guided MCTS vs Standard REPL LLM
            response: str = ""
            code: str | None = None
            repl_res: REPLResult | None = None
            actions_in_turn: list[str] = []

            if self.use_mcts:
                if self.verbose:
                    logger.info("Running LLM-Guided MCTS (sims=%d)...", self.mcts_sims)

                mcts_engine = LLMGuidedMCTS(
                    llm=self.llm,
                    num_simulations=self.mcts_sims,
                    verbose=self.verbose,
                )
                mcts_res: MCTSResult = mcts_engine.search(self.env, world_model=self.memory.world_model)

                if mcts_res.winning_path:
                    chosen_actions = mcts_res.winning_path
                elif isinstance(mcts_res.best_action, list):
                    chosen_actions = mcts_res.best_action
                else:
                    chosen_actions = [mcts_res.best_action]

                rationale = mcts_res.llm_rationale or "MCTS highest visit path"
                response = (
                    f"MCTS tree search completed (visits={mcts_res.root_visits}, tree_size={mcts_res.tree_size}).\n"
                    f"Rationale: {rationale}\n\n"
                    f"```python\n"
                    f"action({chosen_actions})\n"
                    f"```"
                )
                self.messages.append({"role": "assistant", "content": response})

                code = f"action({chosen_actions})"
                repl_res = self.repl.execute(code)
                last_repl_output = str(repl_res)
                actions_in_turn = repl_res.actions_taken
            else:
                def _token_cb(delta: str, is_thinking: bool):
                    if self.on_token:
                        try:
                            self.on_token(turn_idx, delta, is_thinking)
                        except Exception:
                            pass

                response = self.llm.generate(
                    self.messages,
                    on_token=_token_cb if self.on_token else None,
                )
                self.messages.append({"role": "assistant", "content": response})

                # 4. Extract and execute Python code in sandbox REPL
                code = extract_python_code(response)
                if code:
                    repl_res = self.repl.execute(code)
                    last_repl_output = str(repl_res)
                    actions_in_turn = repl_res.actions_taken
                else:
                    last_repl_output = "(No Python code block found in response. Remember to use ```python ... ```)"
                    actions_in_turn = []

            # 5. Check progress / idle stall
            if not actions_in_turn:
                idle_count += 1
            else:
                idle_count = 0

            # 6. Record turn into memory & update world model
            self.memory.record_turn(
                step=turn_idx,
                observation_summary=f"Frame step={curr_frame.step}, level={curr_frame.level}",
                response=response,
                code=code,
                repl_output=last_repl_output,
                actions=actions_in_turn,
                previous_info=prev_info,
                current_info=dict(self.env.current_frame.info or {}),
            )

            last_actions = actions_in_turn

            if self.on_step_callback:
                try:
                    self.on_step_callback(turn_idx, self.memory.turns[-1], self.env)
                except TypeError:
                    self.on_step_callback(turn_idx, self.memory.turns[-1])

            if self.verbose:
                act_str = f"actions={actions_in_turn}" if actions_in_turn else "inspecting"
                logger.info("Turn %d: %s | Solved: %s", turn_idx, act_str, self.env.is_done())

            if self.env.is_done():
                break

            if idle_count >= self.max_idle_turns:
                logger.warning("Agent stalled for %d turns without taking actions.", idle_count)
                break

        # Compute total reward from transitions
        total_reward = sum(t.reward for t in self.env.history)

        return GameRunResult(
            solved=self.env.is_done(),
            total_steps=len(self.env.history),
            total_reward=total_reward,
            final_frame=self.env.current_frame,
            world_model=self.memory.world_model,
            turns=list(self.memory.turns),
            history=list(self.env.history),
        )
