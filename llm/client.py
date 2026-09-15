"""LLM client for tau-game — reuses tau's providers, sub-sessions, and local endpoints."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)


def extract_python_code(text: str) -> str:
    """Extract Python code enclosed in markdown code blocks, standalone action calls, or raw code."""
    if not text:
        return ""

    # 1. Standard markdown code blocks: ```python ... ``` or ``` ... ```
    pattern = r"```(?:python)?\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        extracted = "\n\n".join(m.strip() for m in matches if m.strip())
        if extracted:
            return extracted

    # 2. Unclosed code blocks (e.g. if generation hit max_tokens before closing ```)
    unclosed = re.findall(r"```(?:python)?\s*\n(action\(.*)", text, re.DOTALL | re.IGNORECASE)
    if unclosed:
        return unclosed[0].strip()

    # 3. If the response itself appears to be bare python code without markdown
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and all(line.startswith(("action(", "print(", "for ", "if ", "def ", "import ", "#", "world_model", "current_frame", "previous_frame", "valid_actions")) for line in lines):
        return text.strip()

    # 4. Standalone action(...) calls anywhere in natural language text (e.g. "I will action(['DOWN'])")
    action_matches = re.findall(
        r"action\s*\(\s*(?:\[[^\]]*\]|\"[^\"]*\"|'[^']*')\s*\)", text
    )
    if action_matches:
        return "\n".join(action_matches)

    return ""


class StreamTokenFilter:
    """Detects and separates thinking tokens from visible content in streaming token streams.

    Handles:
    1. Provider-level thinking flags (reasoning_content, TextDelta.is_thinking).
    2. In-band <think>...</think> tags emitted by models like Qwen 2.5 and DeepSeek.
    """

    def __init__(self, callback: Callable[[str, bool], None]) -> None:
        self.callback = callback
        self.in_think = False
        self.buffer = ""

    def push(self, text: str, is_thinking: bool = False) -> None:
        if is_thinking:
            self.callback(text, True)
            return

        self.buffer += text
        while self.buffer:
            if not self.in_think:
                if "<think>" in self.buffer:
                    pre, _, post = self.buffer.partition("<think>")
                    if pre:
                        self.callback(pre, False)
                    self.in_think = True
                    self.buffer = post
                else:
                    # Check for partial prefix of "<think>" at end of buffer
                    partial = None
                    for i in range(1, min(7, len(self.buffer) + 1)):
                        if "<think>"[:i] == self.buffer[-i:]:
                            partial = i
                            break
                    if partial:
                        to_emit = self.buffer[:-partial]
                        if to_emit:
                            self.callback(to_emit, False)
                        self.buffer = self.buffer[-partial:]
                        break
                    else:
                        self.callback(self.buffer, False)
                        self.buffer = ""
            else:
                if "</think>" in self.buffer:
                    think_text, _, post = self.buffer.partition("</think>")
                    if think_text:
                        self.callback(think_text, True)
                    self.in_think = False
                    self.buffer = post
                else:
                    # Check for partial prefix of "</think>" at end of buffer
                    partial = None
                    for i in range(1, min(8, len(self.buffer) + 1)):
                        if "</think>"[:i] == self.buffer[-i:]:
                            partial = i
                            break
                    if partial:
                        to_emit = self.buffer[:-partial]
                        if to_emit:
                            self.callback(to_emit, True)
                        self.buffer = self.buffer[-partial:]
                        break
                    else:
                        self.callback(self.buffer, True)
                        self.buffer = ""

    def flush(self) -> None:
        if self.buffer:
            self.callback(self.buffer, self.in_think)
            self.buffer = ""


class LLMClient:
    """Unified LLM interface supporting Tau sub-sessions, Tau providers, and OpenAI endpoints."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 512,
        extension_context: Any = None,
        mock_fn: Callable[[list[dict[str, str]]], str] | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.extension_context = extension_context
        self.mock_fn = mock_fn

        self._tau_provider: Any = None
        self._init_tau_provider()

    def _init_tau_provider(self) -> None:
        """Initialize tau provider if running standalone without extension context."""
        if self.mock_fn is not None or self.extension_context is not None:
            return

        try:
            from tau.config import load_config
            from tau.core.types import AgentConfig
            from tau.providers import get_provider

            tau_cfg = load_config()
            prov_name = self.provider or tau_cfg.default_provider or "openai"
            model_name = self.model or tau_cfg.default_model or "gpt-4o"

            if self.base_url:
                tau_cfg.openai.base_url = self.base_url
            if self.api_key:
                tau_cfg.openai.api_key = self.api_key

            agent_cfg = AgentConfig(
                provider=prov_name,
                model=model_name,
                max_tokens=self.max_tokens,
            )
            self._tau_provider = get_provider(tau_cfg, agent_cfg)
            logger.info("Initialized tau provider: %s (%s)", prov_name, model_name)
        except Exception as exc:
            logger.debug("Could not initialize native tau provider: %s", exc)

    def generate(
        self,
        messages: list[dict[str, str]],
        on_token: Callable[[str, bool], None] | None = None,
    ) -> str:
        """Generate assistant response given conversation messages with streaming support."""
        # 1. Mock response for testing
        if self.mock_fn is not None:
            return self.mock_fn(messages)

        # 2. Extension sub-session (when running inside active Tau CLI)
        if self.extension_context is not None:
            return self._call_via_sub_session(messages, on_token=on_token)

        # 3. Native Tau Provider
        if self._tau_provider is not None:
            return self._call_via_tau_provider(messages, on_token=on_token)

        # 4. Fallback to direct OpenAI client if installed
        return self._call_via_openai_fallback(messages, on_token=on_token)

    def _call_via_sub_session(
        self,
        messages: list[dict[str, str]],
        on_token: Callable[[str, bool], None] | None = None,
    ) -> str:
        """Call LLM via Tau's isolated sub-session with streaming and thinking fallback."""
        from tau.core.types import TextDelta

        system_prompt = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        user_prompt = messages[-1]["content"] if messages else ""

        with self.extension_context.create_sub_session(
            system_prompt=system_prompt,
            load_skills=False,
            load_extensions=False,
            load_context_files=False,
            allowed_tools=[],
            max_turns=1,
        ) as sub:
            # Enforce max_tokens on child agent config so large models do not ramble indefinitely
            if hasattr(sub, "agent") and hasattr(sub.agent, "config"):
                sub.agent.config.max_tokens = self.max_tokens

            visible_parts: list[str] = []
            thinking_parts: list[str] = []

            def _handle_token(chunk_text: str, is_th: bool) -> None:
                if is_th:
                    thinking_parts.append(chunk_text)
                else:
                    visible_parts.append(chunk_text)
                if on_token:
                    on_token(chunk_text, is_th)

            token_filter = StreamTokenFilter(_handle_token)

            # Stream events live from the session generator
            for event in sub.prompt(user_prompt):
                if isinstance(event, TextDelta):
                    if event.is_thinking:
                        token_filter.push(event.text, is_thinking=True)
                    else:
                        token_filter.push(event.text, is_thinking=False)
                elif hasattr(event, "content") and event.content:
                    token_filter.push(event.content, is_thinking=False)

            token_filter.flush()

            visible_text = "".join(visible_parts).strip()
            thinking_text = "".join(thinking_parts).strip()

            # Case A: If visible text is empty, fall back entirely to thinking text
            if not visible_text:
                return thinking_text

            # Case B: If visible text has no code, but thinking text contains actions, include thinking
            if not extract_python_code(visible_text) and extract_python_code(thinking_text):
                return f"{thinking_text}\n\n{visible_text}"

            # Case C: If visible text is bare code without explanation, prepend brief thought preview
            if thinking_text and not re.search(r"[a-zA-Z]{4,}", re.sub(r"```.*?```", "", visible_text, flags=re.DOTALL)):
                thought_preview = thinking_text.replace("\n", " ")[:200]
                return f"Thought: {thought_preview}...\n\n{visible_text}"

            return visible_text

    def _call_via_tau_provider(
        self,
        messages: list[dict[str, str]],
        on_token: Callable[[str, bool], None] | None = None,
    ) -> str:
        """Call LLM via Tau provider instance with streaming if supported."""
        from tau.core.types import Message, TextDelta

        tau_messages = [
            Message(role=m.get("role", "user"), content=m.get("content", ""))
            for m in messages
        ]

        if hasattr(self._tau_provider, "chat"):
            try:
                gen_or_resp = self._tau_provider.chat(tau_messages, tools=[], stream=True)
                if hasattr(gen_or_resp, "__iter__") and not hasattr(gen_or_resp, "content"):
                    visible_parts: list[str] = []
                    thinking_parts: list[str] = []

                    def _handle_token(chunk_text: str, is_th: bool) -> None:
                        if is_th:
                            thinking_parts.append(chunk_text)
                        else:
                            visible_parts.append(chunk_text)
                        if on_token:
                            on_token(chunk_text, is_th)

                    token_filter = StreamTokenFilter(_handle_token)

                    for event in gen_or_resp:
                        if isinstance(event, TextDelta):
                            token_filter.push(event.text, is_thinking=bool(event.is_thinking))
                        elif hasattr(event, "content") and event.content:
                            token_filter.push(event.content, is_thinking=False)

                    token_filter.flush()
                    visible_text = "".join(visible_parts).strip()
                    if not visible_text and thinking_parts:
                        return "".join(thinking_parts).strip()
                    return visible_text
                elif hasattr(gen_or_resp, "content"):
                    return gen_or_resp.content or ""
            except Exception as exc:
                logger.debug("Provider chat streaming failed, falling back to complete: %s", exc)

        response = self._tau_provider.complete(tau_messages)
        return response.content or ""

    def _call_via_openai_fallback(
        self,
        messages: list[dict[str, str]],
        on_token: Callable[[str, bool], None] | None = None,
    ) -> str:
        """Direct fallback using openai library with streaming support."""
        try:
            from openai import OpenAI

            base_url = self.base_url or os.getenv("OPENAI_BASE_URL") or "http://localhost:8080/v1"
            api_key = self.api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
            model = self.model or "default"

            client = OpenAI(base_url=base_url, api_key=api_key)

            if on_token is not None:
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,  # type: ignore
                    temperature=0.3,
                    max_tokens=self.max_tokens,
                    stream=True,
                )
                visible_parts: list[str] = []
                thinking_parts: list[str] = []

                def _handle_token(chunk_text: str, is_th: bool) -> None:
                    if is_th:
                        thinking_parts.append(chunk_text)
                    else:
                        visible_parts.append(chunk_text)
                    if on_token:
                        on_token(chunk_text, is_th)

                token_filter = StreamTokenFilter(_handle_token)

                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        token_filter.push(reasoning, is_thinking=True)
                    if delta.content:
                        token_filter.push(delta.content, is_thinking=False)

                token_filter.flush()
                vis_text = "".join(visible_parts).strip()
                if not vis_text and thinking_parts:
                    return "".join(thinking_parts).strip()
                return vis_text

            resp = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore
                temperature=0.3,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            raise RuntimeError(
                f"Failed to generate LLM response: {exc}. "
                "Ensure tau is installed or configure --base-url / API keys."
            ) from exc

