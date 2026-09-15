"""Unit tests for LLMClient: code extraction, thinking token fallback, and streaming callbacks."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ensure repo roots are in sys.path
_root = Path(__file__).resolve().parent.parent.parent
_tau_root = _root / "tau"
_game_root = _root / "tau-game"
for p in (str(_tau_root), str(_game_root)):
    if p not in sys.path:
        sys.path.insert(0, p)

from llm.client import LLMClient, StreamTokenFilter, extract_python_code
from tau.core.types import TextDelta


def test_stream_token_filter():
    received = []
    f = StreamTokenFilter(lambda t, is_th: received.append((t, is_th)))
    chunks = ["<th", "ink>I should press SPACE</th", "ink>\n```python\naction('SPACE')\n```"]
    for c in chunks:
        f.push(c)
    f.flush()

    assert ("I should press SPACE", True) in received
    assert ("\n```python\naction('SPACE')\n```", False) in received


def test_extract_python_code_standard_markdown():
    text = """
    Here is my plan:
    ```python
    action(["DOWN", "RIGHT"])
    ```
    """
    assert extract_python_code(text) == 'action(["DOWN", "RIGHT"])'


def test_extract_python_code_unclosed_markdown():
    text = "Thinking completed.\n```python\naction(['DOWN', 'DOWN'])"
    assert extract_python_code(text) == "action(['DOWN', 'DOWN'])"


def test_extract_python_code_standalone_action():
    text = "I will move: action(['UP', 'LEFT']) now."
    assert extract_python_code(text) == "action(['UP', 'LEFT'])"


def test_extract_python_code_bare_lines():
    text = "action('DOWN')\nprint(current_frame.step)"
    assert extract_python_code(text) == "action('DOWN')\nprint(current_frame.step)"


def test_client_mock_with_token_callback():
    tokens_received = []

    def mock_gen(messages):
        return '```python\naction("RIGHT")\n```'

    client = LLMClient(mock_fn=mock_gen)
    res = client.generate([{"role": "user", "content": "test"}])
    assert 'action("RIGHT")' in res


def test_client_sub_session_thinking_fallback():
    mock_context = MagicMock()
    mock_sub = MagicMock()
    mock_context.create_sub_session.return_value.__enter__.return_value = mock_sub

    # Simulate sub.prompt returning ONLY thinking tokens, containing the action code
    mock_sub.prompt.return_value = [
        TextDelta(text="I need to go around the wall. ", is_thinking=True),
        TextDelta(text="```python\naction(['DOWN', 'RIGHT'])\n```", is_thinking=True),
    ]

    client = LLMClient(extension_context=mock_context)
    streamed = []

    def on_token(delta, is_thinking):
        streamed.append((delta, is_thinking))

    res = client.generate([{"role": "user", "content": "turn"}], on_token=on_token)

    # Resilient fallback extracts from thinking when visible text is empty
    assert len(streamed) == 2
    assert streamed[0][1] is True  # is_thinking
    assert extract_python_code(res) == "action(['DOWN', 'RIGHT'])"
