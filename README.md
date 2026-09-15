# `tau-game`: Light-Duck Harness

An autonomous game-solving agent harness inspired by **Tufa Labs' Duck Harness** for **ARC-AGI-3**.

---

## 💡 The Core Philosophy

Traditional LLM game playing:
```
User: "What move?" ────► LLM: "UP" (Blind text guessing)
```

**Duck Harness approach**:
```
┌─────────────────┐
│       LLM       │
│  (Qwen / Brain) │
└────────┬────────┘
         │ writes Python code
         ▼
┌─────────────────┐
│   Python REPL   │  Exposes:
│     Sandbox     │  - current_frame.ascii & .segmentation
└────────┬────────┘  - valid_actions, history, world_model
         │           - action("RIGHT") or action(["UP", "UP"])
         │ executes action
         ▼
┌─────────────────┐
│   Environment   │  GridWorld / MiniArcGame / ARC3
│      (World)    │
└────────┬────────┘
         │ returns new frame observation
         └────────────────────────► (Next Turn)
```

Instead of forcing the LLM to output a single text token, the model is given an **interactive programming workspace**. The model can inspect objects, test hypotheses in Python, take actions, and maintain a persistent **World Model** across turns.

---

## 📂 Architecture & Directory Structure

```
tau-game/
├── pyproject.toml             # Package metadata & CLI entrypoint
├── tau.json                   # Tau extension registry
├── main.py                    # Direct CLI runner
│
├── agent/                     # ── Brain & Execution Loop ──
│   ├── agent.py               # DuckAgent observe-reason-code-act loop
│   ├── repl.py                # Python REPL sandbox with variable injection
│   ├── memory.py              # History, world model tracking & context eviction
│   └── prompts.py             # System prompt & observation builder
│
├── env/                       # ── Game Worlds ──
│   ├── environment.py         # BaseEnvironment interface, Frame & Transition
│   └── game.py                # GridWorld, MiniArcGame (key-door, sokoban, space toggle), Arc3Adapter
│
├── llm/                       # ── Model Client ──
│   └── client.py              # Reuses Tau's providers (openai, ollama, mlx, etc.) and sub-sessions
│
├── perception/                # ── Visual Perception ──
│   └── segmentation.py        # ASCII rendering & 4-connected component segmentation
│
├── extensions/                # ── Tau Extension ──
│   └── game/
│       └── extension.py       # Tau tool `game_run` & slash command `/game`
│
└── tests/                     # ── Test Suite ──
    ├── test_env.py
    ├── test_perception.py
    ├── test_repl.py
    ├── test_memory.py
    └── test_agent.py
```

---

## 🚀 Quick Start

### 1. Run the interactive Demo
Run a mock demonstration showing the Duck agent's reasoning, code execution, and world model update step-by-step:
```bash
python tau-game/main.py --demo
```

### 2. Run with your active LLM (via Tau)
Reuses your existing `tau` configuration (`~/.tau/config.toml`):
```bash
# Play GridWorld with default model
python tau-game/main.py --game gridworld

# Play Mini-ARC puzzle with live thinking streamed in real-time
python tau-game/main.py --game mini-arc --level 3 --provider openai --base-url http://localhost:8080/v1 --model qwen --show-thinking
```

### 3. Use inside Tau CLI (Extension)
In Tau REPL:
```text
/game gridworld
/game mini-arc 3 --show-thinking
```

Or prompt Tau:
```text
"Solve the mini-arc puzzle level 1 using the game_run tool"
```

---

## 🔍 The 5 Core Concepts to Study

1. **Python REPL as Workspace (`agent/repl.py`)**:
   Notice how `current_frame`, `valid_actions`, and `action(...)` are injected into the REPL namespace. The LLM can run loops, analyze arrays, or batch actions.

2. **Perception via Object Segmentation (`perception/segmentation.py`)**:
   Instead of viewing 64x64 raw numbers, `connected_components` turns pixels into distinct objects with color, bounding box, centroid, and shape hashes.

3. **World Model Persistence (`agent/memory.py`)**:
   The agent writes discoveries under `World model:` which are parsed and re-injected into subsequent turns.

4. **Infinite Play via Context Eviction (`agent/memory.py`)**:
   `trim_messages` discards older conversation turns when context grows, while strictly preserving the system prompt and world model.

5. **Extensibility for ARC-AGI-3 (`env/game.py`)**:
   The `Arc3Adapter` connects any external ARC-AGI-3 GameAPI or TAAF competition server directly to this harness.

---

## 📽️ Presentation Slides / Slide Thuyết Trình

Interactive and export-ready presentation decks are included in both English and Vietnamese:

### 🇻🇳 Tiếng Việt:
- **Interactive Web Deck**: [`tau-game/slides_vi.html`](./slides_vi.html)
  - Giao diện dark mode hiện đại, điều hướng phím (`←`/`→`, `Space`, `F`), trình xem tương tác Cấp độ 1–7, và bảng ghi chú thuyết trình (`N`).
  - Mở trực tiếp trên trình duyệt: `open tau-game/slides_vi.html`.
- **Bản Xuất PDF / Marp**: [Artifact `slides_vi.md`](file:///Users/trantandat/.gemini/antigravity-ide/brain/86245f57-7bc3-46d1-af55-c1ad2886c40c/slides_vi.md)
  - 12 slide đầy đủ chuyên sâu bằng Tiếng Việt, tương thích Marp CLI và Reveal.js.

### 🇬🇧 English:
- **Interactive Web Deck**: [`tau-game/slides.html`](./slides.html)
  - Dark-mode aesthetics, keyboard navigation (`←`/`→`, `Space`, `F`), live interactive Level 1–7 switcher, and real-time speaker notes panel (`N`).
  - Open directly in any browser: `open tau-game/slides.html`.
- **Marp / PDF Export Source**: [Artifact `slides.md`](file:///Users/trantandat/.gemini/antigravity-ide/brain/86245f57-7bc3-46d1-af55-c1ad2886c40c/slides.md)
  - 12 comprehensive slides ready for Marp CLI, Reveal.js, or PDF export.


