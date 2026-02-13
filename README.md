# Adaptive RPG Storytelling (LLM-driven NPC Dialogue Game)

This repository is a graduation-project prototype for a multi-genre, LLM-driven narrative RPG.

Players can:
- describe a world in natural language (Chinese or English),
- explore a generated map,
- talk to NPCs,
- collect and deliver quest items,
- complete side quests and unlock the main finale,
- enter a final trial and receive an end-of-run recap page.

The project focus is **dynamic generation with LLM + engine constraints + RAG memory**, rather than hardcoded scripts.

---

## 1. Current Capabilities

### 1.1 Multi-genre world generation
- Generates worlds from user prompts (fantasy, campus, modern, sci-fi, etc.).
- Produces:
  - world background and starting hook,
  - locations and connectivity,
  - NPC roster with personality/agency parameters,
  - one main quest plus side quests,
  - map layout coordinates for visualization.

### 1.2 Language consistency (CN/EN)
- Detects the world prompt language.
- Keeps player-facing outputs in the same language as much as possible:
  dialogue, narration, quest text, and item names.

### 1.3 Quest system (main + side)
- Side quests are progressed via item collection and explicit delivery to NPCs.
- Main quest progression depends on side-quest rewards / key items.
- Main quest is finalized through a dedicated **Final Trial** (not auto-completed).
- Quest journal includes status and progress guidance.

### 1.4 Collection and inventory
- Per-location resource stock with depletion.
- Select item + quantity when collecting.
- Inventory accumulates quest and reward items.

### 1.5 Explicit delivery flow
- Delivery is a separate UI action (`Deliver to current NPC`).
- Chat turns do not auto-submit required items.
- Delivery updates quest progress and triggers rewards when complete.

### 1.6 NPC dialogue consistency improvements
- Per-NPC chat history view.
- Scrollable chat panel (page does not infinitely extend).
- Better continuity constraints for identity, tone, and prior context.
- NPC movement changes generate UI notices.

### 1.7 Interactive map
- Zoom + pan map interaction.
- Travel buttons for connected locations.
- Location-specific NPC list and resource stock update after movement.

### 1.8 Final recap page + story archive
- After passing Final Trial, user is redirected to a recap page containing:
  - collected item snapshot,
  - dialogue summary,
  - story summary,
  - plot arc summary,
  - epilogue.
- Recaps are archived and visible on the start page history panel.

---

## 2. Architecture Overview

### 2.1 Two-phase flow
1. **World Generation**: `world_prompt -> WorldSpec -> GameState initialization`
2. **Turn Pipeline**: `player input -> RAG retrieval -> LLM output -> validation -> state update -> persistence`

### 2.2 Design principles
- LLM handles generation and narration.
- Engine enforces executable rules (movement validity, quest state, item accounting).
- RAG injects world bible, location, NPC, and recent summaries to reduce context drift.
- Persistent state/logs support resume and replay.

### 2.3 Main modules
- `rpg_story/world/`: world generation, sanitation, consistency checks
- `rpg_story/engine/`: turn orchestration, state sync, delivery, agency/validation
- `rpg_story/rag/`: indexing and retrieval
- `rpg_story/ui/streamlit_app.py`: game UI
- `rpg_story/persistence/`: state/turn/summary storage
- `tests/`: world, quest, persistence, and orchestrator tests

---

## 3. Setup

### 3.1 Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Current dependencies include:
- `openai`
- `streamlit`
- `python-dotenv`
- `pyyaml`
- `pytest`

### 3.2 Configure API key
Default key environment variable: `DASHSCOPE_API_KEY`

Create `.env` in project root:
```env
DASHSCOPE_API_KEY=your_key_here
# optional overrides:
# LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# LLM_MODEL=qwen3-max
```

---

## 4. Run

### 4.1 Streamlit UI (recommended)
```bash
streamlit run rpg_story/ui/streamlit_app.py
```

Typical loop:
1. Create world from prompt.
2. Move on map.
3. Talk to local NPCs.
4. Collect resources.
5. Deliver required items to quest NPCs.
6. Complete side quests.
7. Trigger Final Trial with the finale NPC.
8. View recap page and archive entry.

### 4.2 CLI (debug)
```bash
python -m rpg_story.cli --config configs/config.yaml --npc <npc_id> --text "hello"
```

Optional flags:
- `--session <session_id>` to continue a saved run
- `--world <world_json_path>` to bootstrap from a world file

---

## 5. Configuration

Config file: `configs/config.yaml`

Important fields:
- `app.sessions_dir`: session data (`state.json`, `turns.jsonl`)
- `app.worlds_dir`: generated world files
- `llm.base_url`, `llm.model`, `llm.api_key_env`
- `rag.enabled`, `rag.top_k`, `rag.summary_window`
- `worldgen.locations_min/max`, `worldgen.npcs_min/max`

---

## 6. Persistence Layout

### 6.1 Per session
- `data/sessions/<session_id>/state.json`
- `data/sessions/<session_id>/turns.jsonl`

### 6.2 World snapshot
- `data/worlds/<session_id>/world.json`

### 6.3 Story recap archive
- `data/stories.jsonl`

---

## 7. Testing

Run all tests:
```bash
PYTHONPATH=. pytest -q
```

Key coverage areas:
- world generation validity
- quest/delivery/reward progression
- final-trial main-quest flow
- persistence and summary history

---

## 8. Known Limitations / Next Steps

- World generation still uses some rule-based fallback logic.
- Very long sessions can still produce occasional narrative drift.
- Final recap is hybrid (template + LLM); event-level summarization can be improved.
- UI is function-first; visual polish and animation can be expanded.

---

## 9. Thesis Alignment

This project targets **LLM-based NPC dialogue gameplay with consistency and controllability**:
- player-driven world generation,
- personality-aware NPC interaction,
- integration of dialogue, quests, map, and inventory,
- replayable finale summary and archive.

It aligns with thesis goal of moving from hardcoded content to **model-generated content under engine constraints**.
