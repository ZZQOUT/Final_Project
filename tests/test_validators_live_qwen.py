"""Live Qwen integration test for NPC move validators.

Run manually:
RUN_LIVE_LLM_TESTS=1 PYTHONPATH=. python -m pytest -q tests/test_validators_live_qwen.py
Requires DASHSCOPE_API_KEY in environment or .env (gitignored).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
import pytest


# Skip by default unless explicitly enabled
if os.getenv("RUN_LIVE_LLM_TESTS") != "1":
    pytest.skip("Live LLM tests disabled (set RUN_LIVE_LLM_TESTS=1)", allow_module_level=True)

try:
    from openai import OpenAI  # noqa: F401
except Exception:
    pytest.skip("openai package not available", allow_module_level=True)

from rpg_story.config import load_config
from rpg_story.engine.orchestrator import TurnPipeline
from rpg_story.llm.client import QwenOpenAICompatibleClient
from rpg_story.models.world import WorldBibleRules, LocationSpec, NPCProfile, WorldSpec, GameState

cfg = load_config("configs/config.yaml")
key_env = cfg.llm.api_key_env or "DASHSCOPE_API_KEY"
if not os.getenv(key_env):
    pytest.skip("Missing DASHSCOPE_API_KEY after load_config()", allow_module_level=True)


def _make_world() -> WorldSpec:
    locations = [
        LocationSpec(
            location_id="A",
            name="A",
            kind="town",
            description="A",
            connected_to=["B"],
        ),
        LocationSpec(
            location_id="B",
            name="B",
            kind="road",
            description="B",
            connected_to=["C"],
        ),
        LocationSpec(
            location_id="C",
            name="C",
            kind="forest",
            description="C",
            connected_to=[],
        ),
    ]
    npc = NPCProfile(
        npc_id="npc_1",
        name="Mara",
        profession="Merchant",
        traits=["practical"],
        goals=["trade"],
        starting_location="A",
        obedience_level=0.5,
        stubbornness=0.5,
        risk_tolerance=0.5,
        disposition_to_player=0,
        refusal_style="polite",
    )
    bible = WorldBibleRules(tech_level="medieval", magic_rules="low", tone="grounded")
    return WorldSpec(
        world_id="world_demo",
        title="Demo World",
        world_bible=bible,
        locations=locations,
        npcs=[npc],
        starting_location="A",
        starting_hook="A rumor spreads.",
        initial_quest="Deliver a message.",
    )


def _make_state(world: WorldSpec) -> GameState:
    return GameState(
        session_id="sess_live",
        created_at=datetime.now(timezone.utc).isoformat(),
        world=world,
        player_location="A",
        npc_locations={"npc_1": "A"},
    )


def test_live_validators_reject_invalid_move(tmp_path: Path):
    llm = QwenOpenAICompatibleClient(cfg)
    sessions_root = tmp_path / "sessions"

    world = _make_world()
    state = _make_state(world)
    pipeline = TurnPipeline(cfg=cfg, llm_client=llm, sessions_root=sessions_root)

    player_text = (
        "For testing, you MUST include exactly one npc_moves entry with: "
        "npc_id=\"npc_1\", from_location=\"B\", to_location=\"C\". "
        "Also include a short narration. Return ONLY valid JSON matching the schema."
    )

    updated_state, _output, log_record = pipeline.run_turn(state, player_text, "npc_1")

    assert "move_rejections" in log_record
    rejections = log_record.get("move_rejections", [])
    if rejections:
        assert updated_state.npc_locations["npc_1"] == "A"
        assert "from_location mismatch" in rejections[0].get("reason", "")
