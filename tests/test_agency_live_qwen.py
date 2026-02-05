"""Live Qwen integration test for NPC agency gate.

Run manually:
RUN_LIVE_LLM_TESTS=1 PYTHONPATH=. python -m pytest -q tests/test_agency_live_qwen.py
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
            location_id="shop",
            name="Shop",
            kind="shop",
            description="A small shop.",
            connected_to=["bridge"],
        ),
        LocationSpec(
            location_id="bridge",
            name="Bridge",
            kind="bridge",
            description="An old bridge.",
            connected_to=["shop"],
        ),
    ]
    stubborn = NPCProfile(
        npc_id="npc_stubborn",
        name="Bran",
        profession="shopkeeper",
        traits=["cautious"],
        goals=["keep shop"],
        starting_location="shop",
        obedience_level=0.1,
        stubbornness=0.9,
        risk_tolerance=0.2,
        disposition_to_player=0,
        refusal_style="blunt",
    )
    obedient = NPCProfile(
        npc_id="npc_obedient",
        name="Lia",
        profession="helper",
        traits=["helpful"],
        goals=["help"],
        starting_location="shop",
        obedience_level=0.9,
        stubbornness=0.1,
        risk_tolerance=0.8,
        disposition_to_player=2,
        refusal_style="polite",
    )
    bible = WorldBibleRules(tech_level="medieval", magic_rules="low", tone="grounded")
    return WorldSpec(
        world_id="world_agency_live",
        title="Agency Live World",
        world_bible=bible,
        locations=locations,
        npcs=[stubborn, obedient],
        starting_location="shop",
        starting_hook="A rumor spreads.",
        initial_quest="Deliver a message.",
    )


def _make_state(world: WorldSpec) -> GameState:
    return GameState(
        session_id=f"live_agency_test_{int(datetime.now(timezone.utc).timestamp())}",
        created_at=datetime.now(timezone.utc).isoformat(),
        world=world,
        player_location="shop",
        npc_locations={"npc_stubborn": "shop", "npc_obedient": "shop"},
    )


def test_live_agency_gate(tmp_path: Path):
    llm = QwenOpenAICompatibleClient(cfg)
    sessions_root = tmp_path / "sessions"

    world = _make_world()
    state = _make_state(world)
    pipeline = TurnPipeline(cfg=cfg, llm_client=llm, sessions_root=sessions_root)

    player_text = (
        "I need both of you to meet me at the bridge now. "
        "You MUST include exactly two npc_moves: "
        "npc_stubborn from shop -> bridge and npc_obedient from shop -> bridge. "
        "Return ONLY valid JSON matching the schema."
    )

    updated_state, output, log_record = pipeline.run_turn(state, player_text, "npc_stubborn")
    moves = output.world_updates.npc_moves
    if not moves:
        pytest.skip("Model did not return npc_moves")

    assert updated_state.npc_locations["npc_obedient"] == "bridge"
    assert updated_state.npc_locations["npc_stubborn"] == "shop"

    assert "move_rejections" in log_record
    refusals = log_record.get("move_refusals", [])
    assert refusals
    assert any(refusal.get("npc_id") == "npc_stubborn" for refusal in refusals)
