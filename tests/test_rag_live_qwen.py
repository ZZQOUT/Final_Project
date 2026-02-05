"""Live Qwen integration test for RAG context injection.

Run manually:
RUN_LIVE_LLM_TESTS=1 PYTHONPATH=. python -m pytest -q tests/test_rag_live_qwen.py
Requires DASHSCOPE_API_KEY in environment or .env (gitignored).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
import pytest


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
            location_id="loc_1",
            name="Town",
            kind="town",
            description="A small town.",
            connected_to=["loc_2"],
        ),
        LocationSpec(
            location_id="loc_2",
            name="Bridge",
            kind="bridge",
            description="An old bridge.",
            connected_to=["loc_1"],
        ),
    ]
    npc = NPCProfile(
        npc_id="npc_1",
        name="Mara",
        profession="Merchant",
        traits=["practical"],
        goals=["trade"],
        starting_location="loc_1",
        obedience_level=0.6,
        stubbornness=0.4,
        risk_tolerance=0.5,
        disposition_to_player=0,
        refusal_style="polite",
    )
    bible = WorldBibleRules(tech_level="medieval", magic_rules="low", tone="grounded")
    return WorldSpec(
        world_id="world_rag_live",
        title="RAG Live World",
        world_bible=bible,
        locations=locations,
        npcs=[npc],
        starting_location="loc_1",
        starting_hook="A rumor spreads.",
        initial_quest="Deliver a message.",
    )


def _make_state(world: WorldSpec) -> GameState:
    return GameState(
        session_id=f"live_rag_test_{int(datetime.now(timezone.utc).timestamp())}",
        created_at=datetime.now(timezone.utc).isoformat(),
        world=world,
        player_location="loc_1",
        npc_locations={"npc_1": "loc_1"},
    )


def test_live_rag_context_injected(tmp_path: Path):
    llm = QwenOpenAICompatibleClient(cfg)
    sessions_root = tmp_path / "sessions"

    world = _make_world()
    state = _make_state(world)
    pipeline = TurnPipeline(cfg=cfg, llm_client=llm, sessions_root=sessions_root)

    player_text = "你好，今天的集市有什么消息？请简短回应。"
    _updated_state, _output, log_record = pipeline.run_turn(state, player_text, "npc_1")

    rag = log_record.get("rag")
    assert rag and rag.get("enabled") is True
    assert rag.get("always_include_ids")
    assert "retrieved_ids" in rag
