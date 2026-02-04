"""Live Qwen integration test for world generation.

Run manually:
RUN_LIVE_LLM_TESTS=1 PYTHONPATH=. python -m pytest -q tests/test_worldgen_live_qwen.py
Requires DASHSCOPE_API_KEY in environment or .env (gitignored).
"""
from __future__ import annotations

import os
import pytest


# Skip by default unless explicitly enabled
if os.getenv("RUN_LIVE_LLM_TESTS") != "1":
    pytest.skip("Live LLM tests disabled (set RUN_LIVE_LLM_TESTS=1)", allow_module_level=True)

try:
    from openai import OpenAI  # noqa: F401
except Exception:
    pytest.skip("openai package not available", allow_module_level=True)

from rpg_story.config import load_config
from rpg_story.llm.client import QwenOpenAICompatibleClient
from rpg_story.world.generator import generate_world_spec, initialize_game_state

cfg = load_config("configs/config.yaml")
key_env = cfg.llm.api_key_env or "DASHSCOPE_API_KEY"
if not os.getenv(key_env):
    pytest.skip("Missing DASHSCOPE_API_KEY after load_config()", allow_module_level=True)


def test_live_worldgen_qwen():
    llm = QwenOpenAICompatibleClient(cfg)
    world_prompt = (
        "A small medieval riverside town with a market, an old bridge, and a nearby forest. "
        "Keep it grounded and low-magic."
    )
    world = generate_world_spec(cfg, llm, world_prompt)
    state = initialize_game_state(world, session_id="live_test")

    loc_ids = {loc.location_id for loc in world.locations}
    assert world.starting_location in loc_ids
    assert len(loc_ids) == len(world.locations)
    for loc in world.locations:
        for tgt in loc.connected_to:
            assert tgt in loc_ids

    assert state.player_location == world.starting_location
    assert set(state.npc_locations.keys()) == {npc.npc_id for npc in world.npcs}
    for npc_id, loc_id in state.npc_locations.items():
        assert loc_id in loc_ids
