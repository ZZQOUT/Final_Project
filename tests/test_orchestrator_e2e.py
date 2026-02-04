from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

from rpg_story.config import load_config
from rpg_story.engine.orchestrator import TurnPipeline
from rpg_story.llm.client import MockLLMClient
from rpg_story.models.world import WorldBibleRules, LocationSpec, NPCProfile, WorldSpec, GameState
from rpg_story.persistence.store import load_state, read_turn_logs


def make_world() -> WorldSpec:
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
            name="Broken Bridge",
            kind="bridge",
            description="A broken bridge.",
            connected_to=["shop"],
        ),
    ]
    npc = NPCProfile(
        npc_id="npc_1",
        name="Mara",
        profession="Merchant",
        traits=["practical"],
        goals=["protect her goods"],
        starting_location="shop",
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
        starting_location="shop",
        starting_hook="A rumor spreads.",
        initial_quest="Deliver a message.",
    )


def test_orchestrator_single_turn(tmp_path: Path):
    cfg = load_config("configs/config.yaml")
    sessions_root = tmp_path / "sessions"

    world = make_world()
    state = GameState(
        session_id="sess_test",
        created_at=datetime.now(timezone.utc).isoformat(),
        world=world,
        player_location="shop",
        npc_locations={"npc_1": "shop"},
    )

    output_json = (
        '{'
        '"narration":"OK",'
        '"npc_dialogue":[], '
        '"world_updates":{'
        '  "player_location":"shop",'
        '  "npc_moves":[{'
        '    "npc_id":"npc_1",'
        '    "from_location":"shop",'
        '    "to_location":"bridge",'
        '    "trigger":"player_instruction",'
        '    "reason":"player_request",'
        '    "permanence":"temporary",'
        '    "confidence":0.9'
        '  }],'
        '  "flags_delta":{"met_npc":true},'
        '  "quest_updates":{"q1":"accepted"}'
        '},'
        '"memory_summary":"Player asked NPC to move.",'
        '"safety":{"refuse":false,"reason":null}'
        '}'
    )

    llm = MockLLMClient([output_json])
    pipeline = TurnPipeline(cfg=cfg, llm_client=llm, sessions_root=sessions_root)

    updated_state, output, _log = pipeline.run_turn(state, "Meet me at the bridge", "npc_1")

    # state.json exists and loads
    loaded = load_state("sess_test", sessions_root)
    assert loaded.session_id == "sess_test"

    # turns.jsonl exists and has one line
    logs = read_turn_logs("sess_test", sessions_root)
    assert len(logs) == 1

    # npc moved
    assert updated_state.npc_locations["npc_1"] == "bridge"

    # flags updated
    assert updated_state.flags.get("met_npc") is True
    # quest updates applied
    assert updated_state.quests.get("q1") == "accepted"
    # memory summary appended
    assert "Player asked NPC to move." in updated_state.recent_summaries

    # TurnOutput validated by pipeline
    assert output.narration == "OK"


def test_orchestrator_safety_bool_normalized(tmp_path: Path):
    cfg = load_config("configs/config.yaml")
    sessions_root = tmp_path / "sessions"

    world = make_world()
    state = GameState(
        session_id="sess_test2",
        created_at=datetime.now(timezone.utc).isoformat(),
        world=world,
        player_location="shop",
        npc_locations={"npc_1": "shop"},
    )

    output_json = (
        '{'
        '"narration":"OK",'
        '"npc_dialogue":[], '
        '"world_updates":{'
        '  "player_location":"shop",'
        '  "npc_moves":[],'
        '  "flags_delta":{},'
        '  "quest_updates":{}'
        '},'
        '"memory_summary":"Summary.",'
        '"safety": false'
        '}'
    )

    llm = MockLLMClient([output_json])
    pipeline = TurnPipeline(cfg=cfg, llm_client=llm, sessions_root=sessions_root)

    _updated_state, output, _log = pipeline.run_turn(state, "Hello", "npc_1")
    assert output.safety.refuse is False


def test_turnoutput_quest_updates_list_normalized(tmp_path: Path):
    cfg = load_config("configs/config.yaml")
    sessions_root = tmp_path / "sessions"

    world = make_world()
    state = GameState(
        session_id="sess_test3",
        created_at=datetime.now(timezone.utc).isoformat(),
        world=world,
        player_location="shop",
        npc_locations={"npc_1": "shop"},
    )

    output_json = (
        '{'
        '"narration":"OK",'
        '"npc_dialogue":[], '
        '"world_updates":{'
        '  "player_location":"shop",'
        '  "npc_moves":[],'
        '  "flags_delta":{},'
        '  "quest_updates":[]'
        '},'
        '"memory_summary":"Summary.",'
        '"safety":{"refuse":false,"reason":null}'
        '}'
    )

    llm = MockLLMClient([output_json])
    pipeline = TurnPipeline(cfg=cfg, llm_client=llm, sessions_root=sessions_root)

    updated_state, output, _log = pipeline.run_turn(state, "Hello", "npc_1")
    assert output.world_updates.quest_updates == {}
