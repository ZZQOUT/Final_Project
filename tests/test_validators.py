from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from rpg_story.config import load_config
from rpg_story.engine.orchestrator import TurnPipeline
from rpg_story.engine.validators import build_graph, is_reachable, validate_npc_move
from rpg_story.llm.client import MockLLMClient
from rpg_story.models.turn import NPCMove
from rpg_story.models.world import WorldBibleRules, LocationSpec, NPCProfile, WorldSpec, GameState
from rpg_story.persistence.store import read_turn_logs


def make_world() -> WorldSpec:
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
        profession="Courier",
        traits=["brave"],
        goals=["deliver messages"],
        starting_location="A",
        obedience_level=0.8,
        stubbornness=0.2,
        risk_tolerance=0.7,
        disposition_to_player=1,
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


def make_state(world: WorldSpec) -> GameState:
    return GameState(
        session_id="sess_test",
        created_at=datetime.now(timezone.utc).isoformat(),
        world=world,
        player_location="A",
        npc_locations={"npc_1": "A"},
    )


def test_bfs_reachability():
    world = make_world()
    graph = build_graph(world)
    assert is_reachable(graph, "A", "C") is True
    assert is_reachable(graph, "C", "A") is False


def test_validate_to_location_unknown():
    world = make_world()
    state = make_state(world)
    move = NPCMove(
        npc_id="npc_1",
        from_location="A",
        to_location="Z",
        trigger="player_instruction",
        reason="test",
        permanence="temporary",
        confidence=0.5,
    )
    ok, reason = validate_npc_move(move, state, world)
    assert ok is False
    assert "to_location" in reason


def test_validate_from_location_mismatch():
    world = make_world()
    state = make_state(world)
    move = NPCMove(
        npc_id="npc_1",
        from_location="B",
        to_location="C",
        trigger="player_instruction",
        reason="test",
        permanence="temporary",
        confidence=0.5,
    )
    ok, reason = validate_npc_move(move, state, world)
    assert ok is False
    assert "from_location mismatch" in reason


def test_validate_npc_missing():
    world = make_world()
    state = make_state(world)
    move = NPCMove(
        npc_id="npc_999",
        from_location="A",
        to_location="B",
        trigger="player_instruction",
        reason="test",
        permanence="temporary",
        confidence=0.5,
    )
    ok, reason = validate_npc_move(move, state, world)
    assert ok is False
    assert "npc_id" in reason


def test_orchestrator_logs_move_rejections(tmp_path: Path):
    cfg = load_config("configs/config.yaml")
    sessions_root = tmp_path / "sessions"

    world = make_world()
    state = make_state(world)

    output_json = (
        "{"
        '"narration":"OK",'
        '"npc_dialogue":[], '
        '"world_updates":{'
        '  "player_location":"A",'
        '  "npc_moves":[{'
        '    "npc_id":"npc_1",'
        '    "from_location":"A",'
        '    "to_location":"B",'
        '    "trigger":"player_instruction",'
        '    "reason":"player_request",'
        '    "permanence":"temporary",'
        '    "confidence":0.9'
        '  },{'
        '    "npc_id":"npc_1",'
        '    "from_location":"B",'
        '    "to_location":"Z",'
        '    "trigger":"player_instruction",'
        '    "reason":"bad_target",'
        '    "permanence":"temporary",'
        '    "confidence":0.5'
        '  }],'
        '  "flags_delta":{},'
        '  "quest_updates":{}'
        '},'
        '"memory_summary":"Summary.",'
        '"safety":{"refuse":false,"reason":null}'
        "}"
    )

    llm = MockLLMClient([output_json])
    pipeline = TurnPipeline(cfg=cfg, llm_client=llm, sessions_root=sessions_root)

    updated_state, _output, log_record = pipeline.run_turn(state, "Hello", "npc_1")
    assert updated_state.npc_locations["npc_1"] == "B"
    rejections = log_record.get("move_rejections", [])
    assert len(rejections) == 1
    reason = rejections[0].get("reason", "")
    assert "from_location mismatch" in reason or "to_location" in reason
    assert log_record.get("move_rejected_count") == 1

    logs = read_turn_logs("sess_test", sessions_root)
    assert len(logs) == 1
    assert len(logs[0].get("move_rejections", [])) == 1
