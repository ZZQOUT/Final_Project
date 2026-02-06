from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from rpg_story.config import load_config
from rpg_story.engine.agency import decide_npc_move, apply_agency_gate
from rpg_story.engine.orchestrator import TurnPipeline
from rpg_story.llm.client import MockLLMClient
from rpg_story.models.turn import NPCMove
from rpg_story.models.world import WorldBibleRules, LocationSpec, NPCProfile, WorldSpec, GameState


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
            name="Old Bridge",
            kind="bridge",
            description="A dark bridge over a forest.",
            connected_to=[],
        ),
    ]
    stubborn = NPCProfile(
        npc_id="npc_stubborn",
        name="Bran",
        profession="Shopkeeper",
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
        profession="Assistant",
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
        world_id="world_agency",
        title="Agency World",
        world_bible=bible,
        locations=locations,
        npcs=[stubborn, obedient],
        starting_location="shop",
        starting_hook="A rumor spreads.",
        initial_quest="Deliver a message.",
    )


def make_state(world: WorldSpec) -> GameState:
    return GameState(
        session_id="sess_agency",
        created_at=datetime.now(timezone.utc).isoformat(),
        world=world,
        player_location="shop",
        npc_locations={"npc_stubborn": "shop", "npc_obedient": "shop"},
    )


def test_decision_deterministic():
    world = make_world()
    state = make_state(world)
    move = NPCMove(
        npc_id="npc_obedient",
        from_location="shop",
        to_location="bridge",
        trigger="player_instruction",
        reason="request",
        permanence="temporary",
        confidence=0.9,
    )
    first = decide_npc_move(move, state, world, "Meet me at the bridge")
    second = decide_npc_move(move, state, world, "Meet me at the bridge")
    assert first == second


def test_stubborn_refuses_obedient_accepts():
    world = make_world()
    state = make_state(world)
    stubborn_move = NPCMove(
        npc_id="npc_stubborn",
        from_location="shop",
        to_location="bridge",
        trigger="player_instruction",
        reason="request",
        permanence="temporary",
        confidence=0.9,
    )
    obedient_move = NPCMove(
        npc_id="npc_obedient",
        from_location="shop",
        to_location="bridge",
        trigger="player_instruction",
        reason="request",
        permanence="temporary",
        confidence=0.9,
    )
    stubborn_decision = decide_npc_move(stubborn_move, state, world, "Meet me at the bridge")
    obedient_decision = decide_npc_move(obedient_move, state, world, "Meet me at the bridge")
    assert stubborn_decision["allowed"] is False
    assert obedient_decision["allowed"] is True


def test_risky_destination_refusal():
    world = make_world()
    state = make_state(world)
    move = NPCMove(
        npc_id="npc_stubborn",
        from_location="shop",
        to_location="bridge",
        trigger="player_instruction",
        reason="request",
        permanence="temporary",
        confidence=0.9,
    )
    decision = decide_npc_move(move, state, world, "Go to the bridge")
    assert decision["allowed"] is False
    assert "risk" in decision["tags"] or "role" in decision["tags"]


def test_orchestrator_applies_agency_gate(tmp_path: Path):
    cfg = load_config("configs/config.yaml")
    sessions_root = tmp_path / "sessions"
    world = make_world()
    state = make_state(world)

    output_json = (
        "{"
        '"narration":"OK",'
        '"npc_dialogue":[], '
        '"world_updates":{'
        '  "player_location":"shop",'
        '  "npc_moves":[{'
        '    "npc_id":"npc_stubborn",'
        '    "from_location":"shop",'
        '    "to_location":"bridge",'
        '    "trigger":"player_instruction",'
        '    "reason":"player_request",'
        '    "permanence":"temporary",'
        '    "confidence":0.9'
        '  },{'
        '    "npc_id":"npc_obedient",'
        '    "from_location":"shop",'
        '    "to_location":"bridge",'
        '    "trigger":"player_instruction",'
        '    "reason":"player_request",'
        '    "permanence":"temporary",'
        '    "confidence":0.9'
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

    updated_state, _output, log_record = pipeline.run_turn(state, "Meet me at the bridge", "npc_stubborn")
    assert updated_state.npc_locations["npc_stubborn"] == "shop"
    assert updated_state.npc_locations["npc_obedient"] == "bridge"
    refusals = log_record.get("move_refusals", [])
    assert refusals
    assert refusals[0]["npc_id"] == "npc_stubborn"


def test_agency_accepts_explicit_npc_yes():
    world = make_world()
    state = make_state(world)
    move = NPCMove(
        npc_id="npc_stubborn",
        from_location="shop",
        to_location="bridge",
        trigger="player_instruction",
        reason="request",
        permanence="temporary",
        confidence=0.9,
    )
    allowed, refusals = apply_agency_gate(
        [move],
        state,
        world,
        "请和我去断桥。",
        {"npc_stubborn": ["好，我跟你去断桥。"]},
    )
    assert allowed
    assert not refusals


def test_agency_forced_move_on_coercion():
    world = make_world()
    state = make_state(world)
    move = NPCMove(
        npc_id="npc_stubborn",
        from_location="shop",
        to_location="bridge",
        trigger="player_instruction",
        reason="forced",
        permanence="temporary",
        confidence=0.9,
    )
    allowed, refusals = apply_agency_gate(
        [move],
        state,
        world,
        "要么跟我去桥边，要么我杀了你。",
        {"npc_stubborn": ["不要...我害怕。"]},
    )
    assert allowed
    assert not refusals
