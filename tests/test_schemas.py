import pytest

from rpg_story.config import load_config, AppConfig
from rpg_story.models.world import (
    LocationSpec,
    WorldBibleRules,
    NPCProfile,
    WorldSpec,
    GameState,
)
from rpg_story.llm.schemas import validate_turn_output


def make_min_world() -> WorldSpec:
    locs = [
        LocationSpec(
            location_id="loc_001",
            name="Town",
            kind="town",
            description="A small town.",
            connected_to=["loc_002"],
        ),
        LocationSpec(
            location_id="loc_002",
            name="Forest",
            kind="forest",
            description="A dark forest.",
            connected_to=["loc_001"],
        ),
    ]
    npcs = [
        NPCProfile(
            npc_id="npc_001",
            name="Ala",
            profession="Merchant",
            traits=["curious"],
            goals=["sell goods"],
            starting_location="loc_001",
            obedience_level=0.5,
            stubbornness=0.5,
            risk_tolerance=0.5,
            disposition_to_player=0,
            refusal_style="polite",
        )
    ]
    bible = WorldBibleRules(
        tech_level="medieval",
        magic_rules="low",
        tone="adventurous",
    )
    return WorldSpec(
        world_id="world_001",
        title="Test World",
        world_bible=bible,
        locations=locs,
        npcs=npcs,
        starting_location="loc_001",
        starting_hook="You arrive.",
        initial_quest="Find the relic.",
    )


def test_worldspec_valid():
    world = make_min_world()
    assert world.world_id == "world_001"


def test_worldspec_missing_starting_location():
    world = make_min_world().model_copy()
    world.starting_location = "loc_999"
    with pytest.raises(ValueError):
        WorldSpec.model_validate(world.model_dump())


def test_worldspec_invalid_connected_to():
    world = make_min_world().model_copy()
    world.locations[0].connected_to = ["loc_999"]
    with pytest.raises(ValueError):
        WorldSpec.model_validate(world.model_dump())


def test_worldspec_duplicate_npc_id():
    world = make_min_world().model_copy()
    dup = world.npcs[0].model_copy()
    world.npcs.append(dup)
    with pytest.raises(ValueError):
        WorldSpec.model_validate(world.model_dump())


def test_worldspec_bidirectional_edges_helper():
    world = make_min_world()
    # make one-way edge
    world.locations[0].connected_to = ["loc_002"]
    world.locations[1].connected_to = []
    with pytest.raises(ValueError):
        world.validate_bidirectional_edges(strict=True)


def test_npcprofile_range_checks():
    with pytest.raises(ValueError):
        NPCProfile(
            npc_id="npc_bad",
            name="Bad",
            profession="None",
            traits=[],
            goals=[],
            starting_location="loc_001",
            obedience_level=0.5,
            stubbornness=1.2,
            risk_tolerance=0.5,
            disposition_to_player=0,
            refusal_style="",
        )


def test_turnoutput_valid():
    data = {
        "narration": "OK",
        "npc_dialogue": [],
        "world_updates": {"npc_moves": [], "flags_delta": {}, "quest_updates": {}},
        "memory_summary": "",
        "safety": {"refuse": False, "reason": None},
    }
    out = validate_turn_output(data)
    assert out.narration == "OK"


def test_turnoutput_invalid_confidence():
    data = {
        "narration": "OK",
        "npc_dialogue": [],
        "world_updates": {
            "npc_moves": [
                {
                    "npc_id": "npc_001",
                    "from_location": "loc_001",
                    "to_location": "loc_002",
                    "trigger": "player_instruction",
                    "reason": "player_request",
                    "permanence": "temporary",
                    "confidence": 2.0,
                }
            ],
            "flags_delta": {},
            "quest_updates": {},
        },
        "memory_summary": "",
        "safety": {"refuse": False, "reason": None},
    }
    with pytest.raises(ValueError):
        validate_turn_output(data)


def test_turnoutput_invalid_trigger_permanence():
    data = {
        "narration": "OK",
        "npc_dialogue": [],
        "world_updates": {
            "npc_moves": [
                {
                    "npc_id": "npc_001",
                    "from_location": "loc_001",
                    "to_location": "loc_002",
                    "trigger": "bad_trigger",
                    "reason": "player_request",
                    "permanence": "temporary",
                    "confidence": 0.5,
                }
            ],
            "flags_delta": {},
            "quest_updates": {},
        },
        "memory_summary": "",
        "safety": {"refuse": False, "reason": None},
    }
    with pytest.raises(ValueError):
        validate_turn_output(data)


def test_gamestate_invalid_npc_locations_unknown_key():
    world = make_min_world()
    with pytest.raises(ValueError):
        GameState(
            session_id="sess_1",
            created_at="2025-01-01T00:00:00Z",
            world=world,
            player_location="loc_001",
            npc_locations={"npc_999": "loc_001"},
        )


def test_gamestate_missing_npc_location():
    world = make_min_world()
    with pytest.raises(ValueError):
        GameState(
            session_id="sess_1",
            created_at="2025-01-01T00:00:00Z",
            world=world,
            player_location="loc_001",
            npc_locations={},  # missing npc_001
        )


def test_gamestate_invalid_location_value():
    world = make_min_world()
    with pytest.raises(ValueError):
        GameState(
            session_id="sess_1",
            created_at="2025-01-01T00:00:00Z",
            world=world,
            player_location="loc_001",
            npc_locations={"npc_001": "loc_999"},
        )


def test_config_loader_env_override(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "override-model")
    cfg = load_config("configs/config.yaml")
    assert isinstance(cfg, AppConfig)
    assert cfg.llm.model == "override-model"
