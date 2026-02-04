from __future__ import annotations

from pathlib import Path

from rpg_story.config import load_config
from rpg_story.llm.client import MockLLMClient
from rpg_story.models.world import WorldSpec, WorldBibleRules, LocationSpec, NPCProfile
from rpg_story.persistence.store import load_state
from rpg_story.world.generator import generate_world_spec, initialize_game_state, create_new_session
from rpg_story.world.consistency import find_anachronisms


def valid_world_json() -> str:
    return (
        '{'
        '"world_id":"world_001",'
        '"title":"Test World",'
        '"world_bible":{"tech_level":"medieval","magic_rules":"low","tone":"adventurous"},'
        '"locations":['
        ' {"location_id":"loc_001","name":"Town","kind":"town","description":"A small town.","connected_to":["loc_002"],"tags":[]},'
        ' {"location_id":"loc_002","name":"Forest","kind":"forest","description":"A dark forest.","connected_to":["loc_001"],"tags":[]}'
        '],'
        '"npcs":['
        ' {"npc_id":"npc_001","name":"Ala","profession":"Merchant","traits":["curious"],"goals":["trade"],"starting_location":"loc_001",'
        '  "obedience_level":0.5,"stubbornness":0.5,"risk_tolerance":0.5,"disposition_to_player":0,"refusal_style":"polite"},'
        ' {"npc_id":"npc_002","name":"Bren","profession":"Guard","traits":["stern"],"goals":["protect"],"starting_location":"loc_002",'
        '  "obedience_level":0.6,"stubbornness":0.4,"risk_tolerance":0.4,"disposition_to_player":0,"refusal_style":"firm"}'
        '],'
        '"starting_location":"loc_001",'
        '"starting_hook":"You arrive.",'
        '"initial_quest":"Find the relic."'
        '}'
    )


def invalid_connected_world_json() -> str:
    return (
        '{'
        '"world_id":"world_bad",'
        '"title":"Bad World",'
        '"world_bible":{"tech_level":"medieval","magic_rules":"low","tone":"adventurous"},'
        '"locations":['
        ' {"location_id":"loc_001","name":"Town","kind":"town","description":"A small town.","connected_to":["loc_999"],"tags":[]}'
        '],'
        '"npcs":['
        ' {"npc_id":"npc_001","name":"Ala","profession":"Merchant","traits":["curious"],"goals":["trade"],"starting_location":"loc_001",'
        '  "obedience_level":0.5,"stubbornness":0.5,"risk_tolerance":0.5,"disposition_to_player":0,"refusal_style":"polite"}'
        '],'
        '"starting_location":"loc_001",'
        '"starting_hook":"You arrive.",'
        '"initial_quest":"Find the relic."'
        '}'
    )


def banned_world_json() -> str:
    return (
        '{'
        '"world_id":"world_bad2",'
        '"title":"Bad World",'
        '"world_bible":{"tech_level":"medieval","magic_rules":"low","tone":"adventurous"},'
        '"locations":['
        ' {"location_id":"loc_001","name":"Town","kind":"town","description":"A small town with smartphone.","connected_to":[],"tags":[]}'
        '],'
        '"npcs":['
        ' {"npc_id":"npc_001","name":"Ala","profession":"Merchant","traits":["curious"],"goals":["trade"],"starting_location":"loc_001",'
        '  "obedience_level":0.5,"stubbornness":0.5,"risk_tolerance":0.5,"disposition_to_player":0,"refusal_style":"polite"}'
        '],'
        '"starting_location":"loc_001",'
        '"starting_hook":"You arrive.",'
        '"initial_quest":"Find the relic."'
        '}'
    )


def dirty_world_json() -> str:
    return (
        '{'
        '"schema_version":"v1",'
        '"world_id":"world_dirty",'
        '"title":"Dirty World",'
        '"world_bible":{"tech_level":"medieval","magic_rules":"low","tone":"adventurous"},'
        '"locations":['
        ' {"location_id":"loc_001","name":"Town","kind":"town","description":"A small town.","connected_to":["loc_002"],"tags":[]},'
        ' {"location_id":"loc_002","name":"Forest","kind":"forest","description":"A dark forest.","connected_to":["loc_001"],"tags":[]}'
        '],'
        '"npcs":['
        ' {"npc_id":"npc_001","name":"Ala","profession":"Merchant","traits":["curious"],"goals":["trade"],"starting_location":"loc_001",'
        '  "obedience_level":7,"stubbornness":3,"risk_tolerance":12,"disposition_to_player":"Friendly","refusal_style":"polite"}'
        '],'
        '"starting_location":"loc_001",'
        '"starting_hook":"You arrive.",'
        '"initial_quest":"Find the relic."'
        '}'
    )


def live_like_world_json() -> str:
    return (
        '{'
        '"schema_version":"1.0",'
        '"world_id":"world_live_like",'
        '"title":"Riverside Market",'
        '"world_bible":{"tech_level":"medieval","magic_rules":"low","tone":"grounded","taboos":"necromancy, slavery"},'
        '"locations":['
        ' {"location_id":"loc_001","name":"Town","kind":"town","description":"A small town.","connected_to":["loc_002"],"tags":"market"},'
        ' {"location_id":"loc_002","name":"Bridge","kind":"bridge","description":"An old bridge.","connected_to":"loc_001","tags":[]}'
        '],'
        '"npcs":['
        ' {"npc_id":"npc_001","name":"Elin","profession":"Baker","traits":"kind","goals":"feed the town","starting_location":"loc_001",'
        '  "obedience_level":5,"stubbornness":3,"risk_tolerance":2,"disposition_to_player":"Friendly","refusal_style":"polite","extra":"x"},'
        ' {"npc_id":"npc_002","name":"Taro","profession":"Guard","traits":["stern"],"goals":["protect"],"starting_location":"loc_002",'
        '  "obedience_level":8,"stubbornness":6,"risk_tolerance":5,"disposition_to_player":"Skeptical","refusal_style":"firm"}'
        '],'
        '"starting_location":"loc_001",'
        '"starting_hook":"You arrive at the riverside market.",'
        '"initial_quest":"Deliver a message across the bridge."'
        '}'
    )


def test_generate_world_valid_first():
    cfg = load_config("configs/config.yaml")
    llm = MockLLMClient([valid_world_json()])
    world = generate_world_spec(cfg, llm, "A simple world")
    assert isinstance(world, WorldSpec)
    assert world.world_id == "world_001"
    assert llm.last_response_format is not None
    assert llm.last_response_format.get("type") == "json_schema"
    assert llm.last_response_format.get("strict") is True
    assert llm.last_response_format.get("json_schema", {}).get("name") == "WorldSpec"


def test_invalid_connected_to_triggers_rewrite():
    cfg = load_config("configs/config.yaml")
    llm = MockLLMClient([invalid_connected_world_json(), valid_world_json()])
    world = generate_world_spec(cfg, llm, "A simple world")
    assert world.world_id == "world_001"


def test_banned_keyword_triggers_rewrite():
    cfg = load_config("configs/config.yaml")
    llm = MockLLMClient([banned_world_json(), valid_world_json()])
    world = generate_world_spec(cfg, llm, "A simple world")
    assert world.world_id == "world_001"


def test_anachronism_detection_reports_matches():
    bible = WorldBibleRules(tech_level="medieval", magic_rules="low", tone="grounded")
    loc = LocationSpec(
        location_id="loc_001",
        name="Town",
        kind="town",
        description="A small town with a smartphone on display.",
        connected_to=[],
        tags=[],
    )
    npc = NPCProfile(
        npc_id="npc_001",
        name="Ala",
        profession="Merchant",
        traits=["curious"],
        goals=["trade"],
        starting_location="loc_001",
        obedience_level=0.5,
        stubbornness=0.5,
        risk_tolerance=0.5,
        disposition_to_player=0,
        refusal_style="polite",
    )
    world = WorldSpec(
        world_id="world_001",
        title="Test World",
        world_bible=bible,
        locations=[loc],
        npcs=[npc],
        starting_location="loc_001",
        starting_hook="You arrive.",
        initial_quest="Find the relic.",
    )
    matches = find_anachronisms(world)
    assert matches
    first = matches[0]
    assert first["keyword"] == "smartphone"
    assert first["path"] == "locations[0].description"
    assert "smartphone" in first["snippet"].lower()


def test_anachronism_detection_modern_allows_modern_terms():
    bible = WorldBibleRules(tech_level="modern", magic_rules="none", tone="grounded")
    loc = LocationSpec(
        location_id="loc_001",
        name="City",
        kind="city",
        description="A modern city with smartphone kiosks.",
        connected_to=[],
        tags=[],
    )
    npc = NPCProfile(
        npc_id="npc_001",
        name="Ala",
        profession="Engineer",
        traits=["curious"],
        goals=["build"],
        starting_location="loc_001",
        obedience_level=0.5,
        stubbornness=0.5,
        risk_tolerance=0.5,
        disposition_to_player=0,
        refusal_style="polite",
    )
    world = WorldSpec(
        world_id="world_002",
        title="Modern World",
        world_bible=bible,
        locations=[loc],
        npcs=[npc],
        starting_location="loc_001",
        starting_hook="You arrive.",
        initial_quest="Find the relic.",
    )
    assert not find_anachronisms(world)


def test_anachronism_detection_ignores_do_not_mention_only():
    bible = WorldBibleRules(
        tech_level="medieval",
        magic_rules="low",
        tone="grounded",
        do_not_mention=["smartphone"],
    )
    loc = LocationSpec(
        location_id="loc_001",
        name="Town",
        kind="town",
        description="A small town.",
        connected_to=[],
        tags=[],
    )
    npc = NPCProfile(
        npc_id="npc_001",
        name="Ala",
        profession="Merchant",
        traits=["curious"],
        goals=["trade"],
        starting_location="loc_001",
        obedience_level=0.5,
        stubbornness=0.5,
        risk_tolerance=0.5,
        disposition_to_player=0,
        refusal_style="polite",
    )
    world = WorldSpec(
        world_id="world_003",
        title="Test World",
        world_bible=bible,
        locations=[loc],
        npcs=[npc],
        starting_location="loc_001",
        starting_hook="You arrive.",
        initial_quest="Find the relic.",
    )
    assert not find_anachronisms(world)


def test_anachronism_rewrite_removes_keywords():
    cfg = load_config("configs/config.yaml")
    llm = MockLLMClient([banned_world_json(), valid_world_json()])
    world = generate_world_spec(cfg, llm, "A simple world")
    assert not find_anachronisms(world)


def test_sanitization_no_rewrite_needed():
    cfg = load_config("configs/config.yaml")
    llm = MockLLMClient([dirty_world_json()])
    world = generate_world_spec(cfg, llm, "A simple world")
    assert world.world_id == "world_dirty"
    npc = world.npcs[0]
    assert 0.0 <= npc.obedience_level <= 1.0
    assert -5 <= npc.disposition_to_player <= 5


def test_sanitization_live_like_payload():
    cfg = load_config("configs/config.yaml")
    llm = MockLLMClient([live_like_world_json()])
    world = generate_world_spec(cfg, llm, "A simple world")
    assert world.world_id == "world_live_like"
    for npc in world.npcs:
        assert 0.0 <= npc.obedience_level <= 1.0
        assert 0.0 <= npc.stubbornness <= 1.0
        assert 0.0 <= npc.risk_tolerance <= 1.0
        assert -5 <= npc.disposition_to_player <= 5


def test_initialize_game_state_full_coverage():
    cfg = load_config("configs/config.yaml")
    llm = MockLLMClient([valid_world_json()])
    world = generate_world_spec(cfg, llm, "A simple world")
    state = initialize_game_state(world, session_id="sess_x")
    assert set(state.npc_locations.keys()) == {"npc_001", "npc_002"}


def test_create_new_session_persists_files(tmp_path: Path):
    cfg = load_config("configs/config.yaml")
    llm = MockLLMClient([valid_world_json()])
    sessions_root = tmp_path / "sessions"
    worlds_root = tmp_path / "worlds"

    session_id, world, state = create_new_session(
        cfg, llm, "A simple world", sessions_root=sessions_root, worlds_root=worlds_root
    )

    world_path = worlds_root / session_id / "world.json"
    state_path = sessions_root / session_id / "state.json"
    assert world_path.exists()
    assert state_path.exists()

    loaded = load_state(session_id, sessions_root)
    assert loaded.session_id == session_id
    assert loaded.world.world_id == world.world_id
