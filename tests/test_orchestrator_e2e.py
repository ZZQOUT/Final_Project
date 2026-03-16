from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

from rpg_story.config import load_config
from rpg_story.engine.orchestrator import TurnPipeline
from rpg_story.llm.client import MockLLMClient
from rpg_story.models.world import WorldBibleRules, LocationSpec, NPCProfile, WorldSpec, GameState, QuestSpec
from rpg_story.persistence.store import load_state, read_turn_logs
from rpg_story.world.generator import initialize_game_state


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
        profession="Courier",
        traits=["brave"],
        goals=["deliver messages"],
        starting_location="shop",
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
        starting_location="shop",
        starting_hook="A rumor spreads.",
        initial_quest="Deliver a message.",
    )


def make_world_with_quest_grounding() -> WorldSpec:
    locations = [
        LocationSpec(
            location_id="town",
            name="晨光村",
            kind="town",
            description="宁静的村庄。",
            connected_to=["forest"],
        ),
        LocationSpec(
            location_id="forest",
            name="迷雾森林",
            kind="forest",
            description="潮湿而危险的森林。",
            connected_to=["town"],
        ),
    ]
    npcs = [
        NPCProfile(
            npc_id="npc_blacksmith",
            name="托姆",
            profession="铁匠",
            traits=["谨慎"],
            goals=["修复祖传铁砧"],
            starting_location="town",
            obedience_level=0.3,
            stubbornness=0.8,
            risk_tolerance=0.3,
            disposition_to_player=1,
            refusal_style="direct",
        ),
        NPCProfile(
            npc_id="npc_scholar",
            name="艾文",
            profession="学者",
            traits=["理性"],
            goals=["研究星铁矿"],
            starting_location="town",
            obedience_level=0.4,
            stubbornness=0.5,
            risk_tolerance=0.4,
            disposition_to_player=0,
            refusal_style="polite",
        ),
    ]
    side_quests = [
        QuestSpec(
            quest_id="side_anvil",
            title="修复祖传铁砧",
            category="side",
            description="帮托姆收集材料修复铁砧。",
            objective="带回月光草并交给托姆。",
            giver_npc_id="npc_blacksmith",
            suggested_location="forest",
            required_items={"月光草": 2},
            reward_items={"铁匠誓约": 1},
        ),
        QuestSpec(
            quest_id="side_research",
            title="研究样本",
            category="side",
            description="帮艾文研究矿石。",
            objective="收集星铁矿并交给艾文。",
            giver_npc_id="npc_scholar",
            suggested_location="forest",
            required_items={"星铁矿": 2},
            reward_items={"研究笔记": 1},
        ),
    ]
    return WorldSpec(
        world_id="world_grounded",
        title="Grounded World",
        world_bible=WorldBibleRules(
            tech_level="medieval",
            narrative_language="zh",
            magic_rules="low",
            tone="grounded",
        ),
        locations=locations,
        npcs=npcs,
        starting_location="town",
        starting_hook="村里需要修复古老铁砧。",
        initial_quest="先完成支线收集材料。",
        side_quests=side_quests,
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
    assert llm.last_response_format is not None
    assert llm.last_response_format.get("type") == "json_schema"
    assert llm.last_response_format.get("strict") is True
    assert llm.last_response_format.get("json_schema", {}).get("name") == "TurnOutput"

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
    assert updated_state.quests.get("q1") in {"accepted", "active"}
    # memory summary appended
    assert "Player asked NPC to move." in updated_state.recent_summaries

    # TurnOutput validated by pipeline
    if output.narration:
        assert output.narration == "OK"
    else:
        assert output.npc_dialogue


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


def test_orchestrator_repairs_npc_item_request_to_assigned_side_quest(tmp_path: Path):
    cfg = load_config("configs/config.yaml")
    sessions_root = tmp_path / "sessions"
    world = make_world_with_quest_grounding()
    state = initialize_game_state(world, session_id="sess_grounding")
    state.player_location = "town"
    state.npc_locations["npc_blacksmith"] = "town"
    state.location_resource_stock = {"forest": {"月光草": 2, "星铁矿": 2}}

    first_output = (
        "{"
        '"narration":"",'
        '"npc_dialogue":[{"npc_id":"npc_blacksmith","text":"托姆：要修铁砧，得带两块星铁矿过来。"}],'
        '"world_updates":{"player_location":"town","npc_moves":[],"flags_delta":{},"quest_updates":{}},'
        '"memory_summary":"",'
        '"safety":{"refuse":false,"reason":null}'
        "}"
    )
    repaired_output = (
        "{"
        '"narration":"",'
        '"npc_dialogue":[{"npc_id":"npc_blacksmith","text":"托姆：修复铁砧只需要月光草，两份就够。"}],'
        '"world_updates":{"player_location":"town","npc_moves":[],"flags_delta":{},"quest_updates":{}},'
        '"memory_summary":"",'
        '"safety":{"refuse":false,"reason":null}'
        "}"
    )
    llm = MockLLMClient([first_output, repaired_output])
    pipeline = TurnPipeline(cfg=cfg, llm_client=llm, sessions_root=sessions_root)

    _updated_state, output, _log = pipeline.run_turn(state, "你要啥材料", "npc_blacksmith")

    assert llm.calls == 2
    assert output.npc_dialogue
    text = output.npc_dialogue[0].text
    assert "月光草" in text
    assert "星铁矿" not in text
    assert llm.last_user_prompt is not None
    assert "allowed_world_items" in llm.last_user_prompt
    assert "npc_assigned_quests" in llm.last_user_prompt


def test_orchestrator_prompt_forbids_nonquest_npc_collection_request(tmp_path: Path):
    cfg = load_config("configs/config.yaml")
    sessions_root = tmp_path / "sessions"
    world = make_world()
    state = GameState(
        session_id="sess_nonquest_prompt",
        created_at=datetime.now(timezone.utc).isoformat(),
        world=world,
        player_location="shop",
        npc_locations={"npc_1": "shop"},
        location_resource_stock={"bridge": {"healing_herb": 2}},
    )

    output_json = (
        "{"
        '"narration":"",'
        '"npc_dialogue":[{"npc_id":"npc_1","text":"I can tell you what is happening around town."}],'
        '"world_updates":{"player_location":"shop","npc_moves":[],"flags_delta":{},"quest_updates":{}},'
        '"memory_summary":"",'
        '"safety":{"refuse":false,"reason":null}'
        "}"
    )
    llm = MockLLMClient([output_json])
    pipeline = TurnPipeline(cfg=cfg, llm_client=llm, sessions_root=sessions_root)

    _updated_state, _output, _log = pipeline.run_turn(state, "Any work for me?", "npc_1")

    assert llm.last_user_prompt is not None
    assert "If npc_assigned_quests is empty, this NPC must NOT ask the player to collect, bring, deliver, or submit any items." in llm.last_user_prompt
    assert "If the player asks about collection work anyway, say this NPC has no item-collection commission." in llm.last_user_prompt
    assert "During normal conversation, do not keep steering the topic back to quests or tasks." in llm.last_user_prompt


def test_orchestrator_repairs_nonquest_npc_collection_request(tmp_path: Path):
    cfg = load_config("configs/config.yaml")
    sessions_root = tmp_path / "sessions"
    world = make_world()
    state = GameState(
        session_id="sess_nonquest_repair",
        created_at=datetime.now(timezone.utc).isoformat(),
        world=world,
        player_location="shop",
        npc_locations={"npc_1": "shop"},
        location_resource_stock={"bridge": {"healing_herb": 2}},
    )

    first_output = (
        "{"
        '"narration":"",'
        '"npc_dialogue":[{"npc_id":"npc_1","text":"Mara: Bring me three dragon crystals and I will think about it."}],'
        '"world_updates":{"player_location":"shop","npc_moves":[],"flags_delta":{},"quest_updates":{}},'
        '"memory_summary":"",'
        '"safety":{"refuse":false,"reason":null}'
        "}"
    )
    repaired_output = (
        "{"
        '"narration":"",'
        '"npc_dialogue":[{"npc_id":"npc_1","text":"Mara: I do not have any item-collection errand for you. Ask me about the town instead."}],'
        '"world_updates":{"player_location":"shop","npc_moves":[],"flags_delta":{},"quest_updates":{}},'
        '"memory_summary":"",'
        '"safety":{"refuse":false,"reason":null}'
        "}"
    )
    llm = MockLLMClient([first_output, repaired_output])
    pipeline = TurnPipeline(cfg=cfg, llm_client=llm, sessions_root=sessions_root)

    _updated_state, output, _log = pipeline.run_turn(state, "What do you need?", "npc_1")

    assert llm.calls == 2
    assert output.npc_dialogue
    text = output.npc_dialogue[0].text
    assert "item-collection errand" in text
    assert "dragon crystals" not in text
    assert llm.last_user_prompt is not None
    assert "npc_assigned_quests=[]" in llm.last_user_prompt


def test_orchestrator_applies_personality_drift(tmp_path: Path):
    cfg = load_config("configs/config.yaml")
    sessions_root = tmp_path / "sessions"
    world = make_world()
    state = GameState(
        session_id="sess_personality",
        created_at=datetime.now(timezone.utc).isoformat(),
        world=world,
        player_location="shop",
        npc_locations={"npc_1": "shop"},
    )

    output_json = (
        "{"
        '"narration":"Mara seems more trusting now.",'
        '"npc_dialogue":[{"npc_id":"npc_1","text":"I trust you more than before."}],'
        '"world_updates":{'
        '  "player_location":"shop",'
        '  "npc_moves":[],'
        '  "flags_delta":{},'
        '  "quest_updates":{},'
        '  "npc_personality_updates":[{'
        '    "npc_id":"npc_1",'
        '    "obedience_level":0.9,'
        '    "stubbornness":0.1,'
        '    "risk_tolerance":0.8,'
        '    "disposition_to_player":4,'
        '    "refusal_style":"warm and cooperative",'
        '    "confidence":1.0,'
        '    "reason":"player consistently kept promises"'
        "  }]"
        "},"
        '"memory_summary":"Mara became more cooperative after repeated reliable dialogue.",'
        '"safety":{"refuse":false,"reason":null}'
        "}"
    )
    llm = MockLLMClient([output_json])
    pipeline = TurnPipeline(cfg=cfg, llm_client=llm, sessions_root=sessions_root)
    updated_state, output, _log = pipeline.run_turn(state, "I kept every promise.", "npc_1")

    npc = next(n for n in updated_state.world.npcs if n.npc_id == "npc_1")
    assert npc.obedience_level == 0.9
    assert npc.stubbornness == 0.1
    assert npc.risk_tolerance == 0.8
    assert npc.disposition_to_player == 4
    assert npc.refusal_style == "warm and cooperative"
    assert output.world_updates.npc_personality_updates


def test_orchestrator_personality_drift_uses_confidence_blending(tmp_path: Path):
    cfg = load_config("configs/config.yaml")
    sessions_root = tmp_path / "sessions"
    world = make_world()
    state = GameState(
        session_id="sess_personality_blend",
        created_at=datetime.now(timezone.utc).isoformat(),
        world=world,
        player_location="shop",
        npc_locations={"npc_1": "shop"},
    )

    output_json = (
        "{"
        '"narration":"Mara softens a little.",'
        '"npc_dialogue":[{"npc_id":"npc_1","text":"Maybe I can trust you a bit more."}],'
        '"world_updates":{'
        '  "player_location":"shop",'
        '  "npc_moves":[],'
        '  "flags_delta":{},'
        '  "quest_updates":{},'
        '  "npc_personality_updates":[{'
        '    "npc_id":"npc_1",'
        '    "obedience_level":1.0,'
        '    "disposition_to_player":5,'
        '    "confidence":0.5'
        "  }]"
        "},"
        '"memory_summary":"Mara shows moderate trust growth.",'
        '"safety":{"refuse":false,"reason":null}'
        "}"
    )
    llm = MockLLMClient([output_json])
    pipeline = TurnPipeline(cfg=cfg, llm_client=llm, sessions_root=sessions_root)
    updated_state, _output, _log = pipeline.run_turn(state, "Please trust me.", "npc_1")

    npc = next(n for n in updated_state.world.npcs if n.npc_id == "npc_1")
    # old obedience=0.8, target=1.0, conf=0.5 -> 0.9
    assert npc.obedience_level == 0.9
    # old disposition=1, target=5, conf=0.5 -> 3
    assert npc.disposition_to_player == 3
