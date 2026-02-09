from __future__ import annotations

from datetime import datetime, timezone

from rpg_story.engine.state import (
    apply_turn_output,
    deliver_items_to_npc,
    sync_quest_journal,
    resolve_main_trial,
    evaluate_main_trial_readiness,
)
from rpg_story.models.turn import TurnOutput
from rpg_story.models.world import (
    GameState,
    LocationSpec,
    NPCProfile,
    QuestSpec,
    WorldBibleRules,
    WorldSpec,
)
from rpg_story.world.generator import initialize_game_state


def _world_with_quest(required: int = 2) -> WorldSpec:
    locations = [
        LocationSpec(
            location_id="town",
            name="Town",
            kind="town",
            description="Town square.",
            connected_to=["forest"],
        ),
        LocationSpec(
            location_id="forest",
            name="Forest",
            kind="forest",
            description="Dense trees.",
            connected_to=["town"],
        ),
    ]
    npcs = [
        NPCProfile(
            npc_id="npc_herbalist",
            name="Mila",
            profession="Herbalist",
            traits=["careful"],
            goals=["heal villagers"],
            starting_location="town",
            obedience_level=0.7,
            stubbornness=0.3,
            risk_tolerance=0.4,
            disposition_to_player=1,
            refusal_style="calm",
        )
    ]
    return WorldSpec(
        world_id="world_quest",
        title="Quest World",
        world_bible=WorldBibleRules(tech_level="medieval", magic_rules="low", tone="grounded"),
        locations=locations,
        npcs=npcs,
        starting_location="town",
        starting_hook="A village lacks medicine.",
        initial_quest="Collect herbs for the healer.",
        main_quest=QuestSpec(
            quest_id="main_healing",
            title="Village Remedy",
            category="main",
            description="Gather herbs for treatment.",
            objective="Collect healing herbs.",
            giver_npc_id="npc_herbalist",
            suggested_location="forest",
            required_items={"healing_herb": required},
            reward_hint="Unlock next chapter.",
        ),
    )


def _empty_turn(inventory_delta: dict[str, int], quest_updates: list[dict] | None = None) -> TurnOutput:
    return TurnOutput.model_validate(
        {
            "narration": "",
            "npc_dialogue": [],
            "world_updates": {
                "npc_moves": [],
                "flags_delta": {},
                "quest_updates": {},
                "quest_progress_updates": quest_updates or [],
                "inventory_delta": inventory_delta,
            },
            "memory_summary": "",
            "safety": {"refuse": False, "reason": None},
        }
    )


def test_initialize_state_has_main_quest() -> None:
    world = _world_with_quest(required=2)
    state = initialize_game_state(world, session_id="sess_q")
    assert state.main_quest_id == "main_healing"
    assert "main_healing" in state.quest_journal
    assert state.quest_journal["main_healing"].status == "active"


def test_apply_turn_inventory_advances_and_completes_quest() -> None:
    world = _world_with_quest(required=2)
    state = GameState(
        session_id="sess_progress",
        created_at=datetime.now(timezone.utc).isoformat(),
        world=world,
        player_location="town",
        npc_locations={"npc_herbalist": "town"},
        quest_journal={
            "main_healing": {
                "quest_id": "main_healing",
                "title": "Village Remedy",
                "category": "main",
                "status": "active",
                "objective": "Collect healing herbs.",
                "required_items": {"healing_herb": 2},
                "collected_items": {"healing_herb": 0},
            }
        },
        main_quest_id="main_healing",
    )

    state = apply_turn_output(state, _empty_turn({"healing_herb": 1}), "npc_herbalist")
    assert state.inventory["healing_herb"] == 1
    assert state.quest_journal["main_healing"].status == "active"
    assert state.quest_journal["main_healing"].collected_items.get("healing_herb", 0) == 1

    state = apply_turn_output(state, _empty_turn({"healing_herb": 1}), "npc_herbalist")
    assert state.inventory["healing_herb"] == 2
    assert state.quest_journal["main_healing"].status == "active"
    assert state.quest_journal["main_healing"].collected_items.get("healing_herb", 0) == 2
    assert state.quests["main_healing"] == "active"


def test_legacy_quest_status_is_normalized() -> None:
    world = _world_with_quest(required=1)
    state = initialize_game_state(world, session_id="sess_legacy")
    out = TurnOutput.model_validate(
        {
            "narration": "",
            "npc_dialogue": [],
            "world_updates": {
                "npc_moves": [],
                "flags_delta": {},
                "quest_updates": {"side_help": "accepted"},
                "quest_progress_updates": [],
                "inventory_delta": {},
            },
            "memory_summary": "",
            "safety": {"refuse": False, "reason": None},
        }
    )
    updated = apply_turn_output(state, out, "npc_herbalist")
    assert updated.quest_journal["side_help"].status == "active"


def test_side_quest_completion_grants_reward_then_main_needs_delivery() -> None:
    locations = [
        LocationSpec(
            location_id="town",
            name="Town",
            kind="town",
            description="Town square.",
            connected_to=["forest"],
        ),
        LocationSpec(
            location_id="forest",
            name="Forest",
            kind="forest",
            description="Dense trees.",
            connected_to=["town"],
        ),
    ]
    npc = NPCProfile(
        npc_id="npc_herbalist",
        name="Mila",
        profession="Herbalist",
        traits=["careful"],
        goals=["heal villagers"],
        starting_location="forest",
        obedience_level=0.7,
        stubbornness=0.3,
        risk_tolerance=0.4,
        disposition_to_player=1,
        refusal_style="calm",
    )
    world = WorldSpec(
        world_id="world_chain",
        title="Quest Chain",
        world_bible=WorldBibleRules(tech_level="medieval", magic_rules="low", tone="grounded"),
        locations=locations,
        npcs=[npc],
        starting_location="town",
        starting_hook="A village lacks medicine.",
        initial_quest="Complete side tasks first.",
        main_quest=QuestSpec(
            quest_id="main_final",
            title="Final Battle",
            category="main",
            description="Need side reward token.",
            objective="Get token from side quest.",
            giver_npc_id="npc_herbalist",
            suggested_location="forest",
            required_items={"healer_token": 1},
            reward_items={},
        ),
        side_quests=[
            QuestSpec(
                quest_id="side_herb",
                title="Find Moon Herb",
                category="side",
                description="Collect herbs with Mila in forest.",
                objective="Collect moon_herb x2 at forest with Mila.",
                giver_npc_id="npc_herbalist",
                suggested_location="forest",
                required_items={"moon_herb": 2},
                reward_items={"healer_token": 1},
                reward_hint="Reward token",
            )
        ],
    )
    state = GameState(
        session_id="sess_chain",
        created_at=datetime.now(timezone.utc).isoformat(),
        world=world,
        player_location="forest",
        npc_locations={"npc_herbalist": "forest"},
        inventory={"moon_herb": 2},
        quest_journal={
            "main_final": {
                "quest_id": "main_final",
                "title": "Final Battle",
                "category": "main",
                "status": "active",
                "objective": "Get token from side quest.",
                "required_items": {"healer_token": 1},
                "collected_items": {"healer_token": 0},
                "reward_items": {},
            },
            "side_herb": {
                "quest_id": "side_herb",
                "title": "Find Moon Herb",
                "category": "side",
                "status": "active",
                "objective": "Collect moon_herb x2",
                "required_items": {"moon_herb": 2},
                "collected_items": {"moon_herb": 0},
                "reward_items": {"healer_token": 1},
            },
        },
        quests={"main_final": "active", "side_herb": "active"},
        main_quest_id="main_final",
    )
    updated, notices, rewards, delivered = deliver_items_to_npc(
        state,
        "npc_herbalist",
        "forest",
        {"moon_herb": 2},
    )
    assert updated.quest_journal["side_herb"].status == "completed"
    assert delivered == {"moon_herb": 2}
    assert rewards == {"healer_token": 1}
    assert "healer_token" in updated.inventory
    assert updated.quest_journal["main_final"].status == "active"
    assert notices

    # Deliver side reward token to the same NPC should NOT complete main quest directly.
    final_state, final_notices, final_rewards, final_delivered = deliver_items_to_npc(
        updated,
        "npc_herbalist",
        "forest",
        {"healer_token": 1},
    )
    assert final_state.quest_journal["main_final"].status == "active"
    assert final_delivered == {}
    assert not final_rewards
    assert final_notices == []

    ready, progress = evaluate_main_trial_readiness(final_state)
    assert ready is True
    assert progress["healer_token"]["have"] == 1
    resolved = resolve_main_trial(final_state, passed=True)
    assert resolved.quest_journal["main_final"].status == "completed"


def test_delivery_matches_item_alias_between_languages() -> None:
    world = _world_with_quest(required=1)
    state = GameState(
        session_id="sess_alias",
        created_at=datetime.now(timezone.utc).isoformat(),
        world=world,
        player_location="town",
        npc_locations={"npc_herbalist": "town"},
        inventory={"ration": 2},
        quest_journal={
            "side_ration": {
                "quest_id": "side_ration",
                "title": "Ration Test",
                "category": "side",
                "status": "active",
                "giver_npc_id": "npc_herbalist",
                "required_items": {"口粮": 2},
                "collected_items": {"口粮": 0},
                "reward_items": {},
            }
        },
        quests={"side_ration": "active"},
    )
    updated, notices, rewards, delivered = deliver_items_to_npc(
        state,
        "npc_herbalist",
        "town",
        {"ration": 2},
    )
    assert delivered == {"ration": 2}
    assert updated.quest_journal["side_ration"].status == "completed"


def test_main_guidance_points_to_realtime_finale_npc_location() -> None:
    world = _world_with_quest(required=1)
    state = GameState(
        session_id="sess_finale_target",
        created_at=datetime.now(timezone.utc).isoformat(),
        world=world,
        player_location="town",
        npc_locations={"npc_herbalist": "town"},
        inventory={"healing_herb": 1},
        quest_journal={
            "main_healing": {
                "quest_id": "main_healing",
                "title": "Village Remedy",
                "category": "main",
                "status": "active",
                "objective": "Collect healing herbs.",
                "giver_npc_id": "npc_herbalist",
                "required_items": {"healing_herb": 1},
                "collected_items": {"healing_herb": 0},
            }
        },
        quests={"main_healing": "active"},
        main_quest_id="main_healing",
    )
    synced = sync_quest_journal(state)
    guidance = synced.quest_journal["main_healing"].guidance
    assert "Main items ready." in guidance
    assert "Town" in guidance
    assert "Mila" in guidance


def test_main_trial_readiness_true_when_no_required_items() -> None:
    world = _world_with_quest(required=1)
    state = GameState(
        session_id="sess_no_required",
        created_at=datetime.now(timezone.utc).isoformat(),
        world=world,
        player_location="town",
        npc_locations={"npc_herbalist": "town"},
        quest_journal={
            "main_healing": {
                "quest_id": "main_healing",
                "title": "Village Remedy",
                "category": "main",
                "status": "active",
                "objective": "Collect healing herbs.",
                "giver_npc_id": "npc_herbalist",
                "required_items": {},
                "collected_items": {},
            }
        },
        quests={"main_healing": "active"},
        main_quest_id="main_healing",
    )
    ready, progress = evaluate_main_trial_readiness(state)
    assert ready is True
    assert progress == {}


def test_main_progress_tracks_inventory_but_needs_trial_to_complete() -> None:
    world = _world_with_quest(required=2)
    state = GameState(
        session_id="sess_main_progress",
        created_at=datetime.now(timezone.utc).isoformat(),
        world=world,
        player_location="forest",
        npc_locations={"npc_herbalist": "forest"},
        inventory={"healing_herb": 2},
        quest_journal={
            "main_healing": {
                "quest_id": "main_healing",
                "title": "Village Remedy",
                "category": "main",
                "status": "active",
                "required_items": {"healing_herb": 2},
                "collected_items": {"healing_herb": 0},
            }
        },
        quests={"main_healing": "active"},
        main_quest_id="main_healing",
    )
    synced = sync_quest_journal(state)
    assert synced.quest_journal["main_healing"].collected_items["healing_herb"] == 2
    assert synced.quest_journal["main_healing"].status == "active"


def test_existing_world_quest_definition_is_not_overwritten_by_dialogue_update() -> None:
    world = _world_with_quest(required=2)
    state = initialize_game_state(world, session_id="sess_lock")
    original = state.quest_journal["main_healing"]

    out = _empty_turn(
        {},
        quest_updates=[
            {
                "quest_id": "main_healing",
                "title": "寻找秘银",
                "objective": "去矿洞寻找秘银",
                "required_items": {"秘银": 3},
                "collected_items_delta": {"秘银": 2},
                "guidance": "测试引导",
                "status": "active",
            }
        ],
    )
    updated = apply_turn_output(state, out, "npc_herbalist")
    locked = updated.quest_journal["main_healing"]

    assert locked.title == original.title
    assert locked.objective == original.objective
    assert locked.required_items == original.required_items
    assert locked.collected_items.get("healing_herb", 0) == 0
    assert "秘银" not in locked.collected_items


def test_unknown_ad_hoc_quest_update_is_ignored_for_stability() -> None:
    world = _world_with_quest(required=1)
    state = initialize_game_state(world, session_id="sess_ignore_unknown")
    out = _empty_turn(
        {},
        quest_updates=[
            {
                "quest_id": "side_random_mithril",
                "title": "寻找秘银",
                "category": "side",
                "status": "active",
                "required_items": {"秘银": 2},
            }
        ],
    )
    updated = apply_turn_output(state, out, "npc_herbalist")
    assert "side_random_mithril" not in updated.quest_journal


def test_sync_repairs_mismatched_side_quest_text_against_required_items() -> None:
    world = _world_with_quest(required=1)
    world.world_bible.narrative_language = "zh"
    state = GameState(
        session_id="sess_text_repair",
        created_at=datetime.now(timezone.utc).isoformat(),
        world=world,
        player_location="town",
        npc_locations={"npc_herbalist": "town"},
        quest_journal={
            "side_mismatch": {
                "quest_id": "side_mismatch",
                "title": "寻找秘银",
                "category": "side",
                "status": "active",
                "objective": "前往森林寻找秘银。",
                "required_items": {"月光草": 3},
                "collected_items": {"月光草": 0},
                "reward_items": {},
            }
        },
        quests={"side_mismatch": "active"},
    )
    synced = sync_quest_journal(state)
    quest = synced.quest_journal["side_mismatch"]
    assert "月光草" in quest.title
    assert "月光草" in quest.objective


def test_chat_turn_cannot_auto_submit_item_delivery_progress() -> None:
    world = _world_with_quest(required=1)
    state = GameState(
        session_id="sess_no_auto_submit",
        created_at=datetime.now(timezone.utc).isoformat(),
        world=world,
        player_location="town",
        npc_locations={"npc_herbalist": "town"},
        inventory={"moon_herb": 4},
        quest_journal={
            "side_herb": {
                "quest_id": "side_herb",
                "title": "Find Herbs",
                "category": "side",
                "status": "active",
                "objective": "Collect moon herb and deliver to Mila.",
                "giver_npc_id": "npc_herbalist",
                "required_items": {"moon_herb": 4},
                "collected_items": {"moon_herb": 0},
                "reward_items": {"healer_token": 1},
            }
        },
        quests={"side_herb": "active"},
    )
    out = _empty_turn(
        {},
        quest_updates=[
            {
                "quest_id": "side_herb",
                "status": "completed",
                "collected_items_delta": {"moon_herb": 4},
            }
        ],
    )
    updated = apply_turn_output(state, out, "npc_herbalist")
    quest = updated.quest_journal["side_herb"]
    assert quest.status != "completed"
    assert quest.collected_items.get("moon_herb", 0) == 0
    assert updated.inventory.get("moon_herb", 0) == 4


def test_delivery_allows_handover_at_npc_location_even_if_suggested_location_differs() -> None:
    world = _world_with_quest(required=1)
    side = QuestSpec(
        quest_id="side_blacksmith",
        title="打造龙鳞匕首",
        category="side",
        description="交给铁匠材料",
        objective="收集并交付",
        giver_npc_id="npc_herbalist",
        suggested_location="forest",
        required_items={"星铁矿": 3},
        reward_items={"龙鳞匕首": 1},
    )
    world.side_quests = [side]
    state = initialize_game_state(world, session_id="sess_deliver_loc")
    state.player_location = "town"
    state.npc_locations["npc_herbalist"] = "town"
    state.inventory = {"星铁矿": 3}
    side_progress = state.quest_journal["side_blacksmith"].model_copy(deep=True)
    side_progress.status = "active"
    state.quest_journal["side_blacksmith"] = side_progress
    state.quests["side_blacksmith"] = "active"

    updated, notices, rewards, delivered = deliver_items_to_npc(
        state,
        "npc_herbalist",
        "town",
        {"星铁矿": 3},
    )
    assert delivered == {"星铁矿": 3}
    assert updated.quest_journal["side_blacksmith"].status == "completed"
    assert rewards == {"龙鳞匕首": 1}
    assert notices
