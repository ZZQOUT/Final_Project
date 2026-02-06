from __future__ import annotations

from datetime import datetime, timezone

from rpg_story.engine.state import apply_turn_output, deliver_items_to_npc
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
    assert state.quest_journal["main_healing"].collected_items.get("healing_herb", 0) == 0

    state = apply_turn_output(state, _empty_turn({"healing_herb": 1}), "npc_herbalist")
    assert state.inventory["healing_herb"] == 2
    assert state.quest_journal["main_healing"].status == "active"
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

    # Deliver side reward token to the same NPC for main quest completion.
    final_state, final_notices, final_rewards, final_delivered = deliver_items_to_npc(
        updated,
        "npc_herbalist",
        "forest",
        {"healer_token": 1},
    )
    assert final_state.quest_journal["main_final"].status == "completed"
    assert final_delivered == {"healer_token": 1}
    assert not final_rewards
    assert final_notices
