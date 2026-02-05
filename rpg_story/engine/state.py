"""Minimal state update helpers (Milestone 3)."""
from __future__ import annotations

from typing import Dict, Any
from copy import deepcopy

from rpg_story.models.world import GameState
from rpg_story.models.turn import TurnOutput


def apply_turn_output(state: GameState, output: TurnOutput, npc_id: str) -> GameState:
    """Apply a TurnOutput to GameState and return a new validated state."""
    data: Dict[str, Any] = state.model_dump()

    # Update last_turn_id
    data["last_turn_id"] = int(data.get("last_turn_id", 0)) + 1

    # Summary handling (append to recent_summaries)
    summary = output.memory_summary
    if summary:
        summaries = data.get("recent_summaries", [])
        if not isinstance(summaries, list):
            summaries = []
        summaries.append(summary)
        data["recent_summaries"] = summaries

    # Flags merge
    if output.world_updates.flags_delta:
        flags = data.get("flags", {})
        flags.update(output.world_updates.flags_delta)
        data["flags"] = flags

    # Quest updates
    if output.world_updates.quest_updates:
        quests = data.get("quests", {})
        quests.update(output.world_updates.quest_updates)
        data["quests"] = quests

    # Player movement
    location_ids = {loc["location_id"] for loc in data["world"]["locations"]}
    if output.world_updates.player_location:
        if output.world_updates.player_location not in location_ids:
            raise ValueError(f"Invalid player_location: {output.world_updates.player_location}")
        data["player_location"] = output.world_updates.player_location

    return GameState.model_validate(data)
