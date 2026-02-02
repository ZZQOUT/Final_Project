"""Validation helpers for movement and location legality."""
from __future__ import annotations

from typing import Tuple

from rpg_story.engine.state import GameState


def is_location_valid(state: GameState, location_id: str) -> bool:
    return any(loc.location_id == location_id for loc in state.world.locations)


def is_reachable(state: GameState, from_loc: str, to_loc: str) -> bool:
    if from_loc == to_loc:
        return True
    adjacency = {loc.location_id: set(loc.connected_to) for loc in state.world.locations}
    visited = set()
    queue = [from_loc]
    while queue:
        current = queue.pop(0)
        if current == to_loc:
            return True
        if current in visited:
            continue
        visited.add(current)
        for nxt in adjacency.get(current, set()):
            if nxt not in visited:
                queue.append(nxt)
    return False


def validate_npc_move(
    state: GameState,
    npc_id: str,
    from_location: str,
    to_location: str,
) -> Tuple[bool, str]:
    if npc_id not in state.npc_locations:
        return False, "npc_id_not_found"
    if state.npc_locations.get(npc_id) != from_location:
        return False, "from_location_mismatch"
    if not is_location_valid(state, to_location):
        return False, "to_location_invalid"
    if not state.world.bible.allow_special_travel and not is_reachable(state, from_location, to_location):
        return False, "not_reachable"
    return True, "ok"
