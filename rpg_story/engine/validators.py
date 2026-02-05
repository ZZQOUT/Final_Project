"""Movement validators for NPC moves."""
from __future__ import annotations

from collections import deque
from typing import Dict, Set, Tuple, List

from rpg_story.models.world import WorldSpec, GameState
from rpg_story.models.turn import NPCMove


def build_graph(world: WorldSpec) -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = {}
    loc_ids = {loc.location_id for loc in world.locations}
    for loc in world.locations:
        neighbors = {target for target in loc.connected_to if target in loc_ids}
        graph[loc.location_id] = neighbors
    return graph


def is_reachable(graph: Dict[str, Set[str]], start: str, goal: str) -> bool:
    if start == goal:
        return start in graph
    if start not in graph or goal not in graph:
        return False
    visited = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in graph.get(node, set()):
            if neighbor == goal:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False


def validate_npc_move(move: NPCMove, state: GameState, world: WorldSpec) -> Tuple[bool, str]:
    graph = build_graph(world)
    return _validate_npc_move_with_graph(move, state, world, graph)


def _validate_npc_move_with_graph(
    move: NPCMove,
    state: GameState,
    world: WorldSpec,
    graph: Dict[str, Set[str]],
) -> Tuple[bool, str]:
    npc_ids = {npc.npc_id for npc in world.npcs}
    if move.npc_id not in npc_ids:
        return False, "npc_id not found"
    if move.npc_id not in state.npc_locations:
        return False, "npc_id missing in npc_locations"

    current_loc = state.npc_locations[move.npc_id]
    if move.from_location != current_loc:
        return False, f"from_location mismatch (expected {current_loc})"

    loc_ids = world.location_ids()
    if move.to_location not in loc_ids:
        return False, "to_location unknown"

    if not is_reachable(graph, move.from_location, move.to_location):
        return False, "to_location unreachable"

    return True, "ok"


def apply_validated_moves(
    moves: List[NPCMove],
    state: GameState,
    world: WorldSpec,
) -> Tuple[GameState, List[dict]]:
    working_state = state.model_copy(deep=True)
    events: List[dict] = []
    graph = build_graph(world)

    for move in moves:
        ok, reason = _validate_npc_move_with_graph(move, working_state, world, graph)
        if ok:
            working_state.npc_locations[move.npc_id] = move.to_location
            continue
        events.append(
            {
                "type": "move_rejected",
                "npc_id": move.npc_id,
                "from_location": move.from_location,
                "to_location": move.to_location,
                "reason": reason,
            }
        )

    return GameState.model_validate(working_state.model_dump()), events
