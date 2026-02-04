"""World consistency checks and rewrite helpers (Milestone 4)."""
from __future__ import annotations

from typing import List, Dict, Any, Iterator, Tuple
import re

from rpg_story.models.world import WorldSpec


class WorldConsistencyError(Exception):
    pass


DEFAULT_MEDIEVAL_ANACHRONISMS = [
    "smartphone",
    "internet",
    "WiFi",
    "email",
    "app",
    "credit card",
    "gun",
]


def _is_word_keyword(keyword: str) -> bool:
    lowered = keyword.lower()
    return lowered == keyword and re.fullmatch(r"[a-z0-9]+", lowered) is not None


def _match_span(text: str, keyword: str) -> Tuple[int, int] | None:
    if not keyword:
        return None
    if _is_word_keyword(keyword):
        pattern = re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            return match.start(), match.end()
        return None
    lowered = text.lower()
    key_lower = keyword.lower()
    idx = lowered.find(key_lower)
    if idx == -1:
        return None
    return idx, idx + len(key_lower)


def _snippet(text: str, start: int, end: int, context: int = 20) -> str:
    left = max(0, start - context)
    right = min(len(text), end + context)
    snippet = text[left:right].replace("\n", " ").strip()
    return snippet


def _iter_anachronism_fields(world: WorldSpec) -> Iterator[Tuple[str, str]]:
    yield "starting_hook", world.starting_hook
    yield "initial_quest", world.initial_quest

    for loc_idx, loc in enumerate(world.locations):
        yield f"locations[{loc_idx}].name", loc.name
        yield f"locations[{loc_idx}].description", loc.description

    for npc_idx, npc in enumerate(world.npcs):
        yield f"npcs[{npc_idx}].name", npc.name
        yield f"npcs[{npc_idx}].profession", npc.profession
        for trait_idx, trait in enumerate(npc.traits):
            yield f"npcs[{npc_idx}].traits[{trait_idx}]", trait
        for goal_idx, goal in enumerate(npc.goals):
            yield f"npcs[{npc_idx}].goals[{goal_idx}]", goal


def validate_world(world: WorldSpec, *, strict_bidirectional: bool = False) -> None:
    """Validate world references and optional bidirectional edges."""
    # WorldSpec model_validate already enforces core constraints
    if strict_bidirectional:
        world.validate_bidirectional_edges(strict=True)


def find_anachronisms(world: WorldSpec) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    tech_level = getattr(world.world_bible, "tech_level", "medieval") or "medieval"
    forbidden = [kw for kw in world.world_bible.do_not_mention if kw]
    if not forbidden and tech_level == "medieval":
        forbidden = list(DEFAULT_MEDIEVAL_ANACHRONISMS)

    if not forbidden:
        return matches

    fields = list(_iter_anachronism_fields(world))
    for keyword in forbidden:
        for path, text in fields:
            if not text:
                continue
            span = _match_span(text, keyword)
            if span is None:
                continue
            start, end = span
            matches.append(
                {
                    "keyword": keyword,
                    "path": path,
                    "snippet": _snippet(text, start, end),
                }
            )
    return matches
