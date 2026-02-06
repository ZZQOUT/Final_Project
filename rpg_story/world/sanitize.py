"""Sanitize/normalize world payloads from LLMs before strict validation."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import re

WORLD_FIELDS = {
    "world_id",
    "title",
    "world_bible",
    "locations",
    "npcs",
    "starting_location",
    "starting_hook",
    "initial_quest",
    "main_quest",
    "side_quests",
    "map_layout",
}

WORLD_BIBLE_FIELDS = {
    "tech_level",
    "narrative_language",
    "magic_rules",
    "tone",
    "anachronism_policy",
    "taboos",
    "do_not_mention",
    "anachronism_blocklist",
}

LOCATION_FIELDS = {
    "location_id",
    "name",
    "kind",
    "description",
    "connected_to",
    "tags",
}

NPC_FIELDS = {
    "npc_id",
    "name",
    "profession",
    "traits",
    "goals",
    "starting_location",
    "obedience_level",
    "stubbornness",
    "risk_tolerance",
    "disposition_to_player",
    "refusal_style",
}

QUEST_FIELDS = {
    "quest_id",
    "title",
    "category",
    "description",
    "objective",
    "giver_npc_id",
    "suggested_location",
    "required_items",
    "reward_items",
    "reward_hint",
}

MAP_LAYOUT_FIELDS = {
    "location_id",
    "x",
    "y",
}

LEVEL_MAP = {
    "low": 0.2,
    "medium": 0.5,
    "med": 0.5,
    "average": 0.5,
    "high": 0.8,
    "very high": 0.9,
}

DISPOSITION_KEYWORDS = {
    "very hostile": -5,
    "hostile": -4,
    "hate": -5,
    "angry": -3,
    "suspicious": -2,
    "wary": -2,
    "skeptical": -1,
    "cautious": -1,
    "neutral": 0,
    "friendly": 2,
    "warm": 2,
    "kind": 2,
    "helpful": 3,
    "trusting": 3,
    "ally": 4,
}

SCRUB_TEXT_FIELDS = {
    "title",
    "starting_hook",
    "initial_quest",
}

SCRUB_WORLD_BIBLE_TEXT_FIELDS = {"tech_level", "magic_rules", "tone"}
SCRUB_WORLD_BIBLE_LIST_FIELDS = {"taboos", "do_not_mention", "anachronism_blocklist"}
SCRUB_LOCATION_TEXT_FIELDS = {"name", "kind", "description"}
SCRUB_LOCATION_LIST_FIELDS = {"tags"}
SCRUB_NPC_TEXT_FIELDS = {"name", "profession", "refusal_style"}
SCRUB_NPC_LIST_FIELDS = {"traits", "goals"}
SCRUB_QUEST_TEXT_FIELDS = {"title", "description", "objective", "reward_hint"}


def summarize_changes(changes: List[str], limit: int = 12) -> str:
    if not changes:
        return "none"
    if len(changes) > limit:
        return "; ".join(changes[:limit]) + f"; +{len(changes) - limit} more"
    return "; ".join(changes)


def sanitize_world_payload(data: Any) -> Tuple[Any, List[str]]:
    changes: List[str] = []
    if not isinstance(data, dict):
        changes.append("payload_not_dict")
        return data, changes

    cleaned: Dict[str, Any] = {}
    for key, value in data.items():
        if key in WORLD_FIELDS:
            cleaned[key] = value
        else:
            changes.append(f"drop_top:{key}")

    world_bible = cleaned.get("world_bible")
    if isinstance(world_bible, dict):
        cleaned_bible: Dict[str, Any] = {}
        for key, value in world_bible.items():
            if key in WORLD_BIBLE_FIELDS:
                if key in {"taboos", "do_not_mention", "anachronism_blocklist"}:
                    cleaned_bible[key] = _coerce_str_list(value)
                elif key == "narrative_language":
                    cleaned_bible[key] = _normalize_narrative_language(value, changes, f"world_bible.{key}")
                else:
                    cleaned_bible[key] = value
            else:
                changes.append(f"drop_world_bible:{key}")
        cleaned["world_bible"] = cleaned_bible

    locations = cleaned.get("locations")
    if isinstance(locations, list):
        cleaned_locations: List[Any] = []
        for idx, loc in enumerate(locations):
            if not isinstance(loc, dict):
                cleaned_locations.append(loc)
                continue
            loc_clean: Dict[str, Any] = {}
            for key, value in loc.items():
                if key in LOCATION_FIELDS:
                    if key in {"connected_to", "tags"}:
                        loc_clean[key] = _coerce_str_list(value)
                    else:
                        loc_clean[key] = value
                else:
                    changes.append(f"drop_location[{idx}]:{key}")
            cleaned_locations.append(loc_clean)
        cleaned["locations"] = cleaned_locations

    npcs = cleaned.get("npcs")
    if isinstance(npcs, list):
        cleaned_npcs: List[Any] = []
        for idx, npc in enumerate(npcs):
            if not isinstance(npc, dict):
                cleaned_npcs.append(npc)
                continue
            npc_clean: Dict[str, Any] = {}
            for key, value in npc.items():
                if key in NPC_FIELDS:
                    if key in {"traits", "goals"}:
                        npc_clean[key] = _coerce_str_list(value)
                    elif key in {"obedience_level", "stubbornness", "risk_tolerance"}:
                        npc_clean[key] = _normalize_unit(value, changes, f"npcs[{idx}].{key}")
                    elif key == "disposition_to_player":
                        npc_clean[key] = _normalize_disposition(value, changes, f"npcs[{idx}].{key}")
                    else:
                        npc_clean[key] = value
                else:
                    changes.append(f"drop_npc[{idx}]:{key}")
            cleaned_npcs.append(npc_clean)
        cleaned["npcs"] = cleaned_npcs

    main_quest = cleaned.get("main_quest")
    if isinstance(main_quest, dict):
        cleaned["main_quest"] = _sanitize_quest(main_quest, changes, "main_quest")

    side_quests = cleaned.get("side_quests")
    if isinstance(side_quests, list):
        cleaned_side_quests: List[Any] = []
        for idx, quest in enumerate(side_quests):
            if not isinstance(quest, dict):
                cleaned_side_quests.append(quest)
                continue
            cleaned_side_quests.append(_sanitize_quest(quest, changes, f"side_quests[{idx}]"))
        cleaned["side_quests"] = cleaned_side_quests

    map_layout = cleaned.get("map_layout")
    if isinstance(map_layout, list):
        cleaned_layout: List[Any] = []
        for idx, node in enumerate(map_layout):
            if not isinstance(node, dict):
                cleaned_layout.append(node)
                continue
            node_clean: Dict[str, Any] = {}
            for key, value in node.items():
                if key in MAP_LAYOUT_FIELDS:
                    if key in {"x", "y"}:
                        node_clean[key] = _to_float(value)
                    else:
                        node_clean[key] = value
                else:
                    changes.append(f"drop_map_layout[{idx}]:{key}")
            cleaned_layout.append(node_clean)
        cleaned["map_layout"] = cleaned_layout

    return cleaned, changes


def scrub_banned_terms(data: Any, banned: List[str]) -> Tuple[Any, List[str]]:
    changes: List[str] = []
    if not isinstance(data, dict):
        changes.append("scrub_skip_non_dict")
        return data, changes

    patterns = _build_banned_patterns(banned)
    cleaned = dict(data)

    for field in SCRUB_TEXT_FIELDS:
        if isinstance(cleaned.get(field), str):
            cleaned[field] = _scrub_text(cleaned[field], field, patterns, changes)

    world_bible = cleaned.get("world_bible")
    if isinstance(world_bible, dict):
        bible_clean = dict(world_bible)
        for field in SCRUB_WORLD_BIBLE_TEXT_FIELDS:
            if isinstance(bible_clean.get(field), str):
                bible_clean[field] = _scrub_text(bible_clean[field], f"world_bible.{field}", patterns, changes)
        for field in SCRUB_WORLD_BIBLE_LIST_FIELDS:
            if isinstance(bible_clean.get(field), list):
                bible_clean[field] = _scrub_list(bible_clean[field], f"world_bible.{field}", patterns, changes)
        cleaned["world_bible"] = bible_clean

    locations = cleaned.get("locations")
    if isinstance(locations, list):
        new_locations: List[Any] = []
        for idx, loc in enumerate(locations):
            if not isinstance(loc, dict):
                new_locations.append(loc)
                continue
            loc_clean = dict(loc)
            for field in SCRUB_LOCATION_TEXT_FIELDS:
                if isinstance(loc_clean.get(field), str):
                    loc_clean[field] = _scrub_text(
                        loc_clean[field], f"locations[{idx}].{field}", patterns, changes
                    )
            for field in SCRUB_LOCATION_LIST_FIELDS:
                if isinstance(loc_clean.get(field), list):
                    loc_clean[field] = _scrub_list(
                        loc_clean[field], f"locations[{idx}].{field}", patterns, changes
                    )
            new_locations.append(loc_clean)
        cleaned["locations"] = new_locations

    npcs = cleaned.get("npcs")
    if isinstance(npcs, list):
        new_npcs: List[Any] = []
        for idx, npc in enumerate(npcs):
            if not isinstance(npc, dict):
                new_npcs.append(npc)
                continue
            npc_clean = dict(npc)
            for field in SCRUB_NPC_TEXT_FIELDS:
                if isinstance(npc_clean.get(field), str):
                    npc_clean[field] = _scrub_text(
                        npc_clean[field], f"npcs[{idx}].{field}", patterns, changes
                    )
            for field in SCRUB_NPC_LIST_FIELDS:
                if isinstance(npc_clean.get(field), list):
                    npc_clean[field] = _scrub_list(
                        npc_clean[field], f"npcs[{idx}].{field}", patterns, changes
                    )
            new_npcs.append(npc_clean)
        cleaned["npcs"] = new_npcs

    main_quest = cleaned.get("main_quest")
    if isinstance(main_quest, dict):
        q_clean = dict(main_quest)
        for field in SCRUB_QUEST_TEXT_FIELDS:
            if isinstance(q_clean.get(field), str):
                q_clean[field] = _scrub_text(q_clean[field], f"main_quest.{field}", patterns, changes)
        cleaned["main_quest"] = q_clean

    side_quests = cleaned.get("side_quests")
    if isinstance(side_quests, list):
        new_quests: List[Any] = []
        for idx, quest in enumerate(side_quests):
            if not isinstance(quest, dict):
                new_quests.append(quest)
                continue
            q_clean = dict(quest)
            for field in SCRUB_QUEST_TEXT_FIELDS:
                if isinstance(q_clean.get(field), str):
                    q_clean[field] = _scrub_text(
                        q_clean[field], f"side_quests[{idx}].{field}", patterns, changes
                    )
            new_quests.append(q_clean)
        cleaned["side_quests"] = new_quests

    return cleaned, changes


def _coerce_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, tuple):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if "," in text:
            return [part.strip() for part in text.split(",") if part.strip()]
        return [text]
    return []


def _parse_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return None
        if text.endswith("%"):
            try:
                return float(text[:-1].strip()) / 100.0
            except ValueError:
                return None
        text = text.replace(",", "")
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
    return None


def _to_float(value: Any) -> Any:
    parsed = _parse_number(value)
    if parsed is None:
        return value
    return round(parsed, 3)


def _build_banned_patterns(banned: List[str]) -> List[Tuple[str, re.Pattern]]:
    patterns: List[Tuple[str, re.Pattern]] = []
    for term in banned:
        lowered = term.lower()
        if " " in lowered:
            parts = [re.escape(part) for part in lowered.split()]
            body = r"[-\s]+".join(parts)
        else:
            body = re.escape(lowered)
        patterns.append((term, re.compile(rf"\b{body}\b")))
    return patterns


def _sanitize_quest(quest: Dict[str, Any], changes: List[str], prefix: str) -> Dict[str, Any]:
    quest_clean: Dict[str, Any] = {}
    for key, value in quest.items():
        if key in QUEST_FIELDS:
            if key in {"required_items", "reward_items"}:
                quest_clean[key] = _coerce_item_map(value)
            else:
                quest_clean[key] = value
        else:
            changes.append(f"drop_{prefix}:{key}")
    return quest_clean


def _coerce_item_map(value: Any) -> Dict[str, int]:
    if value is None:
        return {}
    if isinstance(value, dict):
        result: Dict[str, int] = {}
        for key, raw in value.items():
            parsed = _parse_number(raw)
            if parsed is None:
                continue
            amount = int(round(parsed))
            if amount > 0:
                result[str(key)] = amount
        return result
    if isinstance(value, list):
        result: Dict[str, int] = {}
        for item in value:
            if not isinstance(item, dict):
                continue
            key = item.get("item") or item.get("name")
            if not key:
                continue
            parsed = _parse_number(item.get("count") or item.get("qty") or item.get("quantity") or 1)
            if parsed is None:
                continue
            amount = int(round(parsed))
            if amount > 0:
                result[str(key)] = amount
        return result
    return {}


def _scrub_text(text: str, path: str, patterns: List[Tuple[str, re.Pattern]], changes: List[str]) -> str:
    updated = text
    removed: List[str] = []
    for term, pattern in patterns:
        if pattern.search(updated.lower()):
            updated = pattern.sub("forbidden item", updated)
            removed.append(term)
    if removed and updated != text:
        changes.append(f"scrub:{path}:{','.join(removed)}")
    return updated


def _scrub_list(values: List[Any], path: str, patterns: List[Tuple[str, re.Pattern]], changes: List[str]) -> List[Any]:
    updated: List[Any] = []
    for idx, item in enumerate(values):
        if isinstance(item, str):
            cleaned = _scrub_text(item, f"{path}[{idx}]", patterns, changes)
            if cleaned:
                updated.append(cleaned)
        else:
            updated.append(item)
    return updated


def _normalize_unit(value: Any, changes: List[str], path: str) -> Any:
    original = value
    parsed = _parse_number(value)
    if parsed is None and isinstance(value, str):
        parsed = LEVEL_MAP.get(value.strip().lower())
    if parsed is None:
        return value

    normalized = parsed
    if normalized > 1.0:
        if normalized <= 10.0:
            normalized = normalized / 10.0
        elif normalized <= 100.0:
            normalized = normalized / 100.0
    if normalized < 0.0:
        normalized = 0.0
    if normalized > 1.0:
        normalized = 1.0
    normalized = round(normalized, 3)

    if normalized != original:
        changes.append(f"{path}:{original}->{normalized}")
    return normalized


def _normalize_disposition(value: Any, changes: List[str], path: str) -> Any:
    original = value
    parsed = _parse_number(value)
    if parsed is None and isinstance(value, str):
        text = value.strip().lower()
        scores = [score for key, score in DISPOSITION_KEYWORDS.items() if key in text]
        if scores:
            parsed = sum(scores) / len(scores)
    if parsed is None:
        return value

    normalized = parsed
    if normalized > 5.0 or normalized < -5.0:
        if 0.0 <= normalized <= 10.0:
            normalized = normalized - 5.0
        elif -10.0 <= normalized <= 10.0:
            normalized = normalized / 2.0
    normalized = int(round(normalized))
    if normalized > 5:
        normalized = 5
    if normalized < -5:
        normalized = -5

    if normalized != original:
        changes.append(f"{path}:{original}->{normalized}")
    return normalized


def _normalize_narrative_language(value: Any, changes: List[str], path: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    mapping = {
        "zh": "zh",
        "zh-cn": "zh",
        "zh_hans": "zh",
        "chinese": "zh",
        "中文": "zh",
        "cn": "zh",
        "en": "en",
        "en-us": "en",
        "english": "en",
        "英文": "en",
    }
    normalized = mapping.get(text)
    if normalized is None:
        return None
    if normalized != value:
        changes.append(f"{path}:{value}->{normalized}")
    return normalized
