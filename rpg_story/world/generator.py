"""World generation pipeline (Milestone 4)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Any, Dict, List
from datetime import datetime, timezone
import json
import math
import re

from rpg_story.config import AppConfig
from rpg_story.llm.client import BaseLLMClient, make_json_schema_response_format
from rpg_story.models.world import WorldSpec, GameState, QuestSpec, QuestProgress, MapPosition, NPCProfile
from rpg_story.persistence.store import generate_session_id, save_state, append_turn_log
from rpg_story.world.consistency import validate_world, find_anachronisms
from rpg_story.world.sanitize import sanitize_world_payload, summarize_changes

def _schema_hint() -> str:
    return (
        "WorldSpec JSON with fields: world_id, title, world_bible, locations, npcs, "
        "starting_location, starting_hook, initial_quest, main_quest, side_quests, map_layout. "
        "world_bible: {tech_level, narrative_language, magic_rules, tone, anachronism_policy, taboos, do_not_mention, "
        "anachronism_blocklist}. "
        "locations: [{location_id, name, kind, description, connected_to, tags}]. "
        "npcs: [{npc_id, name, profession, traits, goals, starting_location, obedience_level, stubbornness, "
        "risk_tolerance, disposition_to_player, refusal_style}]. "
        "main_quest: {quest_id, title, category='main', description, objective, giver_npc_id, suggested_location, "
        "required_items, reward_items, reward_hint}. "
        "side_quests: list of quests with category='side'. "
        "map_layout: [{location_id, x, y}] where x,y are 0..100 for relative map positions. "
        "Constraints: tech_level in {medieval, modern, sci-fi}; "
        "obedience_level/stubbornness/risk_tolerance are floats in [0.0,1.0]; "
        "disposition_to_player is an int in [-5,5]. No extra keys (do not include schema_version)."
    )


def generate_world_spec(cfg: AppConfig, llm: BaseLLMClient, world_prompt: str) -> WorldSpec:
    target_language = _detect_prompt_language(world_prompt)
    target_language_name = _language_name(target_language)
    response_format = make_json_schema_response_format(
        name="WorldSpec",
        schema=WorldSpec.model_json_schema(),
        description="World specification for a multi-genre narrative RPG world.",
    )
    system = (
        "You are a world generation engine for a multi-genre narrative setting. "
        "Choose the tech level based on the prompt. Output ONLY valid JSON with correct types and ranges."
    )
    user = (
        f"World prompt: {world_prompt}\n"
        f"Language requirement: ALL player-visible text MUST be in {target_language_name}. "
        "This includes title, location names/descriptions, NPC names/professions/traits/goals/refusal_style, "
        "starting_hook, initial_quest, quest title/description/objective/reward_hint, and item names.\n"
        "Generate a coherent world with 3-8 locations and enough NPCs for social play. "
        "Each location should have around 2-5 NPCs when possible. "
        "NPC personalities must be diverse: some cooperative, some stubborn, some neutral. "
        "At least one NPC in each location should be relatively cooperative for travel assistance. "
        "Ensure connected_to references valid location_ids. "
        "Use numeric types (not strings) for all numeric fields. No extra keys. "
        "Set world_bible.tech_level based on the prompt. "
        "Generate one concrete main quest (main_quest) with a clear objective aligned to the user prompt's genre. "
        "Generate exactly 3 side_quests tied to existing NPCs whenever the map has 3+ meaningful locations "
        "(or at least 2 side_quests for smaller maps), "
        "and each side quest should define reward_items. "
        "At generation time (first pass), side_quest required_items must already be world-theme aligned and location-specific. "
        "Do not output generic placeholders expecting later repair. "
        "Collectible (side quest required) item types across the generated world should be at least 5 when possible. "
        "Different locations should emphasize different collectible items instead of repeating the same set. "
        "Each side quest should require 2-3 concrete items that are plausible in that quest's location and social context. "
        "Use concrete, in-world item names. Avoid placeholder names such as '<location>样本', '<location>线索', "
        "'*_sample', '*_clue', '*_material', '*_token', and avoid repetitive patterns like '<location>遗物'/'<location>矿石' unless the world explicitly centers on archaeology/mining. "
        "NPC names must be proper names, not placeholders like '居民1'/'村民1'/'Resident 1'. "
        "NPC profession must fit their starting_location and setting tone. "
        "Main quest required_items should depend on side quest reward_items (main line unlocked via side quests). "
        "Main quest required_items MUST come from side quest reward_items, not directly collectible materials. "
        "Provide map_layout with relative x/y coordinates for each location (0..100). "
        "Populate world_bible.do_not_mention with terms inconsistent for THIS world. "
        "If tech_level is medieval, include modern tech (smartphone, internet, credit card, etc.). "
        "If tech_level is modern, do_not_mention can be empty or contain medieval-only taboos."
    )
    data = llm.generate_json(system, user, response_format=response_format)
    sanitized, changes = sanitize_world_payload(data)
    anachronism_matches: list[dict] | None = None
    try:
        if not isinstance(sanitized, dict):
            raise ValueError("world payload must be a JSON object")
        world = WorldSpec.model_validate(sanitized)
        validate_world(world, strict_bidirectional=cfg.worldgen.strict_bidirectional_edges)
        anachronism_matches = find_anachronisms(world)
        if anachronism_matches:
            summary = _summarize_banned_matches(anachronism_matches)
            raise ValueError(f"anachronism detected: {summary}")
        world = _enforce_world_language(world, llm, response_format, target_language)
        if _needs_semantic_polish(world):
            world = _polish_world_semantics(world, llm, response_format, target_language)
        world = _ensure_story_structures(world, target_language=target_language)
        return world
    except Exception as exc:
        # single rewrite attempt
        error_summary = _summarize_validation_error(exc)
        anachronism_block = ""
        if anachronism_matches:
            keywords = _unique_values(anachronism_matches, "keyword")
            paths = _unique_values(anachronism_matches, "path")
            anachronism_block = (
                "Anachronisms detected:\n"
                f"- Remove ALL occurrences of these anachronisms from player-visible narrative fields: "
                f"{', '.join(keywords)}.\n"
                "- You MAY keep them in world_bible.do_not_mention, but you MUST remove them from narrative fields.\n"
                f"- Matched paths: {', '.join(paths)}.\n\n"
            )
        rewrite_system = (
            "You are a JSON repair tool. Return ONLY valid JSON. No markdown. "
            "Maintain consistency with world_bible.tech_level."
        )
        rewrite_user = (
            "Fix the following JSON to satisfy the schema and constraints.\n\n"
            "Constraints:\n"
            "- No extra keys (remove schema_version or unknown fields)\n"
            "- obedience_level/stubbornness/risk_tolerance: float 0.0..1.0\n"
            "- disposition_to_player: integer -5..5\n"
            "- traits/goals/connected_to/tags: arrays of strings\n\n"
            f"{anachronism_block}"
            f"Validation errors: {error_summary}\n\n"
            f"JSON to fix: {json.dumps(sanitized, ensure_ascii=False)}"
        )
        fixed = llm.generate_json(rewrite_system, rewrite_user, response_format=response_format)
        fixed_sanitized, fixed_changes = sanitize_world_payload(fixed)
        try:
            if not isinstance(fixed_sanitized, dict):
                raise ValueError("world payload must be a JSON object after rewrite")
            world = WorldSpec.model_validate(fixed_sanitized)
        except Exception as exc2:
            error_summary = _summarize_validation_error(exc2)
            change_summary = summarize_changes(changes + fixed_changes)
            raise ValueError(
                "WorldSpec validation failed after rewrite. "
                f"errors: {error_summary}. sanitization: {change_summary}"
            ) from exc2
        validate_world(world, strict_bidirectional=cfg.worldgen.strict_bidirectional_edges)
        anachronism_matches = find_anachronisms(world)
        if anachronism_matches:
            summary = _summarize_banned_matches(anachronism_matches)
            change_summary = summarize_changes(changes + fixed_changes)
            keywords = ", ".join(_unique_values(anachronism_matches, "keyword"))
            raise ValueError(
                "anachronism detected after rewrite: "
                f"{summary}. keywords: {keywords}. sanitization: {change_summary}"
            )
        world = _enforce_world_language(world, llm, response_format, target_language)
        if _needs_semantic_polish(world):
            world = _polish_world_semantics(world, llm, response_format, target_language)
        world = _ensure_story_structures(world, target_language=target_language)
        return world


def _summarize_validation_error(exc: Exception, limit: int = 8) -> str:
    errors_fn = getattr(exc, "errors", None)
    if callable(errors_fn):
        try:
            errors = errors_fn()
        except Exception:
            return str(exc)
        parts = []
        for err in errors[:limit]:
            loc = ".".join(str(item) for item in err.get("loc", []))
            msg = err.get("msg", "invalid")
            if loc:
                parts.append(f"{loc}: {msg}")
            else:
                parts.append(msg)
        if len(errors) > limit:
            parts.append(f"+{len(errors) - limit} more")
        return "; ".join(parts)
    return str(exc)


def _summarize_banned_matches(matches: list[dict], limit: int = 4) -> str:
    parts = []
    for match in matches[:limit]:
        keyword = match.get("keyword", "?")
        path = match.get("path", "?")
        parts.append(f"{keyword}@{path}")
    if len(matches) > limit:
        parts.append(f"+{len(matches) - limit} more")
    return "; ".join(parts)


def _unique_values(matches: list[dict], key: str) -> list[str]:
    seen = set()
    values: list[str] = []
    for match in matches:
        value = match.get(key)
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(str(value))
    return values


def _detect_prompt_language(world_prompt: str) -> str:
    return "zh" if _contains_cjk(world_prompt or "") else "en"


def _language_name(language: str) -> str:
    return "Chinese" if language == "zh" else "English"


def _world_matches_language(world: WorldSpec, target_language: str) -> bool:
    if target_language not in {"zh", "en"}:
        return True
    text_parts = [world.title or "", world.starting_hook or "", world.initial_quest or ""]
    for loc in world.locations:
        text_parts.extend([loc.name or "", loc.description or ""])
    for npc in world.npcs:
        text_parts.extend(
            [
                npc.name or "",
                npc.profession or "",
                " ".join(npc.traits or []),
                " ".join(npc.goals or []),
                npc.refusal_style or "",
            ]
        )
    if world.main_quest:
        text_parts.extend(
            [
                world.main_quest.title or "",
                world.main_quest.description or "",
                world.main_quest.objective or "",
                world.main_quest.reward_hint or "",
            ]
        )
        text_parts.extend(str(item) for item in (world.main_quest.required_items or {}).keys())
        text_parts.extend(str(item) for item in (world.main_quest.reward_items or {}).keys())
    for q in world.side_quests:
        text_parts.extend([q.title or "", q.description or "", q.objective or "", q.reward_hint or ""])
        text_parts.extend(str(item) for item in (q.required_items or {}).keys())
        text_parts.extend(str(item) for item in (q.reward_items or {}).keys())
    cjk_count = 0
    latin_count = 0
    for part in text_parts:
        text = str(part or "").strip()
        if not text:
            continue
        has_cjk = _contains_cjk(text)
        has_latin = bool(re.search(r"[A-Za-z]", text))
        if has_cjk:
            cjk_count += 1
        elif has_latin:
            latin_count += 1

    language_marked = cjk_count + latin_count
    if language_marked == 0:
        return True
    if target_language == "zh":
        # Treat as Chinese only if most user-facing fields are actually Chinese.
        return cjk_count / language_marked >= 0.8
    return latin_count / language_marked >= 0.8


def _enforce_world_language(
    world: WorldSpec,
    llm: BaseLLMClient,
    response_format: dict,
    target_language: str,
) -> WorldSpec:
    language = target_language if target_language in {"zh", "en"} else "en"
    current = world.model_copy(deep=True)
    current.world_bible.narrative_language = language
    if _world_matches_language(current, language):
        return current

    target_name = _language_name(language)
    rewrite_system = (
        "You are a JSON repair and localization tool. Return ONLY valid JSON. "
        "Do not change IDs, graph structure, numeric values, or quest logic."
    )
    rewrite_user = (
        f"Translate/localize ALL player-visible text fields to {target_name}.\n"
        "Keep these fields exactly unchanged: world_id, location_id, npc_id, quest_id, "
        "starting_location, connected_to, suggested_location, map_layout coordinates, and all numeric values.\n"
        "Set world_bible.narrative_language to the target language code ('zh' or 'en').\n"
        "Do not add or remove locations/NPCs/quests. Keep item keys semantically consistent.\n\n"
        f"Target language code: {language}\n"
        f"Original JSON: {json.dumps(current.model_dump(), ensure_ascii=False)}"
    )
    fixed = llm.generate_json(rewrite_system, rewrite_user, response_format=response_format)
    sanitized, _changes = sanitize_world_payload(fixed)
    if not isinstance(sanitized, dict):
        raise ValueError("language localization failed: rewrite payload is not an object")
    fixed_world = WorldSpec.model_validate(sanitized)
    fixed_world.world_bible.narrative_language = language
    if not _world_matches_language(fixed_world, language):
        raise ValueError(f"language localization failed: world is not in target language {language}")
    return fixed_world


_NPC_PLACEHOLDER_ZH = re.compile(r"(居民|村民|市民|角色)\s*\d+$")
_NPC_PLACEHOLDER_EN = re.compile(r"(resident|villager|citizen|character)\s*\d+$", re.IGNORECASE)
_ITEM_PLACEHOLDER_ZH = re.compile(r"(样本|线索|材料|物资)\s*\d*$")
_ITEM_PLACEHOLDER_EN = re.compile(r"(sample|clue|material|token)\s*\d*$", re.IGNORECASE)


def _needs_semantic_polish(world: WorldSpec) -> bool:
    loc_kinds = {loc.location_id: str(loc.kind or "").lower() for loc in world.locations}
    for npc in world.npcs:
        name = str(npc.name or "").strip()
        if _NPC_PLACEHOLDER_ZH.search(name) or _NPC_PLACEHOLDER_EN.search(name):
            return True
        prof = str(npc.profession or "").strip().lower()
        kind = loc_kinds.get(npc.starting_location, "")
        if kind in {"castle", "dungeon", "ruin"} and prof in {"村长", "village chief"}:
            return True
    item_names: list[str] = []
    if world.main_quest:
        item_names.extend(str(k) for k in (world.main_quest.required_items or {}).keys())
        item_names.extend(str(k) for k in (world.main_quest.reward_items or {}).keys())
    for quest in world.side_quests:
        item_names.extend(str(k) for k in (quest.required_items or {}).keys())
        item_names.extend(str(k) for k in (quest.reward_items or {}).keys())
    for item in item_names:
        text = str(item or "").strip()
        if not text:
            continue
        if _ITEM_PLACEHOLDER_ZH.search(text) or _ITEM_PLACEHOLDER_EN.search(text):
            return True
    return False


def _polish_world_semantics(
    world: WorldSpec,
    llm: BaseLLMClient,
    response_format: dict,
    target_language: str,
) -> WorldSpec:
    target_name = _language_name(target_language)
    rewrite_system = (
        "You are a world-semantic polishing tool. Return ONLY valid JSON. "
        "Improve semantic quality while preserving structure."
    )
    rewrite_user = (
        f"Polish this WorldSpec in {target_name}.\n"
        "Requirements:\n"
        "1) Keep ALL ids and graph structure unchanged: world_id, location_id, npc_id, quest_id, connected_to, "
        "starting_location, suggested_location, map_layout coordinates, and all numeric fields.\n"
        "2) NPC names must be proper names (no placeholders like 居民1 / 村民1 / Resident 1 / Character 2).\n"
        "3) NPC profession must fit the NPC's starting location and setting tone.\n"
        "4) Quest required_items/reward_items names must be concrete in-world nouns; avoid placeholders like "
        "<location>样本 / <location>线索 / *_sample / *_clue / *_material / *_token.\n"
        "5) Keep quest logic and dependencies intact (do not break main->side dependency).\n"
        "6) Keep narrative language consistent with world_bible.narrative_language.\n\n"
        f"Original JSON: {json.dumps(world.model_dump(), ensure_ascii=False)}"
    )
    fixed = llm.generate_json(rewrite_system, rewrite_user, response_format=response_format)
    sanitized, _ = sanitize_world_payload(fixed)
    if not isinstance(sanitized, dict):
        raise ValueError("semantic polishing failed: payload is not an object")
    polished = WorldSpec.model_validate(sanitized)
    if _needs_semantic_polish(polished):
        raise ValueError("semantic polishing failed: placeholders still remain")
    return polished


def initialize_game_state(world: WorldSpec, session_id: str, created_at: Optional[str] = None) -> GameState:
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    world = _ensure_story_structures(world)
    npc_locations = {npc.npc_id: npc.starting_location for npc in world.npcs}
    quest_journal, main_quest_id = _build_initial_quest_journal(world)
    quest_status = {quest_id: progress.status for quest_id, progress in quest_journal.items()}
    state = GameState(
        session_id=session_id,
        created_at=created_at,
        world=world,
        player_location=world.starting_location,
        npc_locations=npc_locations,
        flags={},
        quests=quest_status,
        quest_journal=quest_journal,
        main_quest_id=main_quest_id,
        inventory={},
        location_resource_stock={},
        recent_summaries=[],
        last_turn_id=0,
    )
    return GameState.model_validate(state.model_dump())


def create_new_session(
    cfg: AppConfig,
    llm: BaseLLMClient,
    world_prompt: str,
    sessions_root: Optional[Path] = None,
    worlds_root: Optional[Path] = None,
) -> Tuple[str, WorldSpec, GameState]:
    session_id = generate_session_id()
    world = generate_world_spec(cfg, llm, world_prompt)
    state = initialize_game_state(world, session_id=session_id)

    # persist world
    worlds_dir = worlds_root or cfg.app.worlds_dir
    world_dir = Path(worlds_dir) / session_id
    world_dir.mkdir(parents=True, exist_ok=True)
    world_path = world_dir / "world.json"
    world_path.write_text(json.dumps(world.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

    # persist state
    sessions_dir = sessions_root or cfg.app.sessions_dir
    save_state(session_id, state, Path(sessions_dir))

    # optional worldgen log
    append_turn_log(
        session_id,
        {
            "event_type": "worldgen",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "world_id": world.world_id,
        },
        Path(sessions_dir),
    )

    return session_id, world, state


def _build_initial_quest_journal(world: WorldSpec) -> tuple[dict[str, QuestProgress], str | None]:
    journal: dict[str, QuestProgress] = {}
    main_quest_id: str | None = None
    if world.main_quest:
        q = world.main_quest
        main_quest_id = q.quest_id
        journal[q.quest_id] = QuestProgress(
            quest_id=q.quest_id,
            title=q.title,
            category="main",
            status="active",
            objective=q.objective,
            guidance=q.description,
            giver_npc_id=q.giver_npc_id,
            required_items=q.required_items,
            collected_items={item: 0 for item in q.required_items.keys()},
            reward_items=q.reward_items,
            reward_hint=q.reward_hint,
        )
    for side in world.side_quests:
        journal[side.quest_id] = QuestProgress(
            quest_id=side.quest_id,
            title=side.title,
            category="side",
            status="available",
            objective=side.objective,
            guidance=side.description,
            giver_npc_id=side.giver_npc_id,
            required_items=side.required_items,
            collected_items={item: 0 for item in side.required_items.keys()},
            reward_items=side.reward_items,
            reward_hint=side.reward_hint,
        )
    return journal, main_quest_id


def _ensure_story_structures(world: WorldSpec, target_language: str | None = None) -> WorldSpec:
    updated = world.model_copy(deep=True)
    language = target_language or getattr(updated.world_bible, "narrative_language", None)
    if language not in {"zh", "en"}:
        language = "zh" if _is_chinese_world(updated) else "en"
    updated.world_bible.narrative_language = language
    prefer_chinese = language == "zh"
    updated.npcs = _normalize_npc_professions(updated, prefer_chinese=prefer_chinese)
    updated.npcs = _ensure_npc_density(updated, prefer_chinese=prefer_chinese)
    updated.npcs = _ensure_unique_npc_names(updated, updated.npcs, prefer_chinese=prefer_chinese)
    updated.side_quests = _normalize_side_quests(updated, prefer_chinese=prefer_chinese)
    main_required = _aggregate_side_rewards(updated.side_quests)
    if not updated.main_quest:
        updated.main_quest = QuestSpec(
            quest_id=f"main_{updated.world_id}",
            title="主线终章" if prefer_chinese else "Main Quest Finale",
            category="main",
            description=updated.initial_quest,
            objective=updated.initial_quest,
            giver_npc_id=updated.npcs[0].npc_id if updated.npcs else None,
            suggested_location=updated.starting_location,
            required_items=main_required,
            reward_hint="推进主线剧情。" if prefer_chinese else "Advance the main story arc.",
        )
    else:
        updated.main_quest.category = "main"
        if not updated.main_quest.giver_npc_id and updated.npcs:
            updated.main_quest.giver_npc_id = updated.npcs[0].npc_id
        if not updated.main_quest.suggested_location:
            updated.main_quest.suggested_location = updated.starting_location
        updated.main_quest.required_items = main_required
    updated.main_quest.reward_items = {}
    if main_required:
        if prefer_chinese:
            updated.main_quest.objective = "完成支线并收集关键道具：" + "，".join(
                [f"{k} x{v}" for k, v in main_required.items()]
            )
            updated.main_quest.description = (
                "主线推进条件：先完成各支线任务，获得关键任务道具，再返回推进终章。"
            )
        else:
            updated.main_quest.objective = "Finish side quests and collect key items: " + ", ".join(
                [f"{k} x{v}" for k, v in main_required.items()]
            )
            updated.main_quest.description = (
                "Main progression requires completing side quests and obtaining their reward items."
            )
    if not updated.map_layout:
        updated.map_layout = _default_map_layout(updated)
    return WorldSpec.model_validate(updated.model_dump())


def _default_map_layout(world: WorldSpec) -> list[MapPosition]:
    if not world.locations:
        return []
    size = len(world.locations)
    if size == 1:
        return [MapPosition(location_id=world.locations[0].location_id, x=50.0, y=50.0)]
    nodes: list[MapPosition] = []
    for idx, loc in enumerate(world.locations):
        angle = (idx / size) * 6.283185307179586
        x = 50.0 + 35.0 * math.cos(angle)
        y = 50.0 + 30.0 * math.sin(angle)
        nodes.append(MapPosition(location_id=loc.location_id, x=round(x, 2), y=round(y, 2)))
    return nodes


def _is_chinese_world(world: WorldSpec) -> bool:
    lang = getattr(world.world_bible, "narrative_language", None)
    if lang in {"zh", "en"}:
        return lang == "zh"
    text = " ".join([world.title or "", world.starting_hook or "", world.initial_quest or ""])
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))

def _normalize_item_token(value: str) -> str:
    token = re.sub(r"[\s\-]+", "_", str(value or "").strip().lower())
    token = re.sub(r"[^0-9a-z_\u4e00-\u9fff]", "", token)
    token = re.sub(r"_+", "_", token)
    return token.strip("_")


def _derive_item_label_from_token(token: str, *, prefer_chinese: bool, index: int) -> str:
    normalized = _normalize_item_token(token)
    if prefer_chinese:
        if _contains_cjk(normalized):
            return normalized
        return f"任务物资{index}"
    if not normalized:
        return f"quest_item_{index}"
    return normalized


def _localize_item_map(
    items: dict[str, int],
    *,
    prefer_chinese: bool,
) -> tuple[dict[str, int], dict[str, str]]:
    localized: dict[str, int] = {}
    renamed: dict[str, str] = {}
    token_to_local: dict[str, str] = {}
    serial = 1

    for raw_name, raw_count in (items or {}).items():
        try:
            count = int(raw_count)
        except Exception:
            continue
        if count <= 0:
            continue

        source = str(raw_name or "").strip()
        token = _normalize_item_token(source)
        if token in token_to_local:
            target = token_to_local[token]
        else:
            target = _derive_item_label_from_token(source, prefer_chinese=prefer_chinese, index=serial)
            token_to_local[token] = target
            serial += 1

        localized[target] = int(localized.get(target, 0)) + count
        if source and source != target:
            renamed[source] = target
    return localized, renamed


def _replace_item_mentions(text: str, replacements: dict[str, str]) -> str:
    updated = str(text or "")
    if not updated or not replacements:
        return updated

    pairs = sorted(
        [(str(src).strip(), str(dst).strip()) for src, dst in replacements.items() if str(src).strip()],
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
    for src, dst in pairs:
        if not dst or src == dst:
            continue
        variants = {
            src,
            src.replace("_", " "),
            src.replace("-", " "),
            src.replace(" ", "_"),
            src.replace(" ", "-"),
        }
        token = _normalize_item_token(src)
        if token:
            variants.add(token)
            variants.add(token.replace("_", " "))
            variants.add(token.replace("_", "-"))
        for variant in sorted(variants, key=len, reverse=True):
            if not variant:
                continue
            updated = re.sub(rf"(?<![\w]){re.escape(variant)}(?![\w])", dst, updated, flags=re.IGNORECASE)
    return updated


def _collect_item_pool_from_world(world: WorldSpec) -> list[str]:
    pool: list[str] = []
    for quest in world.side_quests:
        for item in (quest.required_items or {}).keys():
            name = str(item or "").strip()
            if name:
                pool.append(name)
        for item in (quest.reward_items or {}).keys():
            name = str(item or "").strip()
            if name:
                pool.append(name)
    if world.main_quest:
        for item in (world.main_quest.required_items or {}).keys():
            name = str(item or "").strip()
            if name:
                pool.append(name)
    seen = set()
    deduped: list[str] = []
    for name in pool:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def _pick_items_from_pool(pool: list[str], *, seed: str, limit: int = 2) -> dict[str, int]:
    if not pool:
        return {}
    start = sum(ord(ch) for ch in seed) % len(pool)
    out: dict[str, int] = {}
    for i in range(min(limit, len(pool))):
        item = pool[(start + i) % len(pool)]
        out[item] = 2 if i == 0 else 1
    return out


def _seed_item_from_location(loc: Any, *, prefer_chinese: bool, variant: int) -> str:
    loc_name = str(getattr(loc, "name", "") or "").strip()
    loc_kind = str(getattr(loc, "kind", "") or "").strip()
    base = loc_name or loc_kind or ("地区" if prefer_chinese else "area")
    if prefer_chinese:
        clean = re.sub(r"\s+", "", base)
        suffixes = ["遗物", "手稿", "矿石", "徽记"]
        return f"{clean}{suffixes[variant % len(suffixes)]}"
    token = _normalize_item_token(base) or "area"
    suffixes = ["relic", "manuscript", "ore", "insignia"]
    return f"{token}_{suffixes[variant % len(suffixes)]}"


def suggest_location_resource_template(world: WorldSpec, loc: Any, *, prefer_chinese: bool) -> dict[str, int]:
    loc_id = str(getattr(loc, "location_id", "") or "")
    blocked_tokens = set()
    if world.main_quest:
        blocked_tokens.update(_normalize_item_token(name) for name in (world.main_quest.required_items or {}).keys())
    for quest in world.side_quests:
        blocked_tokens.update(_normalize_item_token(name) for name in (quest.reward_items or {}).keys())

    required_items: dict[str, int] = {}
    for quest in world.side_quests:
        if quest.suggested_location and quest.suggested_location != loc_id:
            continue
        for item, count in (quest.required_items or {}).items():
            required_items[str(item)] = max(required_items.get(str(item), 0), max(1, int(count)))

    result: dict[str, int] = {}
    for idx, (item, count) in enumerate(sorted(required_items.items())):
        result[item] = min(6, max(1, int(count) + (idx % 2)))
        if len(result) >= 3:
            break

    seed_variant = sum(ord(ch) for ch in (loc_id + str(getattr(loc, "kind", "") or ""))) % 11
    while len(result) < 2:
        candidate = _seed_item_from_location(loc, prefer_chinese=prefer_chinese, variant=seed_variant + len(result))
        token = _normalize_item_token(candidate)
        if token and token not in blocked_tokens and candidate not in result:
            result[candidate] = 1
        else:
            seed_variant += 3
            if seed_variant > 50:
                break
    if not result:
        fallback = _default_required_items_for_location(loc, prefer_chinese=prefer_chinese, variant=seed_variant)
        for item, count in fallback.items():
            token = _normalize_item_token(item)
            if token in blocked_tokens:
                continue
            result[item] = count
            if len(result) >= 2:
                break
    return result


def _normalize_side_quests(world: WorldSpec, *, prefer_chinese: bool) -> list[QuestSpec]:
    # Keep generated side quest count whenever possible; only create one when absent.
    target_count = min(3, max(1, len(world.side_quests)))
    if len(world.locations) >= 3:
        target_count = max(target_count, 3)
    locations = [loc for loc in world.locations if loc.location_id != world.starting_location] or list(world.locations)
    npcs = list(world.npcs)
    normalized: list[QuestSpec] = []
    used_ids = set()

    def _pick_reward(idx: int, loc: Any) -> dict[str, int]:
        item_name = _seed_item_from_location(loc, prefer_chinese=prefer_chinese, variant=idx + 7)
        return {item_name: 1}

    for idx, quest in enumerate(world.side_quests):
        if len(normalized) >= target_count:
            break
        loc = locations[idx % len(locations)] if locations else None
        npc = npcs[idx % len(npcs)] if npcs else None

        quest_id = str(quest.quest_id or f"side_{idx+1}")
        if quest_id in used_ids:
            quest_id = f"{quest_id}_{idx+1}"
        used_ids.add(quest_id)

        required_items = dict(quest.required_items or {})
        if not required_items and loc:
            required_items = _default_required_items_for_location(loc, prefer_chinese=prefer_chinese, variant=idx)
        required_items, required_replacements = _localize_item_map(required_items, prefer_chinese=prefer_chinese)

        reward_items = dict(quest.reward_items or {})
        if not reward_items:
            reward_items = _pick_reward(idx, loc)
        reward_items, reward_replacements = _localize_item_map(reward_items, prefer_chinese=prefer_chinese)

        text_replacements = dict(required_replacements)
        text_replacements.update(reward_replacements)

        objective_text = quest.objective or _default_side_objective(required_items, loc, prefer_chinese)
        description_text = quest.description or _default_side_description(loc, prefer_chinese)
        reward_hint_text = quest.reward_hint or _default_reward_hint(reward_items, prefer_chinese)

        objective_text = _replace_item_mentions(objective_text, text_replacements)
        description_text = _replace_item_mentions(description_text, text_replacements)
        reward_hint_text = _replace_item_mentions(reward_hint_text, text_replacements)

        objective_text = _ensure_side_objective_consistency(
            objective_text,
            required_items,
            loc,
            prefer_chinese=prefer_chinese,
        )
        title_text = _ensure_side_title_consistency(
            quest.title or _default_side_title(loc, idx, prefer_chinese),
            required_items,
            prefer_chinese=prefer_chinese,
        )

        normalized.append(
            QuestSpec(
                quest_id=quest_id,
                title=title_text,
                category="side",
                description=description_text,
                objective=objective_text,
                giver_npc_id=quest.giver_npc_id or (npc.npc_id if npc else None),
                suggested_location=quest.suggested_location or (loc.location_id if loc else None),
                required_items=required_items,
                reward_items=reward_items,
                reward_hint=reward_hint_text,
            )
        )

    while len(normalized) < target_count and locations:
        idx = len(normalized)
        loc = locations[idx % len(locations)]
        npc = npcs[idx % len(npcs)] if npcs else None
        required_items = _default_required_items_for_location(loc, prefer_chinese=prefer_chinese, variant=idx)
        reward_items = _pick_reward(idx, loc)
        required_items, _ = _localize_item_map(required_items, prefer_chinese=prefer_chinese)
        reward_items, _ = _localize_item_map(reward_items, prefer_chinese=prefer_chinese)
        quest_id = f"side_{loc.location_id}_{idx+1}"
        if quest_id in used_ids:
            quest_id = f"{quest_id}_{len(used_ids)+1}"
        used_ids.add(quest_id)
        normalized.append(
            QuestSpec(
                quest_id=quest_id,
                title=_default_side_title(loc, idx, prefer_chinese),
                category="side",
                description=_default_side_description(loc, prefer_chinese),
                objective=_default_side_objective(required_items, loc, prefer_chinese),
                giver_npc_id=npc.npc_id if npc else None,
                suggested_location=loc.location_id,
                required_items=required_items,
                reward_items=reward_items,
                reward_hint=_default_reward_hint(reward_items, prefer_chinese),
            )
        )
    return normalized


def _default_required_items_for_location(
    loc: Any,
    prefer_chinese: bool,
    *,
    variant: int = 0,
) -> dict[str, int]:
    primary = _seed_item_from_location(loc, prefer_chinese=prefer_chinese, variant=variant)
    secondary = _seed_item_from_location(loc, prefer_chinese=prefer_chinese, variant=variant + 1)
    result = {primary: 2 + (variant % 2)}
    if secondary != primary:
        result[secondary] = 1 + (variant % 2)
    return result


def _default_side_title(loc: Any, idx: int, prefer_chinese: bool) -> str:
    loc_name = str(getattr(loc, "name", "") or "").strip() or str(getattr(loc, "location_id", "") or f"loc_{idx+1}")
    if prefer_chinese:
        return f"{loc_name}委托"
    return f"{loc_name} Request"


def _default_side_description(loc: Any, prefer_chinese: bool) -> str:
    loc_name = str(getattr(loc, "name", "") or "").strip() if loc is not None else ""
    if prefer_chinese:
        if loc_name:
            return f"前往{loc_name}并完成该地点委托所需的物资准备。"
        return "前往目标地点并完成委托所需的物资准备。"
    if loc_name:
        return f"Travel to {loc_name} and prepare the materials required by this local request."
    return "Travel to the target location and prepare the required materials."


def _default_side_objective(required_items: dict[str, int], loc: Any, prefer_chinese: bool) -> str:
    req = "，".join([f"{name} x{count}" for name, count in required_items.items()])
    loc_name = str(getattr(loc, "name", "") or "").strip() if loc is not None else ""
    if prefer_chinese:
        if loc_name:
            return f"在{loc_name}收集并交付：{req}"
        return f"收集并交付：{req}"
    if loc_name:
        return f"Collect and deliver at {loc_name}: {req}"
    return f"Collect and deliver: {req}"


def _ensure_side_objective_consistency(
    objective_text: str,
    required_items: dict[str, int],
    loc: Any,
    *,
    prefer_chinese: bool,
) -> str:
    text = str(objective_text or "").strip()
    if not required_items:
        return text
    if not text:
        return _default_side_objective(required_items, loc, prefer_chinese)
    item_names = [str(name) for name in required_items.keys()]
    mentions_any = any(name and name in text for name in item_names)
    if mentions_any:
        return text
    req = "，".join([f"{name} x{count}" for name, count in required_items.items()])
    if prefer_chinese:
        return f"{text}（需求：{req}）"
    return f"{text} (required: {req})"


def _ensure_side_title_consistency(title: str, required_items: dict[str, int], *, prefer_chinese: bool) -> str:
    text = str(title or "").strip()
    if not text:
        return text
    if not required_items:
        return text
    first_item = next(iter(required_items.keys()))
    mentions_any = any(str(name) in text for name in required_items.keys())
    if mentions_any:
        return text
    if prefer_chinese:
        if any(token in text for token in ("寻找", "收集", "采集", "获取", "委托")):
            return f"收集{first_item}"
        return text
    lower = text.lower()
    if any(token in lower for token in ("find", "collect", "gather", "obtain", "fetch", "request")):
        return f"Collect {first_item}"
    return text


def _default_reward_hint(reward_items: dict[str, int], prefer_chinese: bool) -> str:
    req = "，".join([f"{name} x{count}" for name, count in reward_items.items()])
    if prefer_chinese:
        return f"完成后可获得：{req}"
    return f"Reward on completion: {req}"


def _aggregate_side_rewards(side_quests: list[QuestSpec]) -> dict[str, int]:
    required: dict[str, int] = {}
    for quest in side_quests:
        for item, count in (quest.reward_items or {}).items():
            required[item] = max(required.get(item, 0), int(count))
    return required


def _normalize_npc_professions(world: WorldSpec, *, prefer_chinese: bool) -> list[NPCProfile]:
    npcs = [npc.model_copy(deep=True) for npc in world.npcs]
    if not npcs:
        return npcs
    loc_map = {loc.location_id: loc for loc in world.locations}
    for npc in npcs:
        loc = loc_map.get(npc.starting_location)
        cleaned = _clean_npc_profession(
            str(npc.profession or "").strip(),
            loc=loc,
            npc_name=str(npc.name or "").strip(),
            prefer_chinese=prefer_chinese,
        )
        if cleaned:
            npc.profession = cleaned
            continue
        npc.profession = _profession_from_location(loc, prefer_chinese=prefer_chinese)
    return npcs


def _generic_trait_sets(*, prefer_chinese: bool) -> list[tuple[list[str], list[str], float, float, float, int, str]]:
    if prefer_chinese:
        return [
            (["友善", "乐于协作"], ["帮助来访者"], 0.82, 0.25, 0.45, 2, "真诚直接"),
            (["谨慎", "保守"], ["避免风险"], 0.38, 0.7, 0.22, -1, "含蓄防备"),
            (["理性", "稳定"], ["完成本地职责"], 0.58, 0.42, 0.4, 1, "简洁克制"),
            (["果断", "强势"], ["维护规则"], 0.65, 0.5, 0.62, 0, "干脆坚定"),
        ]
    return [
        (["friendly", "cooperative"], ["help visitors"], 0.82, 0.25, 0.45, 2, "honest and direct"),
        (["careful", "conservative"], ["avoid risk"], 0.38, 0.7, 0.22, -1, "guarded and cautious"),
        (["rational", "steady"], ["fulfill local duties"], 0.58, 0.42, 0.4, 1, "concise and restrained"),
        (["decisive", "assertive"], ["maintain order"], 0.65, 0.5, 0.62, 0, "firm and efficient"),
    ]


def _profession_seed_pool(world: WorldSpec, *, prefer_chinese: bool) -> list[str]:
    seeds = [str(npc.profession or "").strip() for npc in world.npcs if str(npc.profession or "").strip()]
    if seeds:
        seen = set()
        ordered = []
        for item in seeds:
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)
        return ordered
    if prefer_chinese:
        return ["居民", "工作人员"]
    return ["Resident", "Staff"]


def _profession_from_location(loc: Any, *, prefer_chinese: bool) -> str:
    kind = str(getattr(loc, "kind", "") or "").strip().lower()
    loc_name = str(getattr(loc, "name", "") or "").strip()
    if prefer_chinese:
        if kind in {"castle", "fort", "stronghold"}:
            return "守卫"
        if kind in {"forest", "woods"}:
            return "巡林员"
        if kind in {"town", "village", "city"}:
            return "市民"
        if kind in {"library"}:
            return "馆员"
        if kind in {"dungeon", "ruin"}:
            return "探查员"
        return "工作人员"
    if kind in {"castle", "fort", "stronghold"}:
        return "Guard"
    if kind in {"forest", "woods"}:
        return "Ranger"
    if kind in {"town", "village", "city"}:
        return "Citizen"
    if kind in {"library"}:
        return "Librarian"
    if kind in {"dungeon", "ruin"}:
        return "Scout"
    return "Staff"


def _clean_npc_profession(text: str, *, loc: Any, npc_name: str, prefer_chinese: bool) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    loc_name = str(getattr(loc, "name", "") or "").strip()
    lowered = raw.lower()

    if loc_name:
        raw = raw.replace(loc_name, "").strip()
        lowered = raw.lower()

    raw = re.sub(r"\s+", " ", raw).strip(" -_·")
    lowered = raw.lower()
    if not raw:
        return ""
    if raw == npc_name:
        return ""
    if re.search(r"\d+$", raw):
        return ""
    if _NPC_PLACEHOLDER_ZH.search(raw) or _NPC_PLACEHOLDER_EN.search(raw):
        return ""
    if prefer_chinese and raw in {"人员", "角色"}:
        return ""
    if not prefer_chinese and lowered in {"person", "character"}:
        return ""
    return raw


def _ensure_npc_density(world: WorldSpec, *, prefer_chinese: bool) -> list[NPCProfile]:
    locations = list(world.locations)
    npcs = [npc.model_copy(deep=True) for npc in world.npcs]
    if not locations:
        return npcs

    by_loc: dict[str, list[NPCProfile]] = {loc.location_id: [] for loc in locations}
    for npc in npcs:
        if npc.starting_location in by_loc:
            by_loc[npc.starting_location].append(npc)
        else:
            npc.starting_location = world.starting_location
            by_loc.setdefault(world.starting_location, []).append(npc)

    target_min = 1
    target_max = 5
    profession_pool = _profession_seed_pool(world, prefer_chinese=prefer_chinese)
    trait_sets = _generic_trait_sets(prefer_chinese=prefer_chinese)

    used_ids = {npc.npc_id for npc in npcs}
    serial_id = 1

    def next_id() -> str:
        nonlocal serial_id
        while True:
            candidate = f"npc_auto_{serial_id:03d}"
            serial_id += 1
            if candidate not in used_ids:
                used_ids.add(candidate)
                return candidate

    for loc in locations:
        current = by_loc.get(loc.location_id, [])
        if len(current) <= target_max:
            continue
        # Keep original NPC placement to avoid role/location mismatch introduced by relocation.
        by_loc[loc.location_id] = current[:target_max]

    for loc in locations:
        current = by_loc.get(loc.location_id, [])
        while len(current) < target_min:
            idx = len(npcs)
            traits, goals, obedience, stubborn, risk, disp, refusal = trait_sets[idx % len(trait_sets)]
            profile = NPCProfile(
                npc_id=next_id(),
                name="",
                profession=_profession_from_location(loc, prefer_chinese=prefer_chinese)
                or profession_pool[idx % len(profession_pool)],
                traits=list(traits),
                goals=list(goals),
                starting_location=loc.location_id,
                obedience_level=obedience,
                stubbornness=stubborn,
                risk_tolerance=risk,
                disposition_to_player=disp,
                refusal_style=refusal,
            )
            npcs.append(profile)
            current.append(profile)

    result: list[NPCProfile] = []
    seen = set()
    for loc in locations:
        for npc in by_loc.get(loc.location_id, []):
            if npc.npc_id in seen:
                continue
            seen.add(npc.npc_id)
            result.append(npc)
    for npc in npcs:
        if npc.npc_id in seen:
            continue
        seen.add(npc.npc_id)
        result.append(npc)
    return result


def _ensure_unique_npc_names(world: WorldSpec, npcs: list[NPCProfile], *, prefer_chinese: bool) -> list[NPCProfile]:
    loc_map = {loc.location_id: loc for loc in world.locations}
    used: set[str] = set()
    per_loc_counter: dict[str, int] = {}
    normalized: list[NPCProfile] = []

    for npc in npcs:
        profile = npc.model_copy(deep=True)
        current_name = str(profile.name or "").strip()
        loc = loc_map.get(profile.starting_location)
        if (
            current_name
            and current_name not in used
            and not _npc_name_needs_rewrite(
                current_name,
                profession=str(profile.profession or "").strip(),
                loc=loc,
                prefer_chinese=prefer_chinese,
            )
        ):
            used.add(current_name)
            normalized.append(profile)
            continue

        loc_name = str(getattr(loc, "name", "") or "").strip() or profile.starting_location or "loc"
        role = str(profile.profession or "").strip() or (
            "角色" if prefer_chinese else "Character"
        )
        base = _procedural_npc_name(
            seed=f"{profile.npc_id}:{loc_name}:{role}:{len(used)}",
            loc_name=loc_name,
            profession=role,
            prefer_chinese=prefer_chinese,
        )
        count = per_loc_counter.get(profile.starting_location, 0) + 1
        candidate = base if base not in used else (f"{base}{count}" if prefer_chinese else f"{base} {count}")
        while candidate in used:
            count += 1
            candidate = f"{base}{count}" if prefer_chinese else f"{base} {count}"
        per_loc_counter[profile.starting_location] = count
        profile.name = candidate
        used.add(candidate)
        normalized.append(profile)

    return normalized


def _npc_name_needs_rewrite(name: str, *, profession: str, loc: Any, prefer_chinese: bool) -> bool:
    text = str(name or "").strip()
    if not text:
        return True
    if _NPC_PLACEHOLDER_ZH.search(text) or _NPC_PLACEHOLDER_EN.search(text):
        return True
    loc_name = str(getattr(loc, "name", "") or "").strip()
    prof = str(profession or "").strip()
    if prof and text == prof:
        return True
    if loc_name and text.count(loc_name) >= 2:
        return True
    if prof and loc_name and text == f"{loc_name}{prof}":
        return True
    if prefer_chinese and any(token in text for token in ("工作人员", "居民", "市民")) and len(text) >= 6:
        return True
    if (not prefer_chinese) and any(token in text.lower() for token in ("staff", "resident", "citizen")) and len(text) >= 12:
        return True
    return False


def _procedural_npc_name(*, seed: str, loc_name: str, profession: str, prefer_chinese: bool) -> str:
    score = 0
    for idx, ch in enumerate(seed):
        score += (idx + 1) * ord(ch)
    if prefer_chinese:
        loc_chars = [ch for ch in str(loc_name or "") if _contains_cjk(ch)]
        prof_chars = [ch for ch in str(profession or "") if _contains_cjk(ch)]
        fallback = list("安若清宁泽岚川言")
        pool = loc_chars + prof_chars + fallback
        if not pool:
            pool = fallback
        first = pool[score % len(pool)]
        second = pool[(score // 3 + 1) % len(pool)]
        if first == second:
            second = pool[(score // 5 + 2) % len(pool)]
        return f"{first}{second}"
    consonants = ["b", "d", "f", "g", "k", "l", "m", "n", "r", "s", "t", "v"]
    vowels = ["a", "e", "i", "o", "u", "y"]
    c1 = consonants[score % len(consonants)]
    v1 = vowels[(score // 2) % len(vowels)]
    c2 = consonants[(score // 5 + 3) % len(consonants)]
    v2 = vowels[(score // 7 + 1) % len(vowels)]
    return f"{c1}{v1}{c2}{v2}".capitalize()
