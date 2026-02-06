"""World generation pipeline (Milestone 4)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Any
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

ITEM_ALIAS_TO_ZH: dict[str, str] = {
    "healing_herb": "疗伤草",
    "hardwood": "硬木",
    "ancient_shard": "远古碎片",
    "iron_rivet": "铁铆钉",
    "crest_fragment": "徽印碎片",
    "ration": "口粮",
    "cloth_strip": "布条",
    "strange_trinket": "奇异小物",
    "dragon_scale": "龙鳞",
    "moon_herb": "月光草",
    "moonlight_herb": "月光草",
    "ancient_ore": "古铁矿",
    "royal_leaf": "王庭药叶",
    "field_sample": "探险样本",
    "sacred_amulet": "神圣护符",
    "dragon_scale_shield": "龙鳞盾",
    "royal_writ": "王国军令",
    "oath_signet": "古誓纹章",
    "phoenix_feather": "凤凰羽",
    "ancient_runeblade": "远古符文刃",
    "dragonheart_amulet": "龙心护符",
    "dragon_heart_amulet": "龙心护符",
    "side_reward_1": "支线凭证1",
    "side_reward_2": "支线凭证2",
    "side_reward_3": "支线凭证3",
}

NPC_NAME_POOL_ZH: list[str] = [
    "莉娜",
    "凯洛",
    "艾文",
    "塔莎",
    "罗恩",
    "米娅",
    "维克",
    "艾拉",
    "萨缪尔",
    "诺亚",
    "赛琳",
    "伊桑",
    "索菲",
    "亚伦",
    "露西",
    "格雷",
    "菲奥娜",
    "哈罗德",
    "艾米",
    "布兰",
    "柯琳",
    "马库斯",
    "薇拉",
    "尼克",
    "芙蕾雅",
    "莱恩",
    "卡萝",
    "尤金",
    "阿黛尔",
    "塞德里克",
]

NPC_NAME_POOL_EN: list[str] = [
    "Lina",
    "Kael",
    "Evan",
    "Tasha",
    "Rowan",
    "Mia",
    "Vik",
    "Ella",
    "Samuel",
    "Noah",
    "Selene",
    "Ethan",
    "Sophie",
    "Aaron",
    "Lucy",
    "Gray",
    "Fiona",
    "Harold",
    "Amy",
    "Bran",
    "Corin",
    "Marcus",
    "Vera",
    "Nick",
    "Freya",
    "Ryan",
    "Carol",
    "Eugene",
    "Adele",
    "Cedric",
]


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
        description="World specification for a medieval RPG world.",
    )
    system = (
        "You are a world generation engine for an RPG setting. "
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
        "Generate one concrete main quest (main_quest) with a clear objective. "
        "Generate 1-3 side_quests tied to existing NPCs (e.g. recruit allies, gather supplies), "
        "and each side quest should define reward_items. "
        "Main quest required_items should depend on side quest reward_items (main line unlocked via side quests). "
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
    for q in world.side_quests:
        text_parts.extend([q.title or "", q.description or "", q.objective or "", q.reward_hint or ""])

    text = " ".join(text_parts).strip()
    if not text:
        return True
    has_cjk = _contains_cjk(text)
    if target_language == "zh":
        return has_cjk
    return not has_cjk


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
    updated.npcs = _ensure_npc_density(updated, prefer_chinese=prefer_chinese)
    updated.npcs = _ensure_unique_npc_names(updated.npcs, prefer_chinese=prefer_chinese)
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
            updated.main_quest.objective = "完成支线并收集关键战利品：" + "，".join(
                [f"{k} x{v}" for k, v in main_required.items()]
            )
            updated.main_quest.description = (
                "主线推进条件：先完成各支线任务，获得任务奖励物品，再返回推进终章。"
            )
        else:
            updated.main_quest.objective = "Finish side quests and collect key rewards: " + ", ".join(
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


def _localize_item_name(name: str, *, prefer_chinese: bool) -> str:
    text = str(name or "").strip()
    if not text:
        return "任务物资" if prefer_chinese else "quest_item"
    if prefer_chinese:
        if _contains_cjk(text):
            return text
        token = _normalize_item_token(text)
        if token in ITEM_ALIAS_TO_ZH:
            return ITEM_ALIAS_TO_ZH[token]
        return "任务物资"
    return text


def _localize_item_map(
    items: dict[str, int],
    *,
    prefer_chinese: bool,
) -> tuple[dict[str, int], dict[str, str]]:
    localized: dict[str, int] = {}
    renamed: dict[str, str] = {}
    unknown_token_to_name: dict[str, str] = {}
    unknown_index = 1

    for raw_name, raw_count in (items or {}).items():
        try:
            count = int(raw_count)
        except Exception:
            continue
        if count <= 0:
            continue

        name = str(raw_name or "").strip()
        token = _normalize_item_token(name)
        target = _localize_item_name(name, prefer_chinese=prefer_chinese)

        if prefer_chinese and target == "任务物资":
            if token in unknown_token_to_name:
                target = unknown_token_to_name[token]
            else:
                candidate = "任务物资"
                while candidate in localized:
                    unknown_index += 1
                    candidate = f"任务物资{unknown_index}"
                unknown_token_to_name[token] = candidate
                target = candidate
                unknown_index += 1

        localized[target] = int(localized.get(target, 0)) + count
        if name and name != target:
            renamed[name] = target
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
            title = " ".join(part.capitalize() for part in token.split("_") if part)
            if title:
                variants.add(title)
        for variant in sorted(variants, key=len, reverse=True):
            if not variant:
                continue
            updated = re.sub(rf"(?<![\w]){re.escape(variant)}(?![\w])", dst, updated, flags=re.IGNORECASE)
    return updated


def _normalize_side_quests(world: WorldSpec, *, prefer_chinese: bool) -> list[QuestSpec]:
    target_count = min(3, max(1, len(world.locations)))
    locations = [loc for loc in world.locations if loc.location_id != world.starting_location] or list(world.locations)
    npcs = list(world.npcs)
    rewards_pool = (
        ["神圣护符", "龙鳞盾", "王国军令", "古誓纹章"]
        if prefer_chinese
        else ["sacred_amulet", "dragon_scale_shield", "royal_writ", "oath_signet"]
    )
    normalized: list[QuestSpec] = []
    used_ids = set()
    used_rewards = set()

    def _pick_reward(idx: int) -> dict[str, int]:
        for item in rewards_pool:
            if item in used_rewards:
                continue
            used_rewards.add(item)
            return {item: 1}
        fallback = f"side_reward_{idx+1}" if not prefer_chinese else f"支线凭证{idx+1}"
        return {fallback: 1}

    for idx, quest in enumerate(world.side_quests):
        if len(normalized) >= target_count:
            break
        loc = locations[idx % len(locations)] if locations else None
        npc = npcs[idx % len(npcs)] if npcs else None
        quest_id = quest.quest_id or f"side_{idx+1}"
        if quest_id in used_ids:
            quest_id = f"{quest_id}_{idx+1}"
        used_ids.add(quest_id)
        required_items = dict(quest.required_items or {})
        if not required_items and loc:
            required_items = _default_required_items_for_location(loc, prefer_chinese)
        required_items, required_replacements = _localize_item_map(required_items, prefer_chinese=prefer_chinese)
        reward_items = dict(quest.reward_items or {})
        if not reward_items:
            reward_items = _pick_reward(idx)
        else:
            for item in reward_items.keys():
                used_rewards.add(item)
        reward_items, reward_replacements = _localize_item_map(reward_items, prefer_chinese=prefer_chinese)
        text_replacements = dict(required_replacements)
        text_replacements.update(reward_replacements)
        objective_text = quest.objective or _default_side_objective(required_items, loc, prefer_chinese)
        description_text = quest.description or _default_side_description(loc, prefer_chinese)
        reward_hint_text = quest.reward_hint or _default_reward_hint(reward_items, prefer_chinese)
        objective_text = _replace_item_mentions(objective_text, text_replacements)
        description_text = _replace_item_mentions(description_text, text_replacements)
        reward_hint_text = _replace_item_mentions(reward_hint_text, text_replacements)
        normalized.append(
            QuestSpec(
                quest_id=quest_id,
                title=quest.title or (_default_side_title(idx, prefer_chinese)),
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
        required_items = _default_required_items_for_location(loc, prefer_chinese)
        reward_items = _pick_reward(idx)
        required_items, _ = _localize_item_map(required_items, prefer_chinese=prefer_chinese)
        reward_items, _ = _localize_item_map(reward_items, prefer_chinese=prefer_chinese)
        quest_id = f"side_{loc.location_id}_{idx+1}"
        if quest_id in used_ids:
            quest_id = f"{quest_id}_{len(used_ids)+1}"
        used_ids.add(quest_id)
        normalized.append(
            QuestSpec(
                quest_id=quest_id,
                title=_default_side_title(idx, prefer_chinese),
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


def _default_required_items_for_location(loc: Any, prefer_chinese: bool) -> dict[str, int]:
    kind = (loc.kind or "").lower()
    if prefer_chinese:
        mapping = {
            "forest": {"月光草": 3},
            "dungeon": {"古铁矿": 2},
            "bridge": {"铁铆钉": 2},
            "castle": {"王庭药叶": 2},
            "town": {"口粮": 2},
            "shop": {"布条": 2},
        }
        return mapping.get(kind, {"探险样本": 2})
    mapping = {
        "forest": {"moon_herb": 3},
        "dungeon": {"ancient_ore": 2},
        "bridge": {"iron_rivet": 2},
        "castle": {"royal_leaf": 2},
        "town": {"ration": 2},
        "shop": {"cloth_strip": 2},
    }
    return mapping.get(kind, {"field_sample": 2})


def _default_side_title(idx: int, prefer_chinese: bool) -> str:
    if prefer_chinese:
        names = ["失落药草", "锻造前哨", "边境护送"]
        return names[idx % len(names)]
    names = ["Lost Herbs", "Forge Outpost", "Border Escort"]
    return names[idx % len(names)]


def _default_side_description(loc: Any, prefer_chinese: bool) -> str:
    if prefer_chinese:
        return f"与任务 NPC 一同前往{loc.name}并完成材料采集。"
    return f"Travel with the quest NPC to {loc.name} and gather the required materials."


def _default_side_objective(required_items: dict[str, int], loc: Any, prefer_chinese: bool) -> str:
    req = "，".join([f"{name} x{count}" for name, count in required_items.items()])
    if prefer_chinese:
        return f"在{loc.name}与任务 NPC 同行时收集：{req}"
    return f"Collect at {loc.name} while accompanied by the quest NPC: {req}"


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
            # Fallback to starting location if invalid.
            npc.starting_location = world.starting_location
            by_loc.setdefault(world.starting_location, []).append(npc)

    target_min = 2
    target_max = 5
    name_pool = NPC_NAME_POOL_ZH if prefer_chinese else NPC_NAME_POOL_EN
    profession_pool = (
        ["铁匠", "药师", "猎人", "商贩", "卫兵", "学者", "旅店老板", "向导"]
        if prefer_chinese
        else ["Blacksmith", "Herbalist", "Hunter", "Merchant", "Guard", "Scholar", "Innkeeper", "Guide"]
    )
    trait_sets = (
        [
            (["热情", "果断"], ["协助旅人"], 0.85, 0.2, 0.6, 3, "直爽友好"),
            (["谨慎", "多疑"], ["守护家园"], 0.35, 0.75, 0.3, -1, "冷淡保守"),
            (["冷静", "理性"], ["收集情报"], 0.55, 0.45, 0.5, 1, "简洁克制"),
            (["勇敢", "冲动"], ["猎杀威胁"], 0.7, 0.35, 0.8, 0, "急躁直接"),
        ]
        if prefer_chinese
        else [
            (["warm", "decisive"], ["assist travelers"], 0.85, 0.2, 0.6, 3, "friendly and direct"),
            (["cautious", "suspicious"], ["protect the village"], 0.35, 0.75, 0.3, -1, "cold and guarded"),
            (["calm", "rational"], ["gather intel"], 0.55, 0.45, 0.5, 1, "concise and restrained"),
            (["brave", "impulsive"], ["hunt threats"], 0.7, 0.35, 0.8, 0, "impatient and blunt"),
        ]
    )

    used_ids = {npc.npc_id for npc in npcs}
    used_names = {npc.name for npc in npcs}
    name_index = 0
    generated_index = 1
    fallback_name_index = 1

    def next_name() -> str:
        nonlocal name_index, fallback_name_index
        while True:
            if name_index < len(name_pool):
                candidate = name_pool[name_index]
                name_index += 1
                if candidate in used_names:
                    continue
                used_names.add(candidate)
                return candidate
            fallback = ("旅者" if prefer_chinese else "Wanderer") + str(fallback_name_index)
            fallback_name_index += 1
            if fallback in used_names:
                continue
            used_names.add(fallback)
            return fallback

    def next_id() -> str:
        nonlocal generated_index
        while True:
            candidate = f"npc_auto_{generated_index:03d}"
            generated_index += 1
            if candidate not in used_ids:
                used_ids.add(candidate)
                return candidate

    # Re-balance excessive density first.
    overflow: list[NPCProfile] = []
    for loc in locations:
        current = by_loc.get(loc.location_id, [])
        if len(current) <= target_max:
            continue
        keep = current[:target_max]
        extra = current[target_max:]
        by_loc[loc.location_id] = keep
        overflow.extend(extra)
    if overflow:
        for npc in overflow:
            # Move extras to locations with available room.
            candidates = sorted(
                locations,
                key=lambda l: len(by_loc.get(l.location_id, [])),
            )
            placed = False
            for loc in candidates:
                current = by_loc.get(loc.location_id, [])
                if len(current) >= target_max:
                    continue
                npc.starting_location = loc.location_id
                current.append(npc)
                by_loc[loc.location_id] = current
                placed = True
                break
            if not placed:
                # If all full, keep original location.
                loc_id = npc.starting_location
                by_loc.setdefault(loc_id, []).append(npc)

    for loc in locations:
        current = by_loc.get(loc.location_id, [])
        while len(current) < target_min:
            idx = len(npcs)
            traits, goals, obedience, stubborn, risk, disp, refusal = trait_sets[idx % len(trait_sets)]
            profile = NPCProfile(
                npc_id=next_id(),
                name=next_name(),
                profession=profession_pool[idx % len(profession_pool)],
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
    # Flatten by current starting_location order.
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


def _ensure_unique_npc_names(npcs: list[NPCProfile], *, prefer_chinese: bool) -> list[NPCProfile]:
    pool = NPC_NAME_POOL_ZH if prefer_chinese else NPC_NAME_POOL_EN
    used: set[str] = set()
    pool_index = 0
    fallback_index = 1
    normalized: list[NPCProfile] = []

    def next_name() -> str:
        nonlocal pool_index, fallback_index
        while True:
            while pool_index < len(pool):
                candidate = pool[pool_index]
                pool_index += 1
                if candidate in used:
                    continue
                used.add(candidate)
                return candidate
            candidate = ("旅者" if prefer_chinese else "Wanderer") + str(fallback_index)
            fallback_index += 1
            if candidate in used:
                continue
            used.add(candidate)
            return candidate

    for npc in npcs:
        profile = npc.model_copy(deep=True)
        current_name = str(profile.name or "").strip()
        if current_name and current_name not in used:
            used.add(current_name)
        else:
            profile.name = next_name()
        normalized.append(profile)
    return normalized
