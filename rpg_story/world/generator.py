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
    theme = _infer_world_theme(updated)
    updated.npcs = _normalize_npc_professions_by_theme(updated, theme=theme, prefer_chinese=prefer_chinese)
    updated.npcs = _ensure_npc_density(updated, prefer_chinese=prefer_chinese, theme=theme)
    updated.npcs = _ensure_unique_npc_names(updated.npcs, prefer_chinese=prefer_chinese)
    updated.side_quests = _normalize_side_quests(updated, prefer_chinese=prefer_chinese, theme=theme)
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


THEME_KEYWORDS_ZH: Dict[str, List[str]] = {
    "campus": ["校园", "学院", "大学", "高中", "教室", "图书馆", "宿舍", "社团", "学生会", "恋爱", "告白"],
    "sci_fi": ["太空", "星舰", "机甲", "人工智能", "赛博", "外星", "量子", "空间站"],
    "modern": ["都市", "公司", "办公室", "地铁", "公寓", "商场", "咖啡馆"],
    "fantasy": ["王国", "骑士", "巨龙", "魔法", "神殿", "地牢", "冒险者", "勇者"],
}

THEME_KEYWORDS_EN: Dict[str, List[str]] = {
    "campus": ["campus", "school", "college", "university", "classroom", "library", "dorm", "club", "romance"],
    "sci_fi": ["spaceship", "space station", "mecha", "android", "cyber", "quantum", "alien", "galaxy"],
    "modern": ["city", "office", "apartment", "subway", "mall", "cafe", "corporate"],
    "fantasy": ["kingdom", "dragon", "magic", "temple", "dungeon", "adventurer", "knight"],
}

TOKEN_TO_ZH: Dict[str, str] = {
    "star": "星",
    "iron": "铁",
    "ore": "矿",
    "dragon": "龙",
    "scale": "鳞",
    "fragment": "碎片",
    "letter": "信",
    "love": "情书",
    "flower": "花",
    "bouquet": "花束",
    "badge": "徽章",
    "ticket": "票",
    "book": "书",
    "library": "图书",
    "notebook": "笔记本",
    "note": "笔记",
    "report": "报告",
    "camera": "相机",
    "photo": "照片",
    "coffee": "咖啡",
    "milk": "牛奶",
    "tea": "茶",
    "coupon": "券",
    "key": "钥匙",
    "card": "卡",
    "sticker": "贴纸",
    "gift": "礼物",
    "ring": "戒指",
    "bracelet": "手链",
    "amulet": "护符",
    "chip": "芯片",
    "battery": "电池",
    "module": "模块",
    "data": "数据",
    "sample": "样本",
}

ZH_SURNAMES: List[str] = [
    "林", "陈", "周", "赵", "刘", "黄", "吴", "徐", "孙", "马", "何", "郭", "罗", "朱", "梁", "谢",
    "宋", "郑", "唐", "韩",
]

ZH_GIVEN: List[str] = [
    "晨曦", "子涵", "若宁", "思远", "景行", "以安", "知夏", "清越", "亦辰", "嘉言",
    "书遥", "可心", "雨桐", "安然", "沐阳", "宛宁", "泽川", "予墨", "舒然", "星河",
]

EN_FIRST: List[str] = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Avery", "Quinn", "Cameron", "Parker",
    "Jamie", "Rowan", "Skyler", "Drew", "Reese", "Robin", "Elliot", "Harper", "Hayden", "Blake",
]

EN_LAST: List[str] = [
    "Lin", "Chen", "Morris", "Reed", "Bennett", "Walker", "Carter", "Hayes", "Perry", "Brooks",
    "Turner", "Foster", "Campbell", "Ward", "Price", "Stone", "Bell", "Shaw", "Parker", "Russell",
]


def _infer_world_theme(world: WorldSpec) -> str:
    text_parts = [world.title or "", world.starting_hook or "", world.initial_quest or "", world.world_bible.tone or ""]
    for loc in world.locations:
        text_parts.append(loc.name or "")
        text_parts.append(loc.kind or "")
        text_parts.extend(loc.tags or [])
    for npc in world.npcs:
        text_parts.append(npc.profession or "")
        text_parts.extend(npc.goals or [])
    joined = " ".join(str(part) for part in text_parts if part)
    lowered = joined.lower()
    for theme, keys in THEME_KEYWORDS_ZH.items():
        if any(key in joined for key in keys):
            return theme
    for theme, keys in THEME_KEYWORDS_EN.items():
        if any(key in lowered for key in keys):
            return theme
    tech = getattr(world.world_bible, "tech_level", "medieval") or "medieval"
    if tech == "sci-fi":
        return "sci_fi"
    if tech == "modern":
        return "modern"
    return "fantasy"


def _theme_reward_pool(theme: str, *, prefer_chinese: bool) -> list[str]:
    if theme == "campus":
        return ["晚会入场券", "告白花束", "纪念徽章", "手写情书"] if prefer_chinese else [
            "prom_ticket",
            "confession_bouquet",
            "campus_badge",
            "handwritten_letter",
        ]
    if theme == "sci_fi":
        return ["相位芯片", "量子电池", "权限密钥", "舰桥通行证"] if prefer_chinese else [
            "phase_chip",
            "quantum_battery",
            "access_key",
            "bridge_pass",
        ]
    if theme == "modern":
        return ["推荐信", "演出门票", "项目通行证", "城市徽章"] if prefer_chinese else [
            "recommendation_letter",
            "show_ticket",
            "project_pass",
            "city_badge",
        ]
    return ["神圣护符", "龙鳞盾", "王国军令", "古誓纹章"] if prefer_chinese else [
        "sacred_amulet",
        "dragon_scale_shield",
        "royal_writ",
        "oath_signet",
    ]


def _theme_side_titles(theme: str, *, prefer_chinese: bool) -> list[str]:
    if theme == "campus":
        return ["情书代笔", "文化节徽章", "花束的秘密"] if prefer_chinese else [
            "Ghostwritten Letter",
            "Festival Badge",
            "Secret Bouquet",
        ]
    if theme == "sci_fi":
        return ["修复中继站", "追踪失联信号", "舰桥权限申请"] if prefer_chinese else [
            "Relay Repair",
            "Lost Signal Hunt",
            "Bridge Access",
        ]
    if theme == "modern":
        return ["街区委托", "展会筹备", "夜间快递"] if prefer_chinese else [
            "District Errand",
            "Expo Setup",
            "Night Delivery",
        ]
    return ["失落药草", "锻造前哨", "边境护送"] if prefer_chinese else [
        "Lost Herbs",
        "Forge Outpost",
        "Border Escort",
    ]


def _location_keyword_bucket(loc: Any) -> str:
    text = " ".join(
        [
            str(getattr(loc, "kind", "") or ""),
            str(getattr(loc, "name", "") or ""),
            " ".join(getattr(loc, "tags", []) or []),
        ]
    ).lower()
    if any(key in text for key in ["library", "图书", "书馆"]):
        return "library"
    if any(key in text for key in ["class", "classroom", "教室", "课堂"]):
        return "classroom"
    if any(key in text for key in ["dorm", "宿舍"]):
        return "dorm"
    if any(key in text for key in ["cafeteria", "canteen", "食堂", "餐厅", "咖啡"]):
        return "canteen"
    if any(key in text for key in ["club", "社团", "活动室"]):
        return "club"
    if any(key in text for key in ["garden", "park", "花园", "操场", "广场"]):
        return "square"
    if any(key in text for key in ["lab", "实验", "研究", "station", "控制室"]):
        return "lab"
    if any(key in text for key in ["town", "village", "城", "镇", "村"]):
        return "town"
    if any(key in text for key in ["forest", "woods", "林"]):
        return "forest"
    if any(key in text for key in ["dungeon", "ruin", "遗迹", "地牢"]):
        return "dungeon"
    if any(key in text for key in ["shop", "market", "商店", "集市"]):
        return "shop"
    return "generic"


def _deterministic_index(seed: str, size: int) -> int:
    if size <= 0:
        return 0
    return sum(ord(ch) for ch in seed) % size


def _resource_candidates(theme: str, bucket: str, *, prefer_chinese: bool) -> list[dict[str, int]]:
    if theme == "campus":
        zh = {
            "library": [{"参考书": 2, "借书卡": 1}, {"文献复印件": 2, "书签": 1}],
            "classroom": [{"课堂笔记": 2, "作业纸": 2}, {"实验记录": 2, "演讲提纲": 1}],
            "dorm": [{"宿舍钥匙": 1, "便签": 2}, {"合照": 1, "零食券": 2}],
            "canteen": [{"奶茶券": 2, "点心": 1}, {"咖啡券": 2, "餐券": 1}],
            "club": [{"社团徽章": 2, "活动手册": 1}, {"应援贴纸": 2, "节目单": 1}],
            "square": [{"樱花": 3, "告白卡片": 1}, {"花束包装纸": 2, "丝带": 1}],
            "generic": [{"校园传单": 2, "纪念贴纸": 2}, {"备忘录": 2, "钥匙扣": 1}],
        }
        en = {
            "library": [{"reference_book": 2, "library_card": 1}, {"copied_notes": 2, "bookmark": 1}],
            "classroom": [{"class_notes": 2, "worksheet": 2}, {"lab_report": 2, "speech_outline": 1}],
            "dorm": [{"dorm_key": 1, "sticky_note": 2}, {"group_photo": 1, "snack_coupon": 2}],
            "canteen": [{"milk_tea_coupon": 2, "pastry": 1}, {"coffee_coupon": 2, "meal_ticket": 1}],
            "club": [{"club_badge": 2, "event_booklet": 1}, {"cheer_sticker": 2, "program_sheet": 1}],
            "square": [{"cherry_blossom": 3, "confession_card": 1}, {"bouquet_wrap": 2, "ribbon": 1}],
            "generic": [{"campus_flyer": 2, "souvenir_sticker": 2}, {"memo_note": 2, "keychain": 1}],
        }
        table = zh if prefer_chinese else en
        return table.get(bucket, table["generic"])
    if theme == "sci_fi":
        zh = {
            "lab": [{"相位芯片": 2, "校准模块": 1}, {"量子电池": 2, "冷却液": 1}],
            "town": [{"数据终端钥匙": 1, "维修工具": 2}],
            "generic": [{"合金零件": 2, "信号样本": 1}, {"能源晶片": 2, "权限卡": 1}],
        }
        en = {
            "lab": [{"phase_chip": 2, "calibration_module": 1}, {"quantum_battery": 2, "coolant": 1}],
            "town": [{"terminal_key": 1, "repair_tool": 2}],
            "generic": [{"alloy_part": 2, "signal_sample": 1}, {"energy_chip": 2, "access_card": 1}],
        }
        table = zh if prefer_chinese else en
        return table.get(bucket, table["generic"])
    if theme == "modern":
        zh = {
            "town": [{"商圈通行证": 1, "宣传单": 2}, {"包裹单": 2, "街区地图": 1}],
            "shop": [{"店铺印章": 2, "折扣券": 1}, {"收据": 2, "快递标签": 1}],
            "square": [{"演出门票": 2, "应援贴纸": 1}],
            "generic": [{"资料夹": 2, "工作证": 1}, {"采访录音": 1, "照片": 2}],
        }
        en = {
            "town": [{"district_pass": 1, "flyer": 2}, {"parcel_form": 2, "street_map": 1}],
            "shop": [{"store_stamp": 2, "discount_coupon": 1}, {"receipt": 2, "shipping_label": 1}],
            "square": [{"show_ticket": 2, "cheer_sticker": 1}],
            "generic": [{"folder": 2, "work_badge": 1}, {"interview_record": 1, "photo": 2}],
        }
        table = zh if prefer_chinese else en
        return table.get(bucket, table["generic"])
    zh = {
        "forest": [{"月光草": 3}, {"疗伤草": 2, "硬木": 2}],
        "dungeon": [{"古铁矿": 2}, {"远古碎片": 2, "铁铆钉": 1}],
        "town": [{"口粮": 2}, {"布条": 2, "口粮": 1}],
        "shop": [{"布条": 2}, {"奇异小物": 2}],
        "generic": [{"探险样本": 2}, {"奇异小物": 2}],
    }
    en = {
        "forest": [{"moon_herb": 3}, {"healing_herb": 2, "hardwood": 2}],
        "dungeon": [{"ancient_ore": 2}, {"ancient_shard": 2, "iron_rivet": 1}],
        "town": [{"ration": 2}, {"cloth_strip": 2, "ration": 1}],
        "shop": [{"cloth_strip": 2}, {"strange_trinket": 2}],
        "generic": [{"field_sample": 2}, {"strange_trinket": 2}],
    }
    table = zh if prefer_chinese else en
    return table.get(bucket, table["generic"])


def suggest_location_resource_template(world: WorldSpec, loc: Any, *, prefer_chinese: bool) -> dict[str, int]:
    theme = _infer_world_theme(world)
    bucket = _location_keyword_bucket(loc)
    candidates = _resource_candidates(theme, bucket, prefer_chinese=prefer_chinese)
    idx = _deterministic_index(f"{getattr(loc, 'location_id', '')}:{theme}", len(candidates))
    return {str(k): max(1, int(v)) for k, v in candidates[idx].items()}


def _normalize_item_token(value: str) -> str:
    token = re.sub(r"[\s\-]+", "_", str(value or "").strip().lower())
    token = re.sub(r"[^0-9a-z_\u4e00-\u9fff]", "", token)
    token = re.sub(r"_+", "_", token)
    return token.strip("_")


def _token_to_zh_label(token: str) -> str | None:
    parts = [part for part in str(token or "").split("_") if part]
    if not parts:
        return None
    mapped = [TOKEN_TO_ZH.get(part) for part in parts]
    if all(mapped):
        joined = "".join(mapped)
        # Keep some readable suffixes for common nouns.
        if joined.endswith(("信", "票", "卡", "章", "钥匙", "芯片", "电池", "样本", "书")):
            return joined
        return joined + "物资"
    return None


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
        guess = _token_to_zh_label(token)
        if guess:
            return guess
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


def _normalize_side_quests(world: WorldSpec, *, prefer_chinese: bool, theme: str) -> list[QuestSpec]:
    target_count = min(3, max(1, len(world.locations)))
    locations = [loc for loc in world.locations if loc.location_id != world.starting_location] or list(world.locations)
    npcs = list(world.npcs)
    rewards_pool = _theme_reward_pool(theme, prefer_chinese=prefer_chinese)
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
            required_items = _default_required_items_for_location(loc, prefer_chinese, theme=theme, variant=idx)
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
        objective_text = _ensure_side_objective_consistency(
            objective_text,
            required_items,
            loc,
            prefer_chinese=prefer_chinese,
        )
        title_text = _ensure_side_title_consistency(
            quest.title or (_default_side_title(idx, prefer_chinese, theme=theme)),
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
        required_items = _default_required_items_for_location(loc, prefer_chinese, theme=theme, variant=idx)
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
                title=_default_side_title(idx, prefer_chinese, theme=theme),
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
    theme: str,
    variant: int = 0,
) -> dict[str, int]:
    bucket = _location_keyword_bucket(loc)
    candidates = _resource_candidates(theme, bucket, prefer_chinese=prefer_chinese)
    idx = _deterministic_index(f"{getattr(loc, 'location_id', '')}:{variant}:{theme}", len(candidates))
    selected = candidates[idx]
    return {str(name): max(1, int(count)) for name, count in selected.items()}


def _default_side_title(idx: int, prefer_chinese: bool, *, theme: str) -> str:
    names = _theme_side_titles(theme, prefer_chinese=prefer_chinese)
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
        if any(token in text for token in ("寻找", "收集", "采集", "获取")):
            return f"收集{first_item}"
        return text
    lower = text.lower()
    if any(token in lower for token in ("find", "collect", "gather", "obtain", "fetch")):
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


def _theme_profession_pool(theme: str, *, prefer_chinese: bool) -> list[str]:
    if theme == "campus":
        return ["学生", "班长", "图书管理员", "社团干部", "辅导员", "校医", "摄影社成员", "学生会干事"] if prefer_chinese else [
            "Student",
            "Class Rep",
            "Librarian",
            "Club Officer",
            "Counselor",
            "School Nurse",
            "Photography Member",
            "Student Council Member",
        ]
    if theme == "sci_fi":
        return ["工程师", "导航员", "通讯官", "科研员", "安保员", "后勤官", "维修技师", "舰桥值班员"] if prefer_chinese else [
            "Engineer",
            "Navigator",
            "Comms Officer",
            "Researcher",
            "Security Officer",
            "Logistics Officer",
            "Maintenance Tech",
            "Bridge Watch",
        ]
    if theme == "modern":
        return ["店员", "记者", "快递员", "策展人", "社区志愿者", "程序员", "摄影师", "策划"] if prefer_chinese else [
            "Clerk",
            "Reporter",
            "Courier",
            "Curator",
            "Volunteer",
            "Programmer",
            "Photographer",
            "Planner",
        ]
    return ["铁匠", "药师", "猎人", "商贩", "卫兵", "学者", "旅店老板", "向导"] if prefer_chinese else [
        "Blacksmith",
        "Herbalist",
        "Hunter",
        "Merchant",
        "Guard",
        "Scholar",
        "Innkeeper",
        "Guide",
    ]


def _fallback_name_candidates(prefer_chinese: bool) -> list[str]:
    if prefer_chinese:
        return [f"{surname}{given}" for surname in ZH_SURNAMES for given in ZH_GIVEN]
    return [f"{first} {last}" for first in EN_FIRST for last in EN_LAST]


def _make_name_allocator(used_names: set[str], *, prefer_chinese: bool):
    candidates = _fallback_name_candidates(prefer_chinese)
    index = 0
    suffix = 1

    def next_name() -> str:
        nonlocal index, suffix
        while index < len(candidates):
            candidate = candidates[index]
            index += 1
            if candidate in used_names:
                continue
            used_names.add(candidate)
            return candidate
        while True:
            candidate = f"角色{suffix}" if prefer_chinese else f"Character {suffix}"
            suffix += 1
            if candidate in used_names:
                continue
            used_names.add(candidate)
            return candidate

    return next_name


def _normalize_npc_professions_by_theme(world: WorldSpec, *, theme: str, prefer_chinese: bool) -> list[NPCProfile]:
    npcs = [npc.model_copy(deep=True) for npc in world.npcs]
    if not npcs:
        return npcs
    if theme not in {"campus", "modern", "sci_fi"}:
        return npcs
    if prefer_chinese:
        if theme == "campus":
            replacements = {"铁匠": "校工", "药师": "校医", "猎人": "社团外联", "卫兵": "保安", "商贩": "小卖部店员"}
        elif theme == "modern":
            replacements = {"铁匠": "维修师", "药师": "药店店员", "猎人": "调查员", "卫兵": "保安"}
        else:
            replacements = {"铁匠": "工程师", "药师": "医疗官", "猎人": "侦察员", "卫兵": "安保员"}
    else:
        if theme == "campus":
            replacements = {
                "blacksmith": "Campus Technician",
                "herbalist": "School Nurse",
                "hunter": "Club Runner",
                "guard": "Security Staff",
                "merchant": "Campus Store Clerk",
            }
        elif theme == "modern":
            replacements = {
                "blacksmith": "Mechanic",
                "herbalist": "Pharmacy Clerk",
                "hunter": "Investigator",
                "guard": "Security Staff",
            }
        else:
            replacements = {
                "blacksmith": "Systems Engineer",
                "herbalist": "Medical Officer",
                "hunter": "Recon Specialist",
                "guard": "Security Officer",
            }

    fixed: list[NPCProfile] = []
    for npc in npcs:
        profile = npc.model_copy(deep=True)
        profession = str(profile.profession or "").strip()
        if prefer_chinese:
            if profession in replacements:
                profile.profession = replacements[profession]
        else:
            key = profession.lower()
            if key in replacements:
                profile.profession = replacements[key]
        fixed.append(profile)
    return fixed


def _theme_professions_for_location(theme: str, bucket: str, *, prefer_chinese: bool) -> list[str]:
    if theme == "campus":
        zh = {
            "library": ["图书管理员", "学生", "文学社成员"],
            "classroom": ["学生", "班长", "助教"],
            "dorm": ["学生", "宿管", "室友"],
            "canteen": ["后勤老师", "学生", "社团干部"],
            "club": ["社团干部", "学生会干事", "学生"],
            "square": ["学生", "摄影社成员", "学生会干事"],
            "generic": ["学生", "辅导员", "学生会干事"],
        }
        en = {
            "library": ["Librarian", "Student", "Literature Club Member"],
            "classroom": ["Student", "Class Rep", "Teaching Assistant"],
            "dorm": ["Student", "Dorm Warden", "Roommate"],
            "canteen": ["Staff", "Student", "Club Officer"],
            "club": ["Club Officer", "Student Council Member", "Student"],
            "square": ["Student", "Photography Member", "Student Council Member"],
            "generic": ["Student", "Counselor", "Student Council Member"],
        }
        table = zh if prefer_chinese else en
        return table.get(bucket, table["generic"])
    return _theme_profession_pool(theme, prefer_chinese=prefer_chinese)


def _theme_trait_sets(theme: str, *, prefer_chinese: bool) -> list[tuple[list[str], list[str], float, float, float, int, str]]:
    if theme == "campus":
        return [
            (["开朗", "乐于助人"], ["帮助同学"], 0.82, 0.25, 0.45, 3, "热情直率"),
            (["害羞", "谨慎"], ["保护隐私"], 0.38, 0.72, 0.2, -1, "吞吞吐吐"),
            (["理性", "细致"], ["完成社团任务"], 0.58, 0.42, 0.35, 1, "冷静客观"),
            (["自信", "竞争心强"], ["争取荣誉"], 0.62, 0.5, 0.55, 0, "锋利直接"),
        ] if prefer_chinese else [
            (["outgoing", "helpful"], ["support classmates"], 0.82, 0.25, 0.45, 3, "warm and direct"),
            (["shy", "careful"], ["protect privacy"], 0.38, 0.72, 0.2, -1, "hesitant and guarded"),
            (["rational", "meticulous"], ["finish club tasks"], 0.58, 0.42, 0.35, 1, "calm and objective"),
            (["confident", "competitive"], ["win recognition"], 0.62, 0.5, 0.55, 0, "sharp and blunt"),
        ]
    if theme == "sci_fi":
        return [
            (["冷静", "专业"], ["维持系统稳定"], 0.78, 0.3, 0.58, 2, "简短高效"),
            (["多疑", "严谨"], ["避免事故"], 0.34, 0.74, 0.25, -1, "程序化回复"),
            (["果断", "勇敢"], ["应对危机"], 0.66, 0.4, 0.82, 1, "命令式"),
            (["好奇", "创新"], ["突破瓶颈"], 0.55, 0.46, 0.7, 1, "兴奋急促"),
        ] if prefer_chinese else [
            (["calm", "professional"], ["keep systems stable"], 0.78, 0.3, 0.58, 2, "brief and efficient"),
            (["skeptical", "rigorous"], ["prevent incidents"], 0.34, 0.74, 0.25, -1, "procedural replies"),
            (["decisive", "brave"], ["handle crises"], 0.66, 0.4, 0.82, 1, "commanding"),
            (["curious", "innovative"], ["break bottlenecks"], 0.55, 0.46, 0.7, 1, "excited and quick"),
        ]
    if theme == "modern":
        return [
            (["友善", "务实"], ["推进委托"], 0.76, 0.28, 0.48, 2, "直接明快"),
            (["谨慎", "挑剔"], ["规避风险"], 0.36, 0.68, 0.3, -1, "冷淡克制"),
            (["沉稳", "可靠"], ["维护秩序"], 0.6, 0.45, 0.5, 1, "礼貌专业"),
            (["机敏", "野心"], ["扩大影响"], 0.58, 0.5, 0.65, 0, "自信强势"),
        ] if prefer_chinese else [
            (["friendly", "pragmatic"], ["advance errands"], 0.76, 0.28, 0.48, 2, "clear and upbeat"),
            (["careful", "picky"], ["avoid risk"], 0.36, 0.68, 0.3, -1, "cold and restrained"),
            (["steady", "reliable"], ["maintain order"], 0.6, 0.45, 0.5, 1, "polite and professional"),
            (["quick", "ambitious"], ["expand influence"], 0.58, 0.5, 0.65, 0, "confident and pushy"),
        ]
    return [
        (["热情", "果断"], ["协助旅人"], 0.85, 0.2, 0.6, 3, "直爽友好"),
        (["谨慎", "多疑"], ["守护家园"], 0.35, 0.75, 0.3, -1, "冷淡保守"),
        (["冷静", "理性"], ["收集情报"], 0.55, 0.45, 0.5, 1, "简洁克制"),
        (["勇敢", "冲动"], ["猎杀威胁"], 0.7, 0.35, 0.8, 0, "急躁直接"),
    ] if prefer_chinese else [
        (["warm", "decisive"], ["assist travelers"], 0.85, 0.2, 0.6, 3, "friendly and direct"),
        (["cautious", "suspicious"], ["protect the village"], 0.35, 0.75, 0.3, -1, "cold and guarded"),
        (["calm", "rational"], ["gather intel"], 0.55, 0.45, 0.5, 1, "concise and restrained"),
        (["brave", "impulsive"], ["hunt threats"], 0.7, 0.35, 0.8, 0, "impatient and blunt"),
    ]


def _ensure_npc_density(world: WorldSpec, *, prefer_chinese: bool, theme: str) -> list[NPCProfile]:
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
    profession_pool = _theme_profession_pool(theme, prefer_chinese=prefer_chinese)
    trait_sets = _theme_trait_sets(theme, prefer_chinese=prefer_chinese)

    used_ids = {npc.npc_id for npc in npcs}
    used_names = {npc.name for npc in npcs}
    generated_index = 1
    next_name = _make_name_allocator(used_names, prefer_chinese=prefer_chinese)

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
            bucket = _location_keyword_bucket(loc)
            loc_prof_pool = _theme_professions_for_location(theme, bucket, prefer_chinese=prefer_chinese)
            profile = NPCProfile(
                npc_id=next_id(),
                name=next_name(),
                profession=loc_prof_pool[idx % len(loc_prof_pool)] if loc_prof_pool else profession_pool[idx % len(profession_pool)],
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
    used: set[str] = set()
    normalized: list[NPCProfile] = []
    next_name = _make_name_allocator(used, prefer_chinese=prefer_chinese)

    for npc in npcs:
        profile = npc.model_copy(deep=True)
        current_name = str(profile.name or "").strip()
        if current_name and current_name not in used:
            used.add(current_name)
        else:
            profile.name = next_name()
        normalized.append(profile)
    return normalized
