"""World generation pipeline (Milestone 4)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Any
from datetime import datetime, timezone
import json

from rpg_story.config import AppConfig
from rpg_story.llm.client import BaseLLMClient, make_json_schema_response_format
from rpg_story.models.world import WorldSpec, GameState
from rpg_story.persistence.store import generate_session_id, save_state, append_turn_log
from rpg_story.world.consistency import validate_world, find_anachronisms
from rpg_story.world.sanitize import sanitize_world_payload, summarize_changes


def _schema_hint() -> str:
    return (
        "WorldSpec JSON with fields: world_id, title, world_bible, locations, npcs, "
        "starting_location, starting_hook, initial_quest. "
        "world_bible: {tech_level, magic_rules, tone, anachronism_policy, taboos, do_not_mention, "
        "anachronism_blocklist}. "
        "locations: [{location_id, name, kind, description, connected_to, tags}]. "
        "npcs: [{npc_id, name, profession, traits, goals, starting_location, obedience_level, stubbornness, "
        "risk_tolerance, disposition_to_player, refusal_style}]. "
        "Constraints: tech_level in {medieval, modern, sci-fi}; "
        "obedience_level/stubbornness/risk_tolerance are floats in [0.0,1.0]; "
        "disposition_to_player is an int in [-5,5]. No extra keys (do not include schema_version)."
    )


def generate_world_spec(cfg: AppConfig, llm: BaseLLMClient, world_prompt: str) -> WorldSpec:
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
        "Generate a small coherent world with 3-8 locations and 2-8 NPCs. "
        "Ensure connected_to references valid location_ids. "
        "Use numeric types (not strings) for all numeric fields. No extra keys. "
        "Set world_bible.tech_level based on the prompt. "
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


def initialize_game_state(world: WorldSpec, session_id: str, created_at: Optional[str] = None) -> GameState:
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    npc_locations = {npc.npc_id: npc.starting_location for npc in world.npcs}
    state = GameState(
        session_id=session_id,
        created_at=created_at,
        world=world,
        player_location=world.starting_location,
        npc_locations=npc_locations,
        flags={},
        quests={},
        inventory={},
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
