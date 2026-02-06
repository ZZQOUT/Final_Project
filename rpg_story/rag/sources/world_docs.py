"""World-level document builders."""
from __future__ import annotations

from rpg_story.models.world import WorldSpec
from rpg_story.rag.types import Document, make_doc_id, normalize_metadata


def build_world_bible_doc(world: WorldSpec, session_id: str) -> Document:
    bible = world.world_bible
    parts = [
        f"Title: {world.title}",
        f"Tech level: {bible.tech_level}",
        f"Narrative language: {bible.narrative_language or 'auto'}",
        f"Magic rules: {bible.magic_rules}",
        f"Tone: {bible.tone}",
    ]
    if bible.anachronism_policy:
        parts.append(f"Anachronism policy: {bible.anachronism_policy}")
    if bible.taboos:
        parts.append("Taboos: " + ", ".join(bible.taboos))
    if bible.do_not_mention:
        parts.append("Do not mention: " + ", ".join(bible.do_not_mention))
    if bible.anachronism_blocklist:
        parts.append("Anachronism blocklist: " + ", ".join(bible.anachronism_blocklist))
    parts.append(f"Starting hook: {world.starting_hook}")
    parts.append(f"Initial quest: {world.initial_quest}")
    if world.main_quest:
        mq = world.main_quest
        parts.append(f"Main quest: {mq.title} | objective: {mq.objective}")
        if mq.required_items:
            req = ", ".join([f"{k} x{v}" for k, v in mq.required_items.items()])
            parts.append(f"Main quest required_items: {req}")
    if world.side_quests:
        quest_lines = []
        for quest in world.side_quests:
            req = ", ".join([f"{k} x{v}" for k, v in quest.required_items.items()]) or "none"
            reward = ", ".join([f"{k} x{v}" for k, v in quest.reward_items.items()]) or "none"
            quest_lines.append(
                f"{quest.quest_id}: {quest.title} ({quest.category}) @ {quest.suggested_location} "
                f"giver={quest.giver_npc_id} required={req} reward={reward}"
            )
        parts.append("Side quests: " + "; ".join(quest_lines))
    if world.npcs:
        roster = []
        for npc in world.npcs:
            roster.append(f"{npc.npc_id}: {npc.name} ({npc.profession}) @ {npc.starting_location}")
        parts.append("NPC roster: " + "; ".join(roster))
        professions = sorted({npc.profession for npc in world.npcs if npc.profession})
        if professions:
            parts.append("Known professions: " + ", ".join(professions))
    if world.locations:
        locs = [f"{loc.location_id}: {loc.name}" for loc in world.locations]
        parts.append("Location roster: " + "; ".join(locs))

    text = "\n".join([p for p in parts if p])
    metadata = normalize_metadata(
        {
            "doc_type": "world_bible",
            "session_id": session_id,
        }
    )
    return Document(id=make_doc_id(metadata, text), text=text, metadata=metadata)
