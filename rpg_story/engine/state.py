"""State update helpers for turn outputs and quest/inventory progression."""
from __future__ import annotations

from typing import Dict, Any
import re

from rpg_story.models.world import GameState
from rpg_story.models.turn import TurnOutput, QuestProgressUpdate


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

    quests = data.get("quests", {})
    quest_journal = data.get("quest_journal", {})

    # Legacy quest updates (status only)
    if output.world_updates.quest_updates:
        quests.update(output.world_updates.quest_updates)
        data["quests"] = quests
        for quest_id, status in output.world_updates.quest_updates.items():
            entry = quest_journal.get(quest_id) or _new_quest_progress(quest_id)
            entry["status"] = _normalize_status(status)
            if not entry.get("title"):
                entry["title"] = _fallback_title(quest_id)
            quest_journal[quest_id] = entry

    # Inventory updates
    inventory = data.get("inventory", {})
    _merge_inventory(inventory, output.world_updates.inventory_delta)
    data["inventory"] = inventory

    # Structured quest progress updates
    for update in output.world_updates.quest_progress_updates:
        _apply_quest_progress_update(data, quest_journal, quests, update, npc_id)

    data["quest_journal"] = quest_journal
    data["quests"] = quests
    _sync_quest_journal_with_inventory(data)

    # Player movement
    location_ids = {loc["location_id"] for loc in data["world"]["locations"]}
    if output.world_updates.player_location:
        if output.world_updates.player_location not in location_ids:
            raise ValueError(f"Invalid player_location: {output.world_updates.player_location}")
        data["player_location"] = output.world_updates.player_location

    return GameState.model_validate(data)


def sync_quest_journal(state: GameState) -> GameState:
    """Recompute quest progress from inventory without applying a turn output."""
    data = state.model_dump()
    _sync_quest_journal_with_inventory(data)
    return GameState.model_validate(data)


def deliver_items_to_npc(
    state: GameState,
    npc_id: str,
    location_id: str,
    handover: Dict[str, int],
) -> tuple[GameState, list[str], Dict[str, int], Dict[str, int]]:
    """Deliver selected inventory items to an NPC and update quests.

    Rules:
    - Delivery can only affect quests assigned to that npc_id.
    - Suggested location must match if quest specifies one.
    - Delivered amounts are tracked in quest_journal.collected_items.
    - Side quest completion grants reward_items once.
    """
    data = state.model_dump()
    inventory = data.get("inventory", {})
    quests = data.get("quests", {})
    quest_journal = data.get("quest_journal", {})
    notices: list[str] = []
    rewards_delta: Dict[str, int] = {}
    delivered_delta: Dict[str, int] = {}
    prefer_chinese = _prefer_chinese_state(state)

    normalized_handover = {
        str(item): max(0, int(amount))
        for item, amount in handover.items()
        if _is_int_like(amount) and int(amount) > 0
    }
    if not normalized_handover:
        return state, notices, rewards_delta, delivered_delta

    # Clamp handover by current inventory first.
    remaining = {
        item: min(int(inventory.get(item, 0)), amount)
        for item, amount in normalized_handover.items()
        if int(inventory.get(item, 0)) > 0
    }
    remaining = {item: amount for item, amount in remaining.items() if amount > 0}
    if not remaining:
        return state, notices, rewards_delta, delivered_delta
    npc_locations = data.get("npc_locations", {})
    if npc_locations.get(npc_id) != location_id:
        return state, notices, rewards_delta, delivered_delta

    quests_in_order = list(quest_journal.items())
    quests_in_order.sort(
        key=lambda pair: (
            0 if str((pair[1] or {}).get("category", "side")) == "side" else 1,
            pair[0],
        )
    )

    for quest_id, entry in quests_in_order:
        if not isinstance(entry, dict):
            continue
        status = _normalize_status(entry.get("status"))
        if status in {"completed", "failed"}:
            continue
        giver = entry.get("giver_npc_id")
        if giver and giver != npc_id:
            continue
        suggested = _quest_suggested_location(state, quest_id)
        if suggested and suggested != location_id:
            continue

        required = entry.get("required_items", {}) or {}
        required = {str(k): max(0, int(v)) for k, v in required.items() if _is_int_like(v)}
        if not required:
            continue

        collected = entry.get("collected_items", {}) or {}
        collected = {str(k): max(0, int(v)) for k, v in collected.items() if _is_int_like(v)}

        made_progress = False
        for item, need in required.items():
            if item not in remaining:
                continue
            have_delivered = int(collected.get(item, 0))
            still_need = max(0, int(need) - have_delivered)
            if still_need <= 0:
                continue
            transfer = min(still_need, int(remaining.get(item, 0)))
            if transfer <= 0:
                continue
            collected[item] = have_delivered + transfer
            remaining[item] = int(remaining.get(item, 0)) - transfer
            delivered_delta[item] = int(delivered_delta.get(item, 0)) + transfer
            made_progress = True
            if remaining[item] <= 0:
                remaining.pop(item, None)
        if not made_progress:
            continue

        entry["collected_items"] = collected
        complete = all(int(collected.get(item, 0)) >= int(need) for item, need in required.items())
        category = str(entry.get("category") or "side")
        if complete:
            entry["status"] = "completed"
            if category == "side":
                reward_items = entry.get("reward_items", {}) or {}
                reward_items = {
                    str(k): max(0, int(v)) for k, v in reward_items.items() if _is_int_like(v)
                }
                for item, amount in reward_items.items():
                    inventory[item] = int(inventory.get(item, 0)) + int(amount)
                    rewards_delta[item] = int(rewards_delta.get(item, 0)) + int(amount)
                if prefer_chinese:
                    entry["guidance"] = "支线已完成，奖励已发放。"
                    notices.append(f"支线完成：{entry.get('title', quest_id)}")
                else:
                    entry["guidance"] = "Side quest completed and rewards granted."
                    notices.append(f"Side quest completed: {entry.get('title', quest_id)}")
            else:
                if prefer_chinese:
                    entry["guidance"] = "主线交付完成。"
                    notices.append(f"主线推进：{entry.get('title', quest_id)}")
                else:
                    entry["guidance"] = "Main quest delivery completed."
                    notices.append(f"Main quest progressed: {entry.get('title', quest_id)}")
        else:
            entry["status"] = "active"
            entry["guidance"] = _missing_items_hint(required, collected, prefer_chinese)
        quests[quest_id] = entry["status"]
        quest_journal[quest_id] = entry

    # Consume delivered items from bag.
    for item, amount in delivered_delta.items():
        inventory[item] = int(inventory.get(item, 0)) - int(amount)
        if int(inventory.get(item, 0)) <= 0:
            inventory.pop(item, None)

    data["inventory"] = inventory
    data["quests"] = quests
    data["quest_journal"] = quest_journal
    _sync_quest_journal_with_inventory(data)
    return GameState.model_validate(data), notices, rewards_delta, delivered_delta


def _new_quest_progress(quest_id: str) -> Dict[str, Any]:
    return {
        "quest_id": quest_id,
        "title": _fallback_title(quest_id),
        "category": "side",
        "status": "available",
        "objective": "",
        "guidance": "",
        "giver_npc_id": None,
        "required_items": {},
        "collected_items": {},
        "reward_items": {},
        "reward_hint": None,
    }


def _apply_quest_progress_update(
    data: Dict[str, Any],
    quest_journal: Dict[str, Any],
    quests: Dict[str, Any],
    update: QuestProgressUpdate,
    npc_id: str,
) -> None:
    quest_id = update.quest_id
    current = quest_journal.get(quest_id) or _new_quest_progress(quest_id)
    if update.title is not None:
        current["title"] = update.title
    if not current.get("title"):
        current["title"] = _fallback_title(quest_id)
    if update.category is not None:
        current["category"] = update.category
    if update.objective is not None:
        current["objective"] = update.objective
    if update.guidance is not None:
        current["guidance"] = update.guidance
    if update.giver_npc_id is not None:
        current["giver_npc_id"] = update.giver_npc_id
    elif current.get("giver_npc_id") is None:
        current["giver_npc_id"] = npc_id
    if update.reward_hint is not None:
        current["reward_hint"] = update.reward_hint

    required = current.get("required_items", {})
    if not isinstance(required, dict):
        required = {}
    required = {str(k): max(0, int(v)) for k, v in required.items() if _is_int_like(v)}
    required.update({k: v for k, v in update.required_items.items() if int(v) > 0})
    current["required_items"] = required
    reward_items = current.get("reward_items", {})
    if not isinstance(reward_items, dict):
        reward_items = {}
    reward_items = {str(k): max(0, int(v)) for k, v in reward_items.items() if _is_int_like(v)}
    reward_items.update({k: v for k, v in update.reward_items.items() if int(v) > 0})
    current["reward_items"] = reward_items

    collected = current.get("collected_items", {})
    if not isinstance(collected, dict):
        collected = {}
    collected = {str(k): max(0, int(v)) for k, v in collected.items() if _is_int_like(v)}
    for item, delta in update.collected_items_delta.items():
        current_count = int(collected.get(item, 0))
        new_count = current_count + int(delta)
        if new_count <= 0:
            collected.pop(item, None)
        else:
            collected[item] = new_count
    current["collected_items"] = collected

    if update.status is not None:
        current["status"] = _normalize_status(update.status)
    if current.get("category") == "main" and not data.get("main_quest_id"):
        data["main_quest_id"] = quest_id
    quests[quest_id] = current.get("status") or "available"
    quest_journal[quest_id] = current


def _sync_quest_journal_with_inventory(data: Dict[str, Any]) -> None:
    inventory = data.get("inventory", {})
    quests = data.get("quests", {})
    quest_journal = data.get("quest_journal", {})
    prefer_chinese = _prefer_chinese_data(data)
    if not isinstance(inventory, dict):
        inventory = {}
    if not isinstance(quests, dict):
        quests = {}
    if not isinstance(quest_journal, dict):
        quest_journal = {}

    for quest_id, raw in list(quest_journal.items()):
        if not isinstance(raw, dict):
            continue
        status = _normalize_status(raw.get("status") or "available")
        required = raw.get("required_items", {})
        if not isinstance(required, dict):
            required = {}
        required = {str(k): max(0, int(v)) for k, v in required.items() if _is_int_like(v)}
        raw["required_items"] = required

        collected = raw.get("collected_items", {})
        if not isinstance(collected, dict):
            collected = {}
        collected = {
            str(k): min(max(0, int(v)), int(required.get(str(k), int(v))))
            for k, v in collected.items()
            if _is_int_like(v)
        }
        raw["collected_items"] = collected

        category = str(raw.get("category") or "side")
        if required and status != "failed":
            complete = all(raw["collected_items"].get(item, 0) >= need for item, need in required.items())
            if category == "main":
                if complete:
                    status = "completed"
                    if not raw.get("guidance"):
                        raw["guidance"] = (
                            "主线材料齐备，可以推进终章。"
                            if prefer_chinese
                            else "Main quest materials are complete. You can advance the finale."
                        )
                elif status == "available":
                    status = "active"
                    raw["guidance"] = _delivery_hint(required, raw["collected_items"], inventory, prefer_chinese)
                elif status == "active":
                    raw["guidance"] = _delivery_hint(required, raw["collected_items"], inventory, prefer_chinese)
            else:
                if status == "available":
                    status = "active"
                    raw["guidance"] = _delivery_hint(required, raw["collected_items"], inventory, prefer_chinese)
                elif status == "active" and complete:
                    raw["guidance"] = (
                        "材料已齐，请与任务 NPC 同地交付领取奖励。"
                        if prefer_chinese
                        else "Requirements met. Deliver items to the quest NPC at the same location to claim rewards."
                    )
                elif status == "active":
                    raw["guidance"] = _delivery_hint(required, raw["collected_items"], inventory, prefer_chinese)

        raw["status"] = status
        if not raw.get("title"):
            raw["title"] = _fallback_title(quest_id)
        quests[quest_id] = status
        quest_journal[quest_id] = raw

    data["quests"] = quests
    data["quest_journal"] = quest_journal


def _missing_items_hint(required: Dict[str, int], collected: Dict[str, int], prefer_chinese: bool) -> str:
    missing_parts = []
    for item, need in required.items():
        have = int(collected.get(item, 0))
        if have >= need:
            continue
        missing_parts.append(f"{item} {have}/{need}")
    if not missing_parts:
        return "任务已满足条件，准备交付。" if prefer_chinese else "Requirements met. Ready for delivery."
    if prefer_chinese:
        return "缺少物品：" + "，".join(missing_parts)
    return "Missing items: " + ", ".join(missing_parts)


def _delivery_hint(
    required: Dict[str, int],
    delivered: Dict[str, int],
    inventory: Dict[str, Any],
    prefer_chinese: bool,
) -> str:
    parts = []
    for item, need in required.items():
        done = int(delivered.get(item, 0))
        bag = int(inventory.get(item, 0)) if _is_int_like(inventory.get(item, 0)) else 0
        if prefer_chinese:
            parts.append(f"{item} 已交付{done}/{need}（背包{bag}）")
        else:
            parts.append(f"{item} delivered {done}/{need} (bag {bag})")
    if prefer_chinese:
        return "交付进度：" + "，".join(parts)
    return "Delivery progress: " + ", ".join(parts)


def _merge_inventory(inventory: Dict[str, Any], delta: Dict[str, int]) -> None:
    if not isinstance(inventory, dict) or not delta:
        return
    for item, amount in delta.items():
        if amount == 0:
            continue
        current = inventory.get(item, 0)
        try:
            current_num = int(float(current))
        except Exception:
            current_num = 0
        new_value = current_num + int(amount)
        if new_value <= 0:
            inventory.pop(item, None)
        else:
            inventory[item] = new_value


def _fallback_title(quest_id: str) -> str:
    parts = [part for part in quest_id.replace("-", "_").split("_") if part]
    if not parts:
        return "Quest"
    return " ".join(part.capitalize() for part in parts)


def _normalize_status(status: Any) -> str:
    text = str(status or "").strip().lower()
    mapping = {
        "accepted": "active",
        "in_progress": "active",
        "ongoing": "active",
        "started": "active",
        "done": "completed",
        "finished": "completed",
        "complete": "completed",
    }
    if text in {"available", "active", "completed", "failed"}:
        return text
    return mapping.get(text, "active")


def _is_int_like(value: Any) -> bool:
    try:
        int(float(value))
        return True
    except Exception:
        return False


def _quest_suggested_location(state: GameState, quest_id: str) -> str | None:
    if state.world.main_quest and state.world.main_quest.quest_id == quest_id:
        return state.world.main_quest.suggested_location
    for quest in state.world.side_quests:
        if quest.quest_id == quest_id:
            return quest.suggested_location
    return None


def _prefer_chinese_state(state: GameState) -> bool:
    return _prefer_chinese_data(state.model_dump())


def _prefer_chinese_data(data: Dict[str, Any]) -> bool:
    world = data.get("world") if isinstance(data, dict) else None
    if isinstance(world, dict):
        bible = world.get("world_bible")
        if isinstance(bible, dict):
            lang = bible.get("narrative_language")
            if lang in {"zh", "en"}:
                return lang == "zh"
        text = " ".join(
            [
                str(world.get("title") or ""),
                str(world.get("starting_hook") or ""),
                str(world.get("initial_quest") or ""),
            ]
        )
        return bool(re.search(r"[\u4e00-\u9fff]", text))
    return True
