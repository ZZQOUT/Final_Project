"""State update helpers for turn outputs and quest/inventory progression."""
from __future__ import annotations

from typing import Dict, Any
import re

from rpg_story.models.world import GameState
from rpg_story.models.turn import TurnOutput, QuestProgressUpdate

_ITEM_ALIAS_TO_CANONICAL = {
    "口粮": "ration",
    "ration": "ration",
    "rations": "ration",
    "疗伤草": "healing_herb",
    "healing_herb": "healing_herb",
    "healing herb": "healing_herb",
    "moon_herb": "moon_herb",
    "moon herb": "moon_herb",
    "月光草": "moon_herb",
    "火焰抗性药水": "fire_resistance_potion",
    "fire_resistance_potion": "fire_resistance_potion",
    "fire resistance potion": "fire_resistance_potion",
}


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
        category = str(entry.get("category") or "side")
        if category == "main":
            # Main quest completion is gated by final trial, not direct delivery.
            continue
        giver = entry.get("giver_npc_id")
        if giver and giver != npc_id and not _allow_delivery_fallback(state, entry, npc_id, location_id):
            continue

        required = entry.get("required_items", {}) or {}
        required = {str(k): max(0, int(v)) for k, v in required.items() if _is_int_like(v)}
        if not required:
            continue

        collected = entry.get("collected_items", {}) or {}
        collected = {str(k): max(0, int(v)) for k, v in collected.items() if _is_int_like(v)}

        made_progress = False
        for item, need in required.items():
            have_delivered = int(collected.get(item, 0))
            still_need = max(0, int(need) - have_delivered)
            if still_need <= 0:
                continue
            matched_keys = _matching_item_keys(item, remaining)
            if not matched_keys:
                continue
            transferred = 0
            for inv_key in matched_keys:
                if still_need <= 0:
                    break
                available = int(remaining.get(inv_key, 0))
                if available <= 0:
                    continue
                move_amount = min(still_need, available)
                if move_amount <= 0:
                    continue
                still_need -= move_amount
                transferred += move_amount
                remaining[inv_key] = available - move_amount
                delivered_delta[inv_key] = int(delivered_delta.get(inv_key, 0)) + move_amount
                made_progress = True
                if remaining[inv_key] <= 0:
                    remaining.pop(inv_key, None)
            if transferred > 0:
                collected[item] = have_delivered + transferred
        if not made_progress:
            continue

        entry["collected_items"] = collected
        complete = all(int(collected.get(item, 0)) >= int(need) for item, need in required.items())
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
    world_defs = _world_quest_definitions(data.get("world"))
    world_def = world_defs.get(quest_id)
    current = quest_journal.get(quest_id)
    if not isinstance(current, dict):
        if world_def:
            current = _progress_from_world_definition(quest_id, world_def)
        else:
            # Ignore ad-hoc quest injection from LLM to keep quest pipeline stable.
            return

    if world_def:
        # Existing world quests are immutable in definition fields.
        current["title"] = str(world_def.get("title") or current.get("title") or _fallback_title(quest_id))
        current["category"] = "main" if str(world_def.get("category") or "side") == "main" else "side"
        current["objective"] = str(world_def.get("objective") or current.get("objective") or "")
        current["giver_npc_id"] = world_def.get("giver_npc_id")
        current["reward_hint"] = world_def.get("reward_hint")
        current["required_items"] = _normalize_item_map(world_def.get("required_items"), positive_only=True)
        current["reward_items"] = _normalize_item_map(world_def.get("reward_items"), positive_only=True)
    else:
        # Keep non-world quest definitions stable once initialized.
        if update.title is not None and not current.get("title"):
            current["title"] = update.title
        if not current.get("title"):
            current["title"] = _fallback_title(quest_id)
        if update.category is not None and not current.get("category"):
            current["category"] = update.category
        if update.objective is not None and not current.get("objective"):
            current["objective"] = update.objective
        if update.giver_npc_id is not None and not current.get("giver_npc_id"):
            current["giver_npc_id"] = update.giver_npc_id
        elif current.get("giver_npc_id") is None:
            current["giver_npc_id"] = npc_id
        if update.reward_hint is not None and not current.get("reward_hint"):
            current["reward_hint"] = update.reward_hint
        if not current.get("required_items"):
            current["required_items"] = _normalize_item_map(update.required_items, positive_only=True)
        else:
            current["required_items"] = _normalize_item_map(current.get("required_items"), positive_only=True)
        if not current.get("reward_items"):
            current["reward_items"] = _normalize_item_map(update.reward_items, positive_only=True)
        else:
            current["reward_items"] = _normalize_item_map(current.get("reward_items"), positive_only=True)

    if update.guidance is not None:
        current["guidance"] = update.guidance

    required = _normalize_item_map(current.get("required_items"), positive_only=True)
    current["required_items"] = required
    collected = _coerce_collected_map(current.get("collected_items"), required)
    for item, delta in update.collected_items_delta.items():
        if not _allow_turn_collected_delta(current, required, int(delta)):
            continue
        normalized_item = _resolve_required_item(required, item)
        if normalized_item is None:
            continue
        current_count = int(collected.get(normalized_item, 0))
        new_count = current_count + int(delta)
        if new_count <= 0:
            collected.pop(normalized_item, None)
            continue
        if normalized_item in required:
            new_count = min(new_count, int(required.get(normalized_item, new_count)))
        collected[normalized_item] = new_count
    current["collected_items"] = collected

    if update.status is not None:
        current["status"] = _normalize_status(update.status)
    elif not current.get("status"):
        current["status"] = "available"
    if current.get("category") == "main" and not data.get("main_quest_id"):
        data["main_quest_id"] = quest_id
    quests[quest_id] = current.get("status") or "available"
    quest_journal[quest_id] = current


def _sync_quest_journal_with_inventory(data: Dict[str, Any]) -> None:
    inventory = data.get("inventory", {})
    quests = data.get("quests", {})
    quest_journal = data.get("quest_journal", {})
    flags = data.get("flags", {})
    world_defs = _world_quest_definitions(data.get("world"))
    prefer_chinese = _prefer_chinese_data(data)
    if not isinstance(inventory, dict):
        inventory = {}
    if not isinstance(quests, dict):
        quests = {}
    if not isinstance(quest_journal, dict):
        quest_journal = {}
    if not isinstance(flags, dict):
        flags = {}

    for quest_id, raw in list(quest_journal.items()):
        if not isinstance(raw, dict):
            continue
        world_def = world_defs.get(quest_id)
        if world_def:
            raw["title"] = str(world_def.get("title") or raw.get("title") or _fallback_title(quest_id))
            raw["category"] = "main" if str(world_def.get("category") or "side") == "main" else "side"
            raw["objective"] = str(world_def.get("objective") or raw.get("objective") or "")
            raw["giver_npc_id"] = world_def.get("giver_npc_id")
            raw["reward_hint"] = world_def.get("reward_hint")
            raw["required_items"] = _normalize_item_map(world_def.get("required_items"), positive_only=True)
            raw["reward_items"] = _normalize_item_map(world_def.get("reward_items"), positive_only=True)
        status = _normalize_status(raw.get("status") or "available")
        required = _normalize_item_map(raw.get("required_items"), positive_only=True)
        raw["required_items"] = required
        _normalize_quest_text_consistency(raw, prefer_chinese=prefer_chinese)

        collected = _coerce_collected_map(raw.get("collected_items"), required)
        raw["collected_items"] = collected

        category = str(raw.get("category") or "side")
        if required and status != "failed":
            if category == "main":
                for item, need in required.items():
                    bag = _inventory_amount_for_item(inventory, item)
                    raw["collected_items"][item] = min(int(need), int(bag))
                complete = all(raw["collected_items"].get(item, 0) >= need for item, need in required.items())
                passed_flag = bool(flags.get(f"main_trial_passed_{quest_id}", False))
                failed_flag = bool(flags.get(f"main_trial_failed_{quest_id}", False))
                if passed_flag and complete:
                    status = "completed"
                    if not raw.get("guidance"):
                        raw["guidance"] = (
                            "终局考验通过，主线已完成。"
                            if prefer_chinese
                            else "Final trial passed. Main quest completed."
                        )
                elif failed_flag:
                    status = "failed"
                    raw["guidance"] = (
                        "终局考验失败，请重新集齐物资后再挑战。"
                        if prefer_chinese
                        else "Final trial failed. Gather all required items and challenge again."
                    )
                else:
                    status = "active"
                    finale_npc_id, finale_loc_id = _main_trial_target_from_data(data, raw)
                    raw["guidance"] = _trial_hint(
                        required,
                        raw["collected_items"],
                        prefer_chinese,
                        finale_npc_name=_world_npc_name(data, finale_npc_id),
                        finale_loc_name=_world_location_name(data, finale_loc_id),
                    )
            else:
                complete = all(raw["collected_items"].get(item, 0) >= need for item, need in required.items())
                if status == "completed" and not complete:
                    status = "active"
                    raw["guidance"] = _delivery_hint(required, raw["collected_items"], inventory, prefer_chinese)
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
        bag = _inventory_amount_for_item(inventory, item)
        if prefer_chinese:
            parts.append(f"{item} 已交付{done}/{need}（背包{bag}）")
        else:
            parts.append(f"{item} delivered {done}/{need} (bag {bag})")
    if prefer_chinese:
        return "交付进度：" + "，".join(parts)
    return "Delivery progress: " + ", ".join(parts)


def evaluate_main_trial_readiness(state: GameState) -> tuple[bool, Dict[str, Dict[str, int]]]:
    if not state.main_quest_id:
        return False, {}
    main = state.quest_journal.get(state.main_quest_id)
    if not main:
        return False, {}
    required = main.required_items or {}
    if not required:
        return False, {}
    progress: Dict[str, Dict[str, int]] = {}
    for item, need in required.items():
        have = _inventory_amount_for_item(state.inventory, item)
        progress[item] = {"have": int(have), "need": int(need)}
    ready = all(info["have"] >= info["need"] for info in progress.values())
    return ready, progress


def resolve_main_trial(state: GameState, *, passed: bool) -> GameState:
    data = state.model_dump()
    quest_journal = data.get("quest_journal", {})
    flags = data.get("flags", {})
    if not isinstance(flags, dict):
        flags = {}
    main_quest_id = data.get("main_quest_id")
    if not main_quest_id or main_quest_id not in quest_journal:
        return state
    entry = quest_journal.get(main_quest_id)
    if not isinstance(entry, dict):
        return state
    if passed:
        flags[f"main_trial_passed_{main_quest_id}"] = True
        flags[f"main_trial_failed_{main_quest_id}"] = False
        entry["status"] = "completed"
    else:
        flags[f"main_trial_failed_{main_quest_id}"] = True
        flags[f"main_trial_passed_{main_quest_id}"] = False
        entry["status"] = "failed"
    quest_journal[main_quest_id] = entry
    data["flags"] = flags
    data["quest_journal"] = quest_journal
    _sync_quest_journal_with_inventory(data)
    return GameState.model_validate(data)


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


def _normalize_item_map(value: Any, *, positive_only: bool) -> Dict[str, int]:
    if not isinstance(value, dict):
        return {}
    normalized: Dict[str, int] = {}
    for k, v in value.items():
        if not _is_int_like(v):
            continue
        amount = int(v)
        if positive_only and amount <= 0:
            continue
        if amount < 0:
            continue
        normalized[str(k)] = amount
    return normalized


def _coerce_collected_map(value: Any, required: Dict[str, int]) -> Dict[str, int]:
    if not isinstance(value, dict):
        return {}
    collected: Dict[str, int] = {}
    for raw_item, raw_count in value.items():
        if not _is_int_like(raw_count):
            continue
        item = _resolve_required_item(required, str(raw_item))
        if item is None:
            continue
        amount = max(0, int(raw_count))
        if item in required:
            amount = min(amount, int(required.get(item, amount)))
        collected[item] = max(collected.get(item, 0), amount)
    return collected


def _resolve_required_item(required: Dict[str, int], item: str) -> str | None:
    if not required:
        return str(item)
    text = str(item)
    if text in required:
        return text
    canonical = _canonical_item_key(text)
    for key in required.keys():
        if _canonical_item_key(key) == canonical:
            return key
    return None


def _allow_turn_collected_delta(entry: Dict[str, Any], required: Dict[str, int], delta: int) -> bool:
    # Delivery progress for item quests must be changed by explicit delivery action,
    # not by free-form dialogue turns. This prevents "auto-submit on chat".
    if int(delta) <= 0:
        return True
    category = str(entry.get("category") or "side")
    if category in {"side", "main"} and bool(required):
        return False
    return True


def _world_quest_definitions(world_data: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(world_data, dict):
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    main = world_data.get("main_quest")
    if isinstance(main, dict):
        quest_id = str(main.get("quest_id") or "").strip()
        if quest_id:
            result[quest_id] = {
                "quest_id": quest_id,
                "title": main.get("title"),
                "category": "main",
                "objective": main.get("objective"),
                "giver_npc_id": main.get("giver_npc_id"),
                "required_items": _normalize_item_map(main.get("required_items"), positive_only=True),
                "reward_items": _normalize_item_map(main.get("reward_items"), positive_only=True),
                "reward_hint": main.get("reward_hint"),
            }
    side = world_data.get("side_quests")
    if isinstance(side, list):
        for raw in side:
            if not isinstance(raw, dict):
                continue
            quest_id = str(raw.get("quest_id") or "").strip()
            if not quest_id:
                continue
            result[quest_id] = {
                "quest_id": quest_id,
                "title": raw.get("title"),
                "category": "side",
                "objective": raw.get("objective"),
                "giver_npc_id": raw.get("giver_npc_id"),
                "required_items": _normalize_item_map(raw.get("required_items"), positive_only=True),
                "reward_items": _normalize_item_map(raw.get("reward_items"), positive_only=True),
                "reward_hint": raw.get("reward_hint"),
            }
    return result


def _progress_from_world_definition(quest_id: str, world_def: Dict[str, Any]) -> Dict[str, Any]:
    required = _normalize_item_map(world_def.get("required_items"), positive_only=True)
    category = "main" if str(world_def.get("category") or "side") == "main" else "side"
    return {
        "quest_id": quest_id,
        "title": str(world_def.get("title") or _fallback_title(quest_id)),
        "category": category,
        "status": "active" if category == "main" else "available",
        "objective": str(world_def.get("objective") or ""),
        "guidance": "",
        "giver_npc_id": world_def.get("giver_npc_id"),
        "required_items": required,
        "collected_items": {k: 0 for k in required.keys()},
        "reward_items": _normalize_item_map(world_def.get("reward_items"), positive_only=True),
        "reward_hint": world_def.get("reward_hint"),
    }


def _normalize_quest_text_consistency(entry: Dict[str, Any], *, prefer_chinese: bool) -> None:
    required = _normalize_item_map(entry.get("required_items"), positive_only=True)
    if not required:
        return
    item_names = [str(k) for k in required.keys()]
    first_item = item_names[0]

    objective = str(entry.get("objective") or "").strip()
    if objective:
        mentions = any(name and name in objective for name in item_names)
        if not mentions:
            req = "，".join([f"{name} {required[name]}" for name in item_names])
            objective = f"{objective}（需求：{req}）" if prefer_chinese else f"{objective} (required: {req})"
            entry["objective"] = objective

    title = str(entry.get("title") or "").strip()
    if not title:
        return
    mentions = any(name and name in title for name in item_names)
    if mentions:
        return
    if prefer_chinese:
        if any(token in title for token in ("寻找", "收集", "采集", "获取")):
            entry["title"] = f"收集{first_item}"
        return
    lower = title.lower()
    if any(token in lower for token in ("find", "collect", "gather", "obtain", "fetch")):
        entry["title"] = f"Collect {first_item}"


def _quest_suggested_location(state: GameState, quest_id: str) -> str | None:
    if state.world.main_quest and state.world.main_quest.quest_id == quest_id:
        return state.world.main_quest.suggested_location
    for quest in state.world.side_quests:
        if quest.quest_id == quest_id:
            return quest.suggested_location
    return None


def _allow_delivery_fallback(state: GameState, entry: Dict[str, Any], npc_id: str, location_id: str) -> bool:
    category = str(entry.get("category") or "side")
    if category != "side":
        return False
    suggested = _quest_suggested_location(state, str(entry.get("quest_id") or ""))
    if suggested and suggested != location_id:
        return False
    npc_name = ""
    for npc in state.world.npcs:
        if npc.npc_id == npc_id:
            npc_name = npc.name or ""
            break
    text = " ".join(
        [
            str(entry.get("title") or ""),
            str(entry.get("objective") or ""),
            str(entry.get("guidance") or ""),
        ]
    )
    if npc_name and npc_name in text:
        return True
    # Fallback for noisy LLM quest giver IDs.
    return suggested == location_id if suggested else True


def _matching_item_keys(target_item: str, pool: Dict[str, int]) -> list[str]:
    target = _canonical_item_key(target_item)
    matched = []
    for key, amount in pool.items():
        if int(amount) <= 0:
            continue
        if _canonical_item_key(key) == target:
            matched.append(key)
    return matched


def _inventory_amount_for_item(inventory: Dict[str, Any], item: str) -> int:
    total = 0
    for key, raw in (inventory or {}).items():
        if _canonical_item_key(str(key)) != _canonical_item_key(item):
            continue
        if not _is_int_like(raw):
            continue
        total += max(0, int(raw))
    return total


def _canonical_item_key(item: str) -> str:
    text = str(item or "").strip().lower()
    text = text.replace("-", "_")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if text.endswith("s") and len(text) > 2:
        singular = text[:-1]
        if singular in _ITEM_ALIAS_TO_CANONICAL:
            text = singular
    if text in _ITEM_ALIAS_TO_CANONICAL:
        return _ITEM_ALIAS_TO_CANONICAL[text]
    return text


def _world_npc_name(data: Dict[str, Any], npc_id: str | None) -> str | None:
    if not npc_id:
        return None
    world = data.get("world")
    if not isinstance(world, dict):
        return npc_id
    npcs = world.get("npcs")
    if not isinstance(npcs, list):
        return npc_id
    for raw in npcs:
        if not isinstance(raw, dict):
            continue
        if str(raw.get("npc_id")) == str(npc_id):
            return str(raw.get("name") or npc_id)
    return str(npc_id)


def _world_location_name(data: Dict[str, Any], location_id: str | None) -> str | None:
    if not location_id:
        return None
    world = data.get("world")
    if not isinstance(world, dict):
        return location_id
    locations = world.get("locations")
    if not isinstance(locations, list):
        return location_id
    for raw in locations:
        if not isinstance(raw, dict):
            continue
        if str(raw.get("location_id")) == str(location_id):
            return str(raw.get("name") or location_id)
    return str(location_id)


def _main_trial_target_from_data(data: Dict[str, Any], raw: Dict[str, Any]) -> tuple[str | None, str | None]:
    npc_id = str(raw.get("giver_npc_id") or "").strip() or None
    loc_id = None
    npc_locations = data.get("npc_locations")
    if npc_id and isinstance(npc_locations, dict):
        mapped = npc_locations.get(npc_id)
        if isinstance(mapped, str) and mapped.strip():
            loc_id = mapped.strip()
    if not loc_id:
        world = data.get("world")
        if isinstance(world, dict):
            main = world.get("main_quest")
            if isinstance(main, dict):
                mapped_loc = main.get("suggested_location")
                if isinstance(mapped_loc, str) and mapped_loc.strip():
                    loc_id = mapped_loc.strip()
    if not loc_id:
        player_loc = data.get("player_location")
        if isinstance(player_loc, str) and player_loc.strip():
            loc_id = player_loc.strip()
    return npc_id, loc_id


def _trial_hint(
    required: Dict[str, int],
    collected: Dict[str, int],
    prefer_chinese: bool,
    *,
    finale_npc_name: str | None = None,
    finale_loc_name: str | None = None,
) -> str:
    parts = []
    for item, need in required.items():
        have = int(collected.get(item, 0))
        parts.append(f"{item} {have}/{need}")
    all_ready = all(int(collected.get(item, 0)) >= int(need) for item, need in required.items())
    target_hint_zh = ""
    target_hint_en = ""
    if finale_npc_name and finale_loc_name:
        target_hint_zh = f" 前往{finale_loc_name}与{finale_npc_name}对话发起最终考验。"
        target_hint_en = f" Go to {finale_loc_name} and talk to {finale_npc_name} to start the final trial."
    elif finale_npc_name:
        target_hint_zh = f" 与{finale_npc_name}对话发起最终考验。"
        target_hint_en = f" Talk to {finale_npc_name} to start the final trial."
    elif finale_loc_name:
        target_hint_zh = f" 前往{finale_loc_name}发起最终考验。"
        target_hint_en = f" Go to {finale_loc_name} to start the final trial."
    if prefer_chinese:
        return ("已达成主线物资要求。" + target_hint_zh).strip() if all_ready else "主线准备进度：" + "，".join(parts)
    return ("Main items ready." + target_hint_en).strip() if all_ready else "Main preparation: " + ", ".join(parts)


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
