"""Streamlit UI for world creation and RPG gameplay."""
from __future__ import annotations

from pathlib import Path
from collections import deque
from datetime import datetime, timezone
import html
import json
import os
import re
import sys

import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rpg_story.config import load_config
from rpg_story.engine.orchestrator import TurnPipeline
from rpg_story.engine.state import (
    sync_quest_journal,
    deliver_items_to_npc,
    evaluate_main_trial_readiness,
    resolve_main_trial,
)
from rpg_story.llm.client import QwenOpenAICompatibleClient
from rpg_story.models.world import WorldSpec, GameState, LocationSpec
from rpg_story.persistence.store import (
    generate_session_id,
    load_state,
    save_state,
    read_turn_logs,
    default_sessions_root,
    append_turn_log,
    append_story_summary,
    read_story_summaries,
)
from rpg_story.world.generator import (
    generate_world_spec,
    initialize_game_state,
    suggest_location_resource_template,
)


st.set_page_config(page_title="RPG Story Prototype", layout="wide")


ITEM_EN_TO_ZH = {
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
ITEM_ZH_TO_EN = {v: k for k, v in ITEM_EN_TO_ZH.items()}


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _prefer_chinese_ui(world: WorldSpec) -> bool:
    lang = getattr(world.world_bible, "narrative_language", None)
    if lang in {"zh", "en"}:
        return lang == "zh"
    text = " ".join([world.title, world.starting_hook, world.initial_quest])
    return _contains_cjk(text)


def _display_item_name(item: str, prefer_chinese: bool) -> str:
    normalized = re.sub(r"[\s\-]+", "_", str(item or "").strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if prefer_chinese:
        return ITEM_EN_TO_ZH.get(normalized, ITEM_EN_TO_ZH.get(item, item))
    if item in ITEM_ZH_TO_EN:
        return ITEM_ZH_TO_EN[item]
    return item


def _load_world_for_session(cfg, session_id: str, fallback: WorldSpec) -> WorldSpec:
    world_path = Path(cfg.app.worlds_dir) / session_id / "world.json"
    if world_path.exists():
        try:
            return WorldSpec.model_validate(json.loads(world_path.read_text(encoding="utf-8")))
        except Exception:
            return fallback
    return fallback


def _persist_world(cfg, session_id: str, world: WorldSpec) -> None:
    world_dir = Path(cfg.app.worlds_dir) / session_id
    world_dir.mkdir(parents=True, exist_ok=True)
    world_path = world_dir / "world.json"
    world_path.write_text(json.dumps(world.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")


def _map_positions(world: WorldSpec) -> dict[str, tuple[float, float]]:
    from_layout = {
        node.location_id: (float(node.x), float(node.y))
        for node in world.map_layout
        if world.get_location(node.location_id)
    }
    if len(from_layout) == len(world.locations):
        return from_layout

    if not world.locations:
        return {}
    adjacency = {loc.location_id: list(loc.connected_to) for loc in world.locations}
    start = world.starting_location if world.starting_location in adjacency else world.locations[0].location_id
    dist: dict[str, int] = {start: 0}
    queue: deque[str] = deque([start])
    while queue:
        node = queue.popleft()
        for nxt in adjacency.get(node, []):
            if nxt in dist:
                continue
            dist[nxt] = dist[node] + 1
            queue.append(nxt)
    max_dist = max(dist.values()) if dist else 0
    for loc in world.locations:
        if loc.location_id in dist:
            continue
        max_dist += 1
        dist[loc.location_id] = max_dist

    layers: dict[int, list[str]] = {}
    for loc_id, layer in dist.items():
        layers.setdefault(layer, []).append(loc_id)
    for layer in layers:
        layers[layer].sort()

    max_layer = max(layers.keys()) if layers else 0
    positions: dict[str, tuple[float, float]] = {}
    for layer, ids in layers.items():
        x = 50.0 if max_layer == 0 else 10.0 + (layer / max_layer) * 80.0
        count = len(ids)
        for idx, loc_id in enumerate(ids):
            y = 50.0 if count == 1 else 10.0 + ((idx + 1) / (count + 1)) * 80.0
            positions[loc_id] = (round(x, 2), round(y, 2))
    return positions


def _svg_map(world: WorldSpec, state: GameState) -> str:
    positions = _map_positions(world)
    width = 1200
    height = 760
    margin = 64
    xscale = (width - 2 * margin) / 100.0
    yscale = (height - 2 * margin) / 100.0

    def xy(loc_id: str) -> tuple[float, float]:
        x_norm, y_norm = positions.get(loc_id, (50.0, 50.0))
        return margin + x_norm * xscale, margin + y_norm * yscale

    edge_lines: list[str] = []
    seen_edges = set()
    for loc in world.locations:
        x1, y1 = xy(loc.location_id)
        for dst in loc.connected_to:
            if not world.get_location(dst):
                continue
            key = tuple(sorted((loc.location_id, dst)))
            if key in seen_edges:
                continue
            seen_edges.add(key)
            x2, y2 = xy(dst)
            edge_lines.append(
                f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' "
                "stroke='#8da0b3' stroke-width='3' stroke-opacity='0.9' />"
            )

    node_shapes: list[str] = []
    for loc in world.locations:
        x, y = xy(loc.location_id)
        is_player = loc.location_id == state.player_location
        fill = "#f4d35e" if is_player else "#7fb3d5"
        stroke = "#102a43" if is_player else "#243b53"
        radius = 26 if is_player else 21
        npc_count = len(state.npcs_at(loc.location_id))
        label = html.escape(loc.name)
        node_shapes.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{radius}' fill='{fill}' stroke='{stroke}' stroke-width='3' />"
        )
        node_shapes.append(
            f"<text x='{x:.1f}' y='{y + 42:.1f}' text-anchor='middle' font-size='22' "
            "fill='#102a43' font-weight='700'>"
            f"{label}</text>"
        )
        if npc_count > 0:
            node_shapes.append(
                f"<text x='{x:.1f}' y='{y + 8:.1f}' text-anchor='middle' font-size='16' fill='#102a43'>"
                f"{npc_count}</text>"
            )

    return "".join(
        [
            f"<svg id='rpg-map' width='{width}' height='{height}' viewBox='0 0 {width} {height}' "
            "xmlns='http://www.w3.org/2000/svg'>",
            "<defs>",
            "<linearGradient id='bg' x1='0' y1='0' x2='1' y2='1'>",
            "<stop offset='0%' stop-color='#f8fafc'/>",
            "<stop offset='100%' stop-color='#dce8f1'/>",
            "</linearGradient>",
            "</defs>",
            f"<rect x='0' y='0' width='{width}' height='{height}' fill='url(#bg)' rx='24' />",
            "<g id='viewport'>",
            "".join(edge_lines),
            "".join(node_shapes),
            "</g>",
            "</svg>",
        ]
    )


def _render_interactive_map(svg_markup: str, height: int = 520) -> None:
    html_block = f"""
<style>
  .map-shell {{
    border: 1px solid #e6eef5;
    border-radius: 12px;
    overflow: hidden;
    background: #f8fbff;
  }}
  .map-toolbar {{
    display: flex;
    gap: 8px;
    align-items: center;
    padding: 8px 10px;
    border-bottom: 1px solid #e6eef5;
    background: rgba(255,255,255,0.95);
    font-family: sans-serif;
    font-size: 12px;
    color: #334e68;
  }}
  .map-toolbar button {{
    border: 1px solid #cbd7e2;
    background: #fff;
    border-radius: 6px;
    width: 28px;
    height: 28px;
    cursor: pointer;
    font-size: 16px;
  }}
  .map-stage {{
    width: 100%;
    height: {height - 44}px;
    overflow: hidden;
    cursor: grab;
    user-select: none;
  }}
  .map-stage:active {{
    cursor: grabbing;
  }}
  #rpg-map {{
    width: 100%;
    height: 100%;
    touch-action: none;
    display: block;
  }}
</style>
<div class="map-shell">
  <div class="map-toolbar">
    <button id="zoom-in">+</button>
    <button id="zoom-out">-</button>
    <button id="zoom-reset" style="width:auto;padding:0 8px;font-size:12px;">重置</button>
    <span>滚轮缩放，按住拖拽移动地图</span>
  </div>
  <div class="map-stage" id="map-stage">
    {svg_markup}
  </div>
</div>
<script>
(() => {{
  const stage = document.getElementById("map-stage");
  const svg = document.getElementById("rpg-map");
  const viewport = document.getElementById("viewport");
  if (!stage || !svg || !viewport) return;

  let scale = 1;
  let tx = 0;
  let ty = 0;
  let dragging = false;
  let lastX = 0;
  let lastY = 0;

  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
  const apply = () => {{
    viewport.setAttribute("transform", `translate(${{tx}} ${{ty}}) scale(${{scale}})`);
  }};
  const zoomBy = (factor) => {{
    scale = clamp(scale * factor, 0.45, 4.2);
    apply();
  }};
  const reset = () => {{
    scale = 1;
    tx = 0;
    ty = 0;
    apply();
  }};

  document.getElementById("zoom-in")?.addEventListener("click", () => zoomBy(1.15));
  document.getElementById("zoom-out")?.addEventListener("click", () => zoomBy(0.87));
  document.getElementById("zoom-reset")?.addEventListener("click", reset);

  stage.addEventListener("wheel", (event) => {{
    event.preventDefault();
    zoomBy(event.deltaY < 0 ? 1.08 : 0.92);
  }}, {{ passive: false }});

  stage.addEventListener("mousedown", (event) => {{
    dragging = true;
    lastX = event.clientX;
    lastY = event.clientY;
  }});
  window.addEventListener("mousemove", (event) => {{
    if (!dragging) return;
    const dx = event.clientX - lastX;
    const dy = event.clientY - lastY;
    tx += dx;
    ty += dy;
    lastX = event.clientX;
    lastY = event.clientY;
    apply();
  }});
  window.addEventListener("mouseup", () => {{
    dragging = false;
  }});
  stage.addEventListener("mouseleave", () => {{
    dragging = false;
  }});

  apply();
}})();
</script>
"""
    components.html(html_block, height=height, scrolling=False)


def _render_map_panel(world: WorldSpec, state: GameState, session_id: str, sessions_root: Path) -> None:
    st.header("地图")
    st.write(f"Session：{session_id}")
    _render_interactive_map(_svg_map(world, state), height=520)

    current = world.get_location(state.player_location)
    if not current:
        return
    st.markdown(f"**当前位置**：{current.name} ({current.kind})")
    st.caption(current.description)

    neighbor_ids = [loc_id for loc_id in current.connected_to if world.get_location(loc_id)]
    st.markdown("**可前往地点**")
    if not neighbor_ids:
        st.write("当前没有已连接地点。")
    else:
        cols = st.columns(max(1, min(3, len(neighbor_ids))))
        for idx, neighbor_id in enumerate(neighbor_ids):
            target = world.get_location(neighbor_id)
            label = target.name if target else neighbor_id
            with cols[idx % len(cols)]:
                if st.button(f"前往 {label}", key=f"go_{session_id}_{state.last_turn_id}_{neighbor_id}"):
                    state.player_location = neighbor_id
                    save_state(session_id, state, sessions_root)
                    st.rerun()


def _quest_suggested_location(world: WorldSpec, quest_id: str) -> str | None:
    if world.main_quest and world.main_quest.quest_id == quest_id:
        return world.main_quest.suggested_location
    for quest in world.side_quests:
        if quest.quest_id == quest_id:
            return quest.suggested_location
    return None


def _base_resource_template(world: WorldSpec, loc: LocationSpec, prefer_chinese: bool) -> dict[str, int]:
    return suggest_location_resource_template(world, loc, prefer_chinese=prefer_chinese)


def _initial_resource_stock(state: GameState, loc: LocationSpec, prefer_chinese: bool) -> dict[str, int]:
    stock = dict(_base_resource_template(state.world, loc, prefer_chinese))
    for quest_id, quest in state.quest_journal.items():
        if quest.category != "side":
            continue
        if quest.status == "completed":
            continue
        suggested = _quest_suggested_location(state.world, quest_id)
        if suggested and suggested != loc.location_id:
            continue
        for item, need in quest.required_items.items():
            have = quest.collected_items.get(item, 0)
            missing = max(0, int(need) - int(have))
            if missing <= 0:
                continue
            stock[item] = max(int(stock.get(item, 0)), min(6, missing + 1))
    return {item: max(0, int(count)) for item, count in stock.items()}


def _ensure_location_stock(state: GameState, loc: LocationSpec, prefer_chinese: bool) -> bool:
    loc_id = loc.location_id
    existing = state.location_resource_stock.get(loc_id)
    target = _initial_resource_stock(state, loc, prefer_chinese)
    if not isinstance(existing, dict):
        state.location_resource_stock[loc_id] = {str(item): max(0, int(count)) for item, count in target.items()}
        return True
    normalized = {str(k): max(0, int(v)) for k, v in existing.items()}
    changed = False
    # Keep depletion persistent; only seed NEW item types that were not present before.
    for item, count in target.items():
        if item in normalized:
            continue
        normalized[item] = max(0, int(count))
        changed = True
    if not normalized and target:
        normalized = {str(item): max(0, int(count)) for item, count in target.items()}
        changed = True
    state.location_resource_stock[loc_id] = normalized
    return changed


def _collect_location_resources(
    state: GameState,
    loc: LocationSpec,
    prefer_chinese: bool,
    item_name: str,
    quantity: int,
) -> tuple[dict[str, int], dict[str, int]]:
    _ensure_location_stock(state, loc, prefer_chinese)
    stock = state.location_resource_stock.get(loc.location_id, {})
    item_stock = int(stock.get(item_name, 0))
    if item_stock <= 0 or quantity <= 0:
        return {}, stock
    take = min(item_stock, int(quantity))
    stock[item_name] = item_stock - take
    state.inventory[item_name] = int(state.inventory.get(item_name, 0)) + take
    delta: dict[str, int] = {item_name: take}
    state.location_resource_stock[loc.location_id] = stock
    return delta, stock


def _inventory_delta_text(delta: dict[str, int], prefer_chinese: bool) -> str:
    parts = []
    for name, amount in sorted(delta.items()):
        sign = "+" if amount >= 0 else ""
        display = _display_item_name(name, prefer_chinese)
        parts.append(f"{display} {sign}{amount}")
    return "，".join(parts)


def _localized_reason(reason: str, prefer_chinese: bool) -> str:
    text = str(reason or "")
    if not prefer_chinese:
        return text
    lowered = text.lower()
    mapping = {
        "refused: guarding their post": "拒绝：必须留守岗位",
        "refused: too risky": "拒绝：风险过高",
        "refused: doesn't trust the player": "拒绝：不信任玩家",
        "refused: too stubborn": "拒绝：过于固执",
        "refused: unwilling to comply": "拒绝：不愿配合",
        "unknown npc": "未知 NPC",
        "ok": "已通过",
    }
    if lowered in mapping:
        return mapping[lowered]
    if lowered.startswith("refused:"):
        return "拒绝：" + text.split(":", 1)[-1].strip()
    return text


def _remaining_stock_text(stock: dict[str, int], prefer_chinese: bool) -> str:
    remaining = [(name, int(count)) for name, count in stock.items() if int(count) > 0]
    if not remaining:
        return "该地点资源已采尽"
    remaining.sort(key=lambda pair: (pair[0]))
    parts = [f"{_display_item_name(name, prefer_chinese)} x{count}" for name, count in remaining]
    return "，".join(parts)


def _quest_notices(before: GameState, after: GameState, prefer_chinese: bool) -> list[str]:
    notices: list[str] = []
    old = before.quest_journal
    new = after.quest_journal
    for quest_id, quest in new.items():
        if quest_id not in old:
            notices.append(f"新任务：{quest.title}" if prefer_chinese else f"New quest: {quest.title}")
            continue
        old_status = old[quest_id].status
        if old_status != quest.status:
            if quest.status == "completed":
                notices.append(f"任务完成：{quest.title}" if prefer_chinese else f"Quest completed: {quest.title}")
            else:
                notices.append(
                    f"任务更新：{quest.title} -> {quest.status}"
                    if prefer_chinese
                    else f"Quest updated: {quest.title} -> {quest.status}"
                )
    return notices


def _state_diff_notices(
    before: GameState,
    after: GameState,
    world: WorldSpec,
    prefer_chinese: bool,
) -> tuple[list[str], list[str], list[str]]:
    move_lines = []
    for npc in world.npcs:
        old_loc = before.npc_locations.get(npc.npc_id)
        new_loc = after.npc_locations.get(npc.npc_id)
        if old_loc and new_loc and old_loc != new_loc:
            old_name = world.get_location(old_loc).name if world.get_location(old_loc) else old_loc
            new_name = world.get_location(new_loc).name if world.get_location(new_loc) else new_loc
            move_lines.append(f"{npc.name}：{old_name} -> {new_name}")

    quest_lines = _quest_notices(before, after, prefer_chinese)
    inventory_delta: dict[str, int] = {}
    keys = set(before.inventory.keys()) | set(after.inventory.keys())
    for key in keys:
        delta = int(after.inventory.get(key, 0)) - int(before.inventory.get(key, 0))
        if delta != 0:
            inventory_delta[key] = delta
    inv_lines = [_inventory_delta_text(inventory_delta, prefer_chinese)] if inventory_delta else []
    return move_lines, quest_lines, inv_lines


def _render_quests(state: GameState, prefer_chinese: bool) -> None:
    st.markdown("**任务日志**")
    if not state.quest_journal:
        st.write("暂无任务。")
        return

    main = state.quest_journal.get(state.main_quest_id) if state.main_quest_id else None
    if main:
        st.markdown(f"**主线**：{main.title} [{main.status}]")
        st.write(main.objective or "暂无描述。")
        if main.guidance:
            st.caption(main.guidance)
        if main.status != "completed":
            st.caption(_main_trial_target_text(state, prefer_chinese))
        if main.required_items:
            req = []
            for item, need in main.required_items.items():
                have = main.collected_items.get(item, 0)
                req.append(f"{_display_item_name(item, prefer_chinese)} {have}/{need}")
            st.caption("主线需求：" + "，".join(req))

    side_items = [q for qid, q in state.quest_journal.items() if qid != state.main_quest_id]
    for quest in side_items:
        st.markdown(f"- {quest.title} [{quest.status}]")
        if quest.guidance:
            st.caption(quest.guidance)
        if quest.required_items:
            req = []
            for item, need in quest.required_items.items():
                have = quest.collected_items.get(item, 0)
                req.append(f"{_display_item_name(item, prefer_chinese)} {have}/{need}")
            st.caption("需求：" + "，".join(req))


def _collect_chat_messages(world: WorldSpec, logs: list[dict], npc_id: str) -> list[tuple[str, str]]:
    npc_name_map = {npc.npc_id: npc.name for npc in world.npcs}
    prefer_chinese = _prefer_chinese_ui(world)
    messages: list[tuple[str, str]] = []
    for record in logs:
        if record.get("npc_id") != npc_id:
            continue
        player_text = (record.get("player_text") or "").strip()
        if player_text:
            messages.append(("player", player_text))
        output = record.get("output", {})
        for line in output.get("npc_dialogue", []):
            line_npc = line.get("npc_id", "")
            text = (line.get("text") or "").strip()
            if not text:
                continue
            if line_npc and line_npc != npc_id:
                continue
            npc_name = npc_name_map.get(npc_id, npc_id)
            messages.append(("npc", f"{npc_name}：{text}"))
        narration = (output.get("narration") or "").strip()
        if narration:
            prefix = "旁白" if prefer_chinese else "Narration"
            messages.append(("system", f"{prefix}: {narration}"))
        for rejection in record.get("move_rejections", []):
            reason = rejection.get("reason", "rejected")
            reason_text = _localized_reason(str(reason), prefer_chinese)
            if prefer_chinese:
                messages.append(("system", f"系统：移动被拒绝（{reason_text}）"))
            else:
                messages.append(("system", f"System: move rejected ({reason_text})"))
        for refusal in record.get("move_refusals", []):
            reason = refusal.get("reason", "refused")
            reason_text = _localized_reason(str(reason), prefer_chinese)
            if prefer_chinese:
                messages.append(("system", f"系统：NPC 拒绝（{reason_text}）"))
            else:
                messages.append(("system", f"System: NPC refused ({reason_text})"))
    return messages


def _render_chat_window(world: WorldSpec, logs: list[dict], npc_id: str | None, npc_name: str | None) -> None:
    if not npc_id or not npc_name:
        st.info("请选择一个 NPC 查看聊天记录。")
        return

    messages = _collect_chat_messages(world, logs, npc_id)
    if not messages:
        st.info(f"与 {npc_name} 暂无对话记录。")
        return

    body_parts = []
    for role, text in messages:
        klass = "msg-player" if role == "player" else "msg-system" if role == "system" else "msg-npc"
        body_parts.append(f"<div class='msg {klass}'>{html.escape(text)}</div>")

    html_block = f"""
<style>
  .chat-frame {{
    border: 1px solid #d8e2eb;
    border-radius: 12px;
    background: rgba(248, 251, 255, 0.9);
    padding: 10px;
    height: 460px;
    overflow-y: auto;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  .msg {{
    padding: 10px 12px;
    border-radius: 10px;
    margin: 8px 0;
    max-width: 90%;
    line-height: 1.45;
    white-space: pre-wrap;
  }}
  .msg-player {{
    background: #d7f1ff;
    margin-left: auto;
    text-align: right;
  }}
  .msg-npc {{
    background: #e7f6e7;
    margin-right: auto;
  }}
  .msg-system {{
    background: #eef2f7;
    margin: 8px auto;
    color: #4b5d70;
    font-size: 13px;
    text-align: center;
  }}
</style>
<div class="chat-frame" id="chat-frame">
  {''.join(body_parts)}
</div>
<script>
  const frame = document.getElementById("chat-frame");
  if (frame) {{
    frame.scrollTop = frame.scrollHeight;
  }}
</script>
"""
    components.html(html_block, height=500, scrolling=False)


def _npc_name(world: WorldSpec, npc_id: str) -> str:
    for npc in world.npcs:
        if npc.npc_id == npc_id:
            return npc.name
    return npc_id


def _build_delivery_reply(
    *,
    world: WorldSpec,
    npc_id: str,
    delivered_delta: Dict[str, int],
    reward_delta: Dict[str, int],
    notices: list[str],
    prefer_chinese: bool,
) -> str:
    npc_name = _npc_name(world, npc_id)
    delivered_text = _inventory_delta_text(delivered_delta, prefer_chinese)
    if not prefer_chinese:
        if notices:
            notice_text = "; ".join(notices)
            if reward_delta:
                reward_text = _inventory_delta_text(reward_delta, prefer_chinese)
                return (
                    f"{npc_name} checks the delivered supplies ({delivered_text}) and nods. "
                    f"{notice_text}. Here is your reward: {reward_text}."
                )
            return f"{npc_name} checks the delivered supplies ({delivered_text}). {notice_text}."
        if reward_delta:
            reward_text = _inventory_delta_text(reward_delta, prefer_chinese)
            return (
                f"{npc_name} accepts your delivery ({delivered_text}) and confirms the objective. "
                f"Here is your reward: {reward_text}."
            )
        return f"{npc_name} accepts your delivery ({delivered_text}), but the quest is not fully satisfied yet."
    if notices:
        notice_text = "；".join(notices)
        if reward_delta:
            reward_text = _inventory_delta_text(reward_delta, prefer_chinese)
            return (
                f"{npc_name}接过你交付的物资（{delivered_text}），认真核对后点头。"
                f"{notice_text}。这是你的奖励：{reward_text}。"
            )
        return f"{npc_name}清点了你交付的物资（{delivered_text}）。{notice_text}。"
    if reward_delta:
        reward_text = _inventory_delta_text(reward_delta, prefer_chinese)
        return (
            f"{npc_name}收下了你交付的物资（{delivered_text}），表示已经足够。"
            f"这是你的奖励：{reward_text}。"
        )
    return f"{npc_name}收下了你交付的物资（{delivered_text}），但任务还未满足全部条件。"


def _append_delivery_log(
    *,
    session_id: str,
    sessions_root: Path,
    state: GameState,
    npc_id: str,
    player_text: str,
    npc_reply: str,
) -> None:
    record = {
        "session_id": session_id,
        "turn_index": state.last_turn_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "player_text": player_text,
        "npc_id": npc_id,
        "location_id": state.player_location,
        "model_used": "system_delivery",
        "output": {
            "narration": "",
            "npc_dialogue": [{"npc_id": npc_id, "text": npc_reply}],
            "world_updates": {
                "player_location": None,
                "npc_moves": [],
                "flags_delta": {},
                "quest_updates": {},
                "quest_progress_updates": [],
                "inventory_delta": {},
            },
            "memory_summary": f"Delivery interaction with {npc_id}: {player_text}",
            "safety": {"refuse": False, "reason": None},
        },
        "move_rejections": [],
        "move_refusals": [],
        "move_applied_count": 0,
        "move_rejected_count": 0,
        "move_refused_count": 0,
        "rag": {"enabled": False, "source": "delivery_ui"},
    }
    append_turn_log(session_id, record, sessions_root)


def _all_side_quests_completed(state: GameState) -> bool:
    world_side_ids = [q.quest_id for q in state.world.side_quests]
    if not world_side_ids:
        return False
    for quest_id in world_side_ids:
        progress = state.quest_journal.get(quest_id)
        if not progress or progress.status != "completed":
            return False
    return True


def _remaining_side_quest_titles(state: GameState) -> list[str]:
    remaining: list[str] = []
    for quest in state.world.side_quests:
        progress = state.quest_journal.get(quest.quest_id)
        if progress and progress.status == "completed":
            continue
        title = progress.title if progress else quest.title
        remaining.append(title)
    return remaining


def _main_trial_target(state: GameState) -> tuple[str | None, str | None]:
    world = state.world
    main_spec = world.main_quest
    if not main_spec:
        return None, None
    npc_id = main_spec.giver_npc_id
    if not npc_id and world.npcs:
        npc_id = world.npcs[0].npc_id
    # NPC may move during gameplay; use realtime npc_locations first.
    loc_id = state.npc_locations.get(npc_id) if npc_id else None
    if not loc_id:
        loc_id = main_spec.suggested_location
    if not loc_id:
        loc_id = state.player_location
    return npc_id, loc_id


def _main_trial_target_text(state: GameState, prefer_chinese: bool) -> str:
    npc_id, loc_id = _main_trial_target(state)
    npc_name = next((npc.name for npc in state.world.npcs if npc.npc_id == npc_id), npc_id or "")
    loc_name = state.world.get_location(loc_id).name if loc_id and state.world.get_location(loc_id) else (loc_id or "")
    if prefer_chinese:
        if npc_name and loc_name:
            return f"终局目标：前往【{loc_name}】并与【{npc_name}】对话。"
        if npc_name:
            return f"终局目标：与【{npc_name}】对话。"
        if loc_name:
            return f"终局目标：前往【{loc_name}】。"
        return "终局目标：等待终局 NPC 线索。"
    if npc_name and loc_name:
        return f"Final target: go to [{loc_name}] and talk to [{npc_name}]."
    if npc_name:
        return f"Final target: talk to [{npc_name}]."
    if loc_name:
        return f"Final target: go to [{loc_name}]."
    return "Final target: waiting for finale NPC clue."


def _can_trigger_main_trial(state: GameState, selected_npc_id: str | None) -> bool:
    if not state.main_quest_id or state.main_quest_id not in state.quest_journal:
        return False
    main = state.quest_journal[state.main_quest_id]
    if main.status == "completed":
        return False
    target_npc, target_loc = _main_trial_target(state)
    if not target_npc or not target_loc:
        return False
    if selected_npc_id != target_npc:
        return False
    if state.player_location != target_loc:
        return False
    if not _all_side_quests_completed(state):
        return False
    return True


def _build_story_summary_fallback(
    world: WorldSpec,
    state: GameState,
    logs: list[dict],
    prefer_chinese: bool,
) -> str:
    recent_talks = [str(r.get("player_text") or "").strip() for r in logs[-8:] if str(r.get("player_text") or "").strip()]
    key_items = ", ".join([f"{_display_item_name(k, prefer_chinese)} x{v}" for k, v in sorted(state.inventory.items())]) or (
        "无" if prefer_chinese else "none"
    )
    main_title = state.quest_journal.get(state.main_quest_id).title if state.main_quest_id and state.main_quest_id in state.quest_journal else world.initial_quest
    if prefer_chinese:
        parts = [
            f"在《{world.title}》中，你从{world.starting_hook}出发，逐步推进主线“{main_title}”。",
            f"你在旅途中与多个 NPC 交流，并完成关键支线，最终整备了决战所需物资：{key_items}。",
            "经过最终考验，你成功通过终章挑战，世界危机得以解除。",
        ]
        if recent_talks:
            parts.append("旅程中的关键抉择包括：" + "；".join(recent_talks[-3:]) + "。")
        return "\n\n".join(parts)
    parts = [
        f"In '{world.title}', you began with: {world.starting_hook}, then advanced the main arc '{main_title}'.",
        f"Through dialogue and side quests, you prepared key resources for the finale: {key_items}.",
        "You passed the final trial and resolved the central world crisis.",
    ]
    if recent_talks:
        parts.append("Key choices included: " + " | ".join(recent_talks[-3:]) + ".")
    return "\n\n".join(parts)


def _build_story_summary(
    cfg,
    world: WorldSpec,
    state: GameState,
    logs: list[dict],
    prefer_chinese: bool,
    has_api_key: bool,
) -> str:
    fallback = _build_story_summary_fallback(world, state, logs, prefer_chinese)
    if not has_api_key:
        return fallback
    try:
        llm = QwenOpenAICompatibleClient(cfg)
        language = "Chinese" if prefer_chinese else "English"
        snippet_lines = []
        for record in logs[-20:]:
            player_text = str(record.get("player_text") or "").strip()
            output = record.get("output") or {}
            narration = str((output.get("narration") if isinstance(output, dict) else "") or "").strip()
            npc_lines = output.get("npc_dialogue", []) if isinstance(output, dict) else []
            if player_text:
                snippet_lines.append(f"P: {player_text}")
            for line in npc_lines:
                text = str((line or {}).get("text") or "").strip()
                if text:
                    snippet_lines.append(f"N: {text}")
            if narration:
                snippet_lines.append(f"S: {narration}")
        convo = "\n".join(snippet_lines[-40:])
        system = "You are a narrative summarizer for RPG session recaps."
        user = (
            f"Write the recap in {language} only.\n"
            "Use 2-4 short paragraphs, coherent and story-like.\n"
            "Do not output JSON, markdown list, or code.\n\n"
            f"World title: {world.title}\n"
            f"Starting hook: {world.starting_hook}\n"
            f"Initial quest: {world.initial_quest}\n"
            f"Current inventory: {json.dumps(state.inventory, ensure_ascii=False)}\n"
            f"Quest journal: {json.dumps({k: v.model_dump() for k, v in state.quest_journal.items()}, ensure_ascii=False)}\n"
            f"Recent dialogue and narration:\n{convo}\n"
        )
        text = llm.generate_text(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.4,
            top_p=0.9,
        ).strip()
        if text:
            return text
    except Exception:
        pass
    return fallback


def _render_story_history_panel(sessions_root: Path) -> None:
    st.markdown("---")
    st.subheader("游玩历史总结")
    history = read_story_summaries(sessions_root, limit=30)
    if not history:
        st.caption("暂无历史总结。")
        return
    for idx, rec in enumerate(history):
        world_title = rec.get("world_title") or rec.get("world_id") or "Unknown World"
        timestamp = rec.get("timestamp") or rec.get("created_at") or ""
        title = f"{world_title} | {timestamp}" if timestamp else str(world_title)
        with st.expander(title, expanded=(idx == 0)):
            st.write(rec.get("summary") or "")
            sid = rec.get("session_id")
            if sid:
                st.caption(f"session_id: {sid}")


def _render_story_summary_page(summary_record: dict, prefer_chinese: bool) -> None:
    st.header("终局总结" if prefer_chinese else "Final Recap")
    st.markdown(f"**{summary_record.get('world_title', '')}**")
    st.write(summary_record.get("summary", ""))

cfg = load_config("configs/config.yaml")
sessions_root = default_sessions_root(cfg)
api_key_env = cfg.llm.api_key_env or "DASHSCOPE_API_KEY"
has_api_key = bool(os.getenv(api_key_env))

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "move_notice" not in st.session_state:
    st.session_state.move_notice = []
if "quest_notice" not in st.session_state:
    st.session_state.quest_notice = []
if "inventory_notice" not in st.session_state:
    st.session_state.inventory_notice = []
if "summary_view" not in st.session_state:
    st.session_state.summary_view = False
if "summary_record" not in st.session_state:
    st.session_state.summary_record = None

st.title("Adaptive RPG Storytelling")

with st.sidebar:
    st.header("连接状态")
    st.write(f"API Key 已设置：{'是' if has_api_key else '否'}")
    st.write(f"模型：{cfg.llm.model}")
    st.write(f"Base URL：{cfg.llm.base_url}")
    if st.button("从磁盘重新加载"):
        st.rerun()

if not st.session_state.session_id:
    st.session_state.summary_view = False
    st.session_state.summary_record = None
    st.header("创建世界")
    world_prompt = st.text_area(
        "世界设定（World Prompt）",
        placeholder="例如：中世纪王国、巨龙威胁、边境小镇。",
        height=120,
    )
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Create World"):
            if not world_prompt.strip():
                st.error("请输入世界设定。")
            elif not has_api_key:
                st.error(f"未检测到 API Key（{api_key_env}）。")
            else:
                progress = st.progress(0)
                status = st.empty()
                try:
                    status.info("正在初始化模型连接...")
                    progress.progress(15)
                    llm = QwenOpenAICompatibleClient(cfg)
                    status.info("正在生成世界，请稍候...")
                    progress.progress(35)
                    with st.spinner("世界生成中..."):
                        session_id = generate_session_id()
                        world = generate_world_spec(cfg, llm, world_prompt.strip())
                    progress.progress(75)
                    status.info("正在初始化会话与存档...")
                    state = initialize_game_state(world, session_id=session_id)
                    save_state(session_id, state, sessions_root)
                    _persist_world(cfg, session_id, world)
                    progress.progress(100)
                    status.success("世界生成完成")
                    st.session_state.session_id = session_id
                    st.success(f"世界已创建，session_id: {session_id}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"世界生成失败：{exc}")
    with col_b:
        load_id = st.text_input("加载 session_id")
        if st.button("Load Session"):
            if load_id:
                try:
                    _ = load_state(load_id, sessions_root)
                    st.session_state.session_id = load_id
                    st.success(f"已加载 session {load_id}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"加载失败：{exc}")
    _render_story_history_panel(sessions_root)
else:
    session_id = st.session_state.session_id
    try:
        state = load_state(session_id, sessions_root)
    except Exception as exc:
        st.error(f"无法加载会话：{exc}")
        st.session_state.session_id = None
        st.stop()

    world = _load_world_for_session(cfg, session_id, state.world)
    state = state.model_copy(deep=True)
    state.world = world
    synced_state = sync_quest_journal(state)
    if synced_state.model_dump() != state.model_dump():
        state = synced_state
        save_state(session_id, state, sessions_root)
    prefer_chinese = _prefer_chinese_ui(world)
    trial_pending_key = f"final_trial_pending_{session_id}"
    if trial_pending_key not in st.session_state:
        st.session_state[trial_pending_key] = False

    if st.session_state.summary_view and isinstance(st.session_state.summary_record, dict):
        _render_story_summary_page(st.session_state.summary_record, prefer_chinese)
        col_back_a, col_back_b = st.columns([1, 1])
        with col_back_a:
            if st.button("返回创建页"):
                st.session_state.session_id = None
                st.session_state.summary_view = False
                st.session_state.summary_record = None
                st.rerun()
        with col_back_b:
            if st.button("返回当前会话"):
                st.session_state.summary_view = False
                st.rerun()
        st.stop()

    col_left, col_mid, col_right = st.columns([1.25, 2.25, 1.15])
    npcs_here = [npc for npc in world.npcs if state.npc_locations.get(npc.npc_id) == state.player_location]
    logs = read_turn_logs(session_id, sessions_root)

    with col_left:
        _render_map_panel(world, state, session_id, sessions_root)

        st.subheader("当前 NPC")
        if npcs_here:
            for npc in npcs_here:
                st.write(f"- {npc.name} ({npc.profession})")
        else:
            st.write("无 NPC")

        st.subheader("背包")
        if state.inventory:
            for item, count in sorted(state.inventory.items()):
                st.write(f"- {_display_item_name(item, prefer_chinese)} x{count}")
        else:
            st.write("背包为空")

        current_loc = world.get_location(state.player_location)
        if current_loc:
            stock_changed = _ensure_location_stock(state, current_loc, prefer_chinese)
            if stock_changed:
                save_state(session_id, state, sessions_root)
            stock = state.location_resource_stock.get(current_loc.location_id, {})
            selectable = [(name, int(count)) for name, count in stock.items() if int(count) > 0]
            if selectable:
                selectable.sort(key=lambda pair: (_display_item_name(pair[0], prefer_chinese), pair[1]))
                item_names = [name for name, _ in selectable]
                selected_item = st.selectbox(
                    "选择要采集的物品",
                    options=item_names,
                    format_func=lambda key: f"{_display_item_name(key, prefer_chinese)}（剩余{stock.get(key, 0)}）",
                    key=f"collect_item_{session_id}_{current_loc.location_id}",
                )
                max_qty = max(1, int(stock.get(selected_item, 1)))
                qty = int(
                    st.number_input(
                        "采集数量",
                        min_value=1,
                        max_value=max_qty,
                        value=1,
                        step=1,
                        key=f"collect_qty_{session_id}_{current_loc.location_id}",
                    )
                )
                if st.button("采集选中物品"):
                    old_state = state.model_copy(deep=True)
                    delta, stock_after = _collect_location_resources(
                        state, current_loc, prefer_chinese, selected_item, qty
                    )
                    if not delta:
                        st.warning("该地点已没有可收集的资源。")
                    else:
                        state = sync_quest_journal(state)
                        save_state(session_id, state, sessions_root)
                        _, quest_lines, inv_lines = _state_diff_notices(
                            old_state, state, world, prefer_chinese
                        )
                        st.session_state.quest_notice = quest_lines
                        st.session_state.inventory_notice = inv_lines or [
                            _inventory_delta_text(delta, prefer_chinese)
                        ]
                        if all(int(v) <= 0 for v in stock_after.values()):
                            st.session_state.inventory_notice.append("该地点资源已采尽")
                        st.rerun()
            else:
                st.info("该地点已没有可收集的资源。")
            st.caption("本地剩余资源：" + _remaining_stock_text(stock, prefer_chinese))

        st.subheader("RAG 调试")
        if logs:
            last_rag = logs[-1].get("rag", {})
            if st.checkbox("显示最近一次 RAG IDs"):
                st.write("always_include_ids:", last_rag.get("always_include_ids", []))
                st.write("retrieved_ids:", last_rag.get("retrieved_ids", []))
        else:
            st.write("暂无日志")

    with col_mid:
        st.header("对话")
        for msg in st.session_state.move_notice:
            if hasattr(st, "toast"):
                st.toast(f"NPC 移动：{msg}")
            st.info(f"NPC 移动：{msg}")
        for msg in st.session_state.quest_notice:
            st.success(msg)
        for msg in st.session_state.inventory_notice:
            st.info(f"背包更新：{msg}")
        st.session_state.move_notice = []
        st.session_state.quest_notice = []
        st.session_state.inventory_notice = []

        loc = world.get_location(state.player_location)
        if loc:
            st.markdown(f"**地点**：{loc.name}")
            st.write(loc.description)

        npc_options = {npc.npc_id: npc.name for npc in npcs_here}
        selected_npc_name = None
        selected_npc_id = None
        if npc_options:
            npc_ids = list(npc_options.keys())
            selected_npc_id = st.selectbox(
                "聊天对象",
                options=npc_ids,
                format_func=lambda npc_id: npc_options.get(npc_id, npc_id),
                key=f"chat_target_{session_id}_{state.player_location}",
            )
            selected_npc_name = npc_options.get(selected_npc_id, selected_npc_id)

        _render_chat_window(world, logs, selected_npc_id, selected_npc_name)

        st.markdown("**交付任务物品**")
        if not selected_npc_id or not selected_npc_name:
            st.info("请选择要交付的 NPC。")
        elif not any(n.npc_id == selected_npc_id for n in npcs_here):
            st.info("该 NPC 不在当前地点，无法交付。")
        elif not state.inventory:
            st.info("背包为空，暂无可交付物品。")
        else:
            deliver_items = sorted(state.inventory.keys(), key=lambda item: _display_item_name(item, prefer_chinese))
            deliver_item = st.selectbox(
                "交付物品",
                options=deliver_items,
                format_func=lambda item: f"{_display_item_name(item, prefer_chinese)}（背包{state.inventory.get(item, 0)}）",
                key=f"deliver_item_{session_id}",
            )
            max_deliver = max(1, int(state.inventory.get(deliver_item, 1)))
            deliver_qty = int(
                st.number_input(
                    "交付数量",
                    min_value=1,
                    max_value=max_deliver,
                    value=1,
                    step=1,
                    key=f"deliver_qty_{session_id}",
                )
            )
            if st.button("交付给当前 NPC"):
                old_state = state.model_copy(deep=True)
                new_state, delivery_notices, reward_delta, delivered_delta = deliver_items_to_npc(
                    state,
                    selected_npc_id,
                    state.player_location,
                    {deliver_item: deliver_qty},
                )
                if not delivered_delta:
                    st.warning("该 NPC 当前没有需要你交付的此类物品。")
                else:
                    new_state = new_state.model_copy(deep=True)
                    new_state.last_turn_id = int(new_state.last_turn_id) + 1
                    reply = _build_delivery_reply(
                        world=world,
                        npc_id=selected_npc_id,
                        delivered_delta=delivered_delta,
                        reward_delta=reward_delta,
                        notices=delivery_notices,
                        prefer_chinese=prefer_chinese,
                    )
                    player_delivery_text = (
                        "我交付了：" if prefer_chinese else "I delivered: "
                    ) + _inventory_delta_text(delivered_delta, prefer_chinese)
                    _append_delivery_log(
                        session_id=session_id,
                        sessions_root=sessions_root,
                        state=new_state,
                        npc_id=selected_npc_id,
                        player_text=player_delivery_text,
                        npc_reply=reply,
                    )
                    save_state(session_id, new_state, sessions_root)

                    moved_lines, quest_lines, inv_lines = _state_diff_notices(
                        old_state, new_state, world, prefer_chinese
                    )
                    quest_lines.extend(delivery_notices)
                    if reward_delta:
                        prefix = "支线奖励发放：" if prefer_chinese else "Side reward granted: "
                        quest_lines.append(prefix + _inventory_delta_text(reward_delta, prefer_chinese))
                    if _can_trigger_main_trial(new_state, selected_npc_id):
                        st.session_state[trial_pending_key] = True
                        quest_lines.append("终局 NPC 已准备好，是否发起最终考验？" if prefer_chinese else "The finale NPC is ready. Start the final trial?")
                    st.session_state.move_notice = moved_lines
                    st.session_state.quest_notice = quest_lines
                    st.session_state.inventory_notice = inv_lines
                    st.rerun()

        if _can_trigger_main_trial(state, selected_npc_id):
            st.markdown("**最终考验**" if prefer_chinese else "**Final Trial**")
            if st.session_state.get(trial_pending_key, False):
                st.warning(
                    "你已满足支线前置条件。是否现在接受终局考验？"
                    if prefer_chinese
                    else "You completed side prerequisites. Do you want to start the final trial now?"
                )
                c_yes, c_no = st.columns([1, 1])
                with c_yes:
                    if st.button("参加最终考验" if prefer_chinese else "Start Final Trial", key=f"trial_yes_{session_id}"):
                        ready, progress = evaluate_main_trial_readiness(state)
                        if not ready:
                            updated_state = resolve_main_trial(state, passed=False)
                            save_state(session_id, updated_state, sessions_root)
                            missing = []
                            for item, info in progress.items():
                                if int(info.get("have", 0)) < int(info.get("need", 0)):
                                    missing.append(
                                        f"{_display_item_name(item, prefer_chinese)} {info.get('have', 0)}/{info.get('need', 0)}"
                                    )
                            st.session_state.quest_notice = [
                                ("最终考验失败，缺少：" if prefer_chinese else "Final trial failed. Missing: ")
                                + ("，".join(missing) if prefer_chinese else ", ".join(missing))
                            ]
                            st.session_state[trial_pending_key] = False
                            st.rerun()
                        updated_state = resolve_main_trial(state, passed=True)
                        save_state(session_id, updated_state, sessions_root)
                        logs_now = read_turn_logs(session_id, sessions_root)
                        summary_text = _build_story_summary(
                            cfg=cfg,
                            world=world,
                            state=updated_state,
                            logs=logs_now,
                            prefer_chinese=prefer_chinese,
                            has_api_key=has_api_key,
                        )
                        summary_record = {
                            "session_id": session_id,
                            "world_id": world.world_id,
                            "world_title": world.title,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "summary": summary_text,
                            "language": "zh" if prefer_chinese else "en",
                        }
                        append_story_summary(summary_record, sessions_root)
                        st.session_state.summary_record = summary_record
                        st.session_state.summary_view = True
                        st.session_state[trial_pending_key] = False
                        st.rerun()
                with c_no:
                    if st.button("暂不参加" if prefer_chinese else "Not now", key=f"trial_no_{session_id}"):
                        st.session_state[trial_pending_key] = False
                        st.rerun()
            else:
                if not _all_side_quests_completed(state):
                    remaining = _remaining_side_quest_titles(state)
                    if remaining:
                        st.info(
                            ("尚未完成支线：" + "，".join(remaining))
                            if prefer_chinese
                            else ("Incomplete side quests: " + ", ".join(remaining))
                        )
                    else:
                        st.info(
                            "请先完成全部支线任务。"
                            if prefer_chinese
                            else "Please complete all side quests first."
                        )
                st.info(_main_trial_target_text(state, prefer_chinese))

        player_text = st.text_input("你的输入")
        if st.button("Send") and player_text:
            if not selected_npc_id or not selected_npc_name:
                st.error("请先选择 NPC。")
            elif not any(n.npc_id == selected_npc_id for n in npcs_here):
                st.error("该 NPC 当前不在你所在地点，请先移动到对应地点。")
            elif not has_api_key:
                st.error(f"未检测到 API Key（{api_key_env}）。")
            else:
                progress = st.progress(0)
                status = st.empty()
                try:
                    status.info("正在请求模型生成对话...")
                    progress.progress(20)
                    llm = QwenOpenAICompatibleClient(cfg)
                    pipeline = TurnPipeline(cfg=cfg, llm_client=llm, sessions_root=sessions_root)
                    before_state = state.model_copy(deep=True)
                    with st.spinner("对话生成中..."):
                        updated_state, _output, _log = pipeline.run_turn(
                            state, player_text, selected_npc_id
                        )
                    progress.progress(85)
                    moved_lines, quest_lines, inv_lines = _state_diff_notices(
                        before_state, updated_state, world, prefer_chinese
                    )
                    if _can_trigger_main_trial(updated_state, selected_npc_id):
                        st.session_state[trial_pending_key] = True
                        quest_lines.append("终局 NPC 已准备好，是否发起最终考验？" if prefer_chinese else "The finale NPC is ready. Start the final trial?")
                    st.session_state.move_notice = moved_lines
                    st.session_state.quest_notice = quest_lines
                    st.session_state.inventory_notice = inv_lines
                    progress.progress(100)
                    status.success("对话生成完成")
                    st.rerun()
                except Exception as exc:
                    st.error(f"对话失败：{exc}")

    with col_right:
        st.header("任务系统")
        st.markdown("**故事背景**")
        st.write(world.starting_hook)
        st.markdown("**主目标**")
        if state.main_quest_id and state.main_quest_id in state.quest_journal:
            st.write(state.quest_journal[state.main_quest_id].objective or world.initial_quest)
        else:
            st.write(world.initial_quest)
        st.markdown("---")
        _render_quests(state, prefer_chinese)
