"""Streamlit UI for world creation and RPG gameplay."""
from __future__ import annotations

from pathlib import Path
import json
import os
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rpg_story.config import load_config
from rpg_story.engine.orchestrator import TurnPipeline
from rpg_story.llm.client import QwenOpenAICompatibleClient
from rpg_story.models.world import WorldSpec, GameState
from rpg_story.persistence.store import (
    generate_session_id,
    load_state,
    save_state,
    read_turn_logs,
    default_sessions_root,
)
from rpg_story.world.generator import generate_world_spec, initialize_game_state


st.set_page_config(page_title="RPG Story Prototype", layout="wide")


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


def _render_log_history(world: WorldSpec, logs: list[dict], limit: int = 10) -> None:
    npc_name_map = {npc.npc_id: npc.name for npc in world.npcs}
    for record in logs[-limit:]:
        player_text = record.get("player_text", "")
        output = record.get("output", {})
        narration = output.get("narration", "")
        npc_dialogue = output.get("npc_dialogue", [])
        move_rejections = record.get("move_rejections", [])
        move_refusals = record.get("move_refusals", [])
        if player_text:
            st.markdown(f"**玩家**：{player_text}")
        if npc_dialogue:
            for line in npc_dialogue:
                npc_id = line.get("npc_id", "")
                name = npc_name_map.get(npc_id, npc_id or "NPC")
                text = line.get("text", "")
                if text:
                    st.markdown(f"**{name}**：{text}")
        if narration:
            st.markdown(f"**旁白**：{narration}")
        for rejection in move_rejections:
            reason = rejection.get("reason", "rejected")
            st.markdown(f"**系统**：移动被拒绝（{reason}）")
        for refusal in move_refusals:
            reason = refusal.get("reason", "refused")
            st.markdown(f"**系统**：NPC 拒绝（{reason}）")
        st.markdown("---")


cfg = load_config("configs/config.yaml")
sessions_root = default_sessions_root(cfg)
api_key_env = cfg.llm.api_key_env or "DASHSCOPE_API_KEY"
has_api_key = bool(os.getenv(api_key_env))

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "move_notice" not in st.session_state:
    st.session_state.move_notice = []

st.title("Adaptive RPG Storytelling")

with st.sidebar:
    st.header("连接状态")
    st.write(f"API Key 已设置：{'是' if has_api_key else '否'}")
    st.write(f"模型：{cfg.llm.model}")
    st.write(f"Base URL：{cfg.llm.base_url}")
    if st.button("从磁盘重新加载"):
        st.rerun()

if not st.session_state.session_id:
    st.header("创建世界")
    world_prompt = st.text_area(
        "世界设定（World Prompt）",
        placeholder="例如：河边的小镇、古桥与森林，低魔设定。",
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
                try:
                    llm = QwenOpenAICompatibleClient(cfg)
                    session_id = generate_session_id()
                    world = generate_world_spec(cfg, llm, world_prompt.strip())
                    state = initialize_game_state(world, session_id=session_id)
                    save_state(session_id, state, sessions_root)
                    _persist_world(cfg, session_id, world)
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

    col_left, col_main = st.columns([1, 3])

    with col_left:
        st.header("地图")
        st.write(f"Session：{session_id}")
        loc_map = {loc.name: loc.location_id for loc in world.locations}
        loc_names = list(loc_map.keys())
        current_name = next((n for n, lid in loc_map.items() if lid == state.player_location), loc_names[0])
        selected_name = st.selectbox("当前位置", options=loc_names, index=loc_names.index(current_name))
        selected_id = loc_map[selected_name]
        if selected_id != state.player_location:
            state.player_location = selected_id
            save_state(session_id, state, sessions_root)
            st.rerun()

        st.subheader("当前 NPC")
        npcs_here = [npc for npc in world.npcs if state.npc_locations.get(npc.npc_id) == state.player_location]
        if npcs_here:
            for npc in npcs_here:
                st.write(f"- {npc.name} ({npc.profession})")
        else:
            st.write("无 NPC")

        st.subheader("RAG 调试")
        logs = read_turn_logs(session_id, sessions_root)
        if logs:
            last_rag = logs[-1].get("rag", {})
            if st.checkbox("显示最近一次 RAG IDs"):
                st.write("always_include_ids:", last_rag.get("always_include_ids", []))
                st.write("retrieved_ids:", last_rag.get("retrieved_ids", []))
        else:
            st.write("暂无日志")

    with col_main:
        st.header("对话")
        if st.session_state.move_notice:
            notice_lines = st.session_state.move_notice
            message = "；".join(notice_lines)
            if hasattr(st, "toast"):
                st.toast(f"NPC 移动：{message}")
            else:
                st.info(f"NPC 移动：{message}")
            st.session_state.move_notice = []
        loc = world.get_location(state.player_location)
        if loc:
            st.markdown(f"**地点**：{loc.name}")
            st.write(loc.description)
        st.markdown("**故事背景**")
        st.write(world.starting_hook)
        st.markdown("**当前任务**")
        st.write(world.initial_quest)

        logs = read_turn_logs(session_id, sessions_root)
        if logs:
            _render_log_history(world, logs, limit=10)
        else:
            st.write("暂无对话记录。")

        npc_options = {npc.name: npc.npc_id for npc in npcs_here}
        if npc_options:
            npc_name = st.selectbox("选择 NPC", options=list(npc_options.keys()))
            npc_id = npc_options[npc_name]
        else:
            npc_id = None

        player_text = st.text_input("你的输入")
        if st.button("Send") and player_text:
            if not npc_id:
                st.error("当前地点没有可对话 NPC。")
            elif not has_api_key:
                st.error(f"未检测到 API Key（{api_key_env}）。")
            else:
                try:
                    llm = QwenOpenAICompatibleClient(cfg)
                    pipeline = TurnPipeline(cfg=cfg, llm_client=llm, sessions_root=sessions_root)
                    updated_state, _output, _log = pipeline.run_turn(state, player_text, npc_id)
                    moved_lines = []
                    for npc in world.npcs:
                        old_loc = state.npc_locations.get(npc.npc_id)
                        new_loc = updated_state.npc_locations.get(npc.npc_id)
                        if old_loc and new_loc and old_loc != new_loc:
                            old_name = world.get_location(old_loc).name if world.get_location(old_loc) else old_loc
                            new_name = world.get_location(new_loc).name if world.get_location(new_loc) else new_loc
                            moved_lines.append(f"{npc.name}：{old_name} → {new_name}")
                    st.session_state.move_notice = moved_lines
                    st.rerun()
                except Exception as exc:
                    st.error(f"对话失败：{exc}")
