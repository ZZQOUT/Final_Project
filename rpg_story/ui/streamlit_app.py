"""Streamlit UI for world generation + RPG gameplay."""
from __future__ import annotations

import streamlit as st

from rpg_story.config import load_config
from rpg_story.engine.orchestrator import Orchestrator
from rpg_story.persistence.store import load_state
from rpg_story.world.schemas import WorldSpec
from rpg_story.engine.state import GameState


st.set_page_config(page_title="RPG Story Prototype", layout="wide")

config = load_config("configs/config.yaml")

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = Orchestrator(config)
if "game_state" not in st.session_state:
    st.session_state.game_state = None
if "chat" not in st.session_state:
    st.session_state.chat = []

st.title("Adaptive RPG Storytelling")

# Phase A: World generation
if st.session_state.game_state is None:
    st.header("Create World")
    world_prompt = st.text_area(
        "World Seed / World Prompt",
        placeholder="e.g., medieval kingdom with dragons, dark forest, sacred temple",
        height=100,
    )
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Generate World") and world_prompt.strip():
            state = st.session_state.orchestrator.create_world(world_prompt.strip())
            st.session_state.game_state = state
            st.session_state.chat.append({"role": "system", "text": "World generated."})
    with col_b:
        if st.button("Use Fallback World"):
            state = GameState.from_world(WorldSpec.minimal_default(), session_id="fallback")
            st.session_state.game_state = state
            st.session_state.chat.append({"role": "system", "text": "Fallback world loaded."})
    with col_c:
        load_id = st.text_input("Load session_id")
        if st.button("Load Session") and load_id:
            st.session_state.game_state = load_state(load_id)
            st.session_state.chat.append({"role": "system", "text": f"Loaded session {load_id}."})

else:
    state: GameState = st.session_state.game_state
    col_left, col_main = st.columns([1, 3])

    with col_left:
        st.header("Map")
        st.write("Session:", state.session_id)
        st.write("Current location:", state.player_location)
        for loc in state.world.locations:
            if st.button(loc.name, key=f"loc_{loc.location_id}"):
                state.player_location = loc.location_id
                st.session_state.chat.append({"role": "system", "text": f"Moved to {loc.name}."})

        st.subheader("NPCs here")
        npcs_here = state.get_npcs_at_location(state.player_location)
        for npc in npcs_here:
            st.write(f"- {npc.name}")

    with col_main:
        st.header("Chat")
        for msg in st.session_state.chat:
            st.write(f"**{msg['role']}**: {msg['text']}")

        npc_options = [npc.npc_id for npc in state.get_npcs_at_location(state.player_location)]
        npc_target = st.selectbox("Talk to", options=npc_options) if npc_options else None
        player_input = st.text_input("Your message")
        if st.button("Send") and player_input and npc_target:
            response = st.session_state.orchestrator.run_turn(
                state=state,
                player_input=player_input,
                npc_target=npc_target,
            )
            st.session_state.chat.append({"role": "player", "text": player_input})
            st.session_state.chat.append({"role": "npc", "text": response.narration})
            st.session_state.game_state = response.updated_state
