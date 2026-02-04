"""Minimal turn pipeline orchestrator (Milestone 3)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone

from rpg_story.config import AppConfig
from rpg_story.llm.client import BaseLLMClient
from rpg_story.llm.schemas import validate_turn_output
from rpg_story.models.world import GameState
from rpg_story.models.turn import TurnOutput
from rpg_story.engine.state import apply_turn_output
from rpg_story.persistence.store import append_turn_log, save_state, default_sessions_root


@dataclass
class TurnPipeline:
    cfg: AppConfig
    llm_client: BaseLLMClient
    sessions_root: Optional[Path] = None

    def _prompts_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "prompts"

    def _load_prompt(self, filename: str) -> str:
        path = self._prompts_dir() / filename
        return path.read_text(encoding="utf-8")

    def _build_prompts(self, state: GameState, player_text: str, npc_id: str) -> tuple[str, str]:
        base = self._load_prompt("system_base.txt")
        narrator = self._load_prompt("narrator.txt")
        persona_template = self._load_prompt("npc_persona.txt")

        npc = next((n for n in state.world.npcs if n.npc_id == npc_id), None)
        npc_name = npc.name if npc else npc_id
        npc_prof = npc.profession if npc else "unknown"
        npc_traits = ", ".join(npc.traits) if npc else ""
        npc_location = state.npc_locations.get(npc_id, state.player_location)
        persona = persona_template.format(
            npc_name=npc_name,
            profession=npc_prof,
            traits=npc_traits,
            location_id=npc_location,
        )

        system_prompt = base + "\n" + narrator + "\n" + persona
        world_constraints = "World constraints: medieval setting, no modern tech, maintain consistency."
        schema_hint = (
            "Return JSON with keys: narration, npc_dialogue, world_updates, memory_summary, safety. "
            "npc_dialogue is a list of {npc_id, text}. world_updates may include player_location, npc_moves, flags_delta, quest_updates. "
            "npc_moves MUST be a list (use [] if none). "
            "quest_updates MUST be an object/dict; if none, use {}. "
            "safety MUST be an object: {refuse: boolean, reason: string|null}."
        )
        user_prompt = (
            f"location_id: {state.player_location}\n"
            f"npc_id: {npc_id}\n"
            f"player_text: {player_text}\n"
            f"{world_constraints}\n"
            f"schema_hint: {schema_hint}"
        )
        return system_prompt, user_prompt

    def run_turn(self, state: GameState, player_text: str, npc_id: str) -> Tuple[GameState, TurnOutput, Dict[str, Any]]:
        system_prompt, user_prompt = self._build_prompts(state, player_text, npc_id)
        data = self.llm_client.generate_json(system_prompt, user_prompt)
        output = validate_turn_output(data)
        updated_state = apply_turn_output(state, output, npc_id)

        turn_index = updated_state.last_turn_id
        timestamp = datetime.now(timezone.utc).isoformat()
        log_record = {
            "session_id": updated_state.session_id,
            "turn_index": turn_index,
            "timestamp": timestamp,
            "player_text": player_text,
            "npc_id": npc_id,
            "location_id": updated_state.player_location,
            "model_used": self.cfg.llm.model,
            "output": output.model_dump(),
        }

        sessions_root = self.sessions_root or default_sessions_root(self.cfg)
        append_turn_log(updated_state.session_id, log_record, sessions_root)
        save_state(updated_state.session_id, updated_state, sessions_root)

        return updated_state, output, log_record
