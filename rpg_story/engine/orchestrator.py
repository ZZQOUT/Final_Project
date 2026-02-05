"""Minimal turn pipeline orchestrator (Milestone 3)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Set
import re
from datetime import datetime, timezone
import json

from rpg_story.config import AppConfig
from rpg_story.llm.client import BaseLLMClient, make_json_schema_response_format
from rpg_story.llm.schemas import validate_turn_output
from rpg_story.models.world import GameState, WorldSpec
from rpg_story.models.turn import TurnOutput, NPCDialogueLine
from rpg_story.engine.state import apply_turn_output
from rpg_story.engine.validators import validate_npc_move
from rpg_story.engine.agency import apply_agency_gate
from rpg_story.rag.index import RAGIndex
from rpg_story.rag.retriever import RAGRetriever
from rpg_story.rag.stores.memory import InMemoryStore
from rpg_story.rag.types import Document
from rpg_story.persistence.store import append_turn_log, save_state, default_sessions_root
from rpg_story.world.term_guard import DEFAULT_ANACHRONISM_TERMS, detect_first_mention


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

    def _build_prompts(
        self,
        state: GameState,
        player_text: str,
        npc_id: str,
        rag_text: str | None = None,
    ) -> tuple[str, str]:
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
        if rag_text:
            system_prompt += "\n\n" + rag_text
        tech_level = getattr(state.world.world_bible, "tech_level", "medieval") or "medieval"
        world_constraints = (
            f"World constraints: tech_level={tech_level}, maintain lore consistency."
        )
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
            f"npc_name: {npc_name}\n"
            f"player_text: {player_text}\n"
            f"{world_constraints}\n"
            "NPC reply MUST be placed in npc_dialogue using the same npc_id. "
            "NPC identity must stay consistent with npc_name; correct the player if they use a wrong name. "
            "If asked about other NPCs or roles, only use names/professions from the NPC roster in WORLD BIBLE; "
            "if a role does not exist, say there is no such person in town. "
            "Keep narration brief and optional.\n"
            f"schema_hint: {schema_hint}"
        )
        return system_prompt, user_prompt

    def run_turn(self, state: GameState, player_text: str, npc_id: str) -> Tuple[GameState, TurnOutput, Dict[str, Any]]:
        rag_text, rag_debug = self._build_rag_context(state, npc_id, player_text)
        system_prompt, user_prompt = self._build_prompts(state, player_text, npc_id, rag_text)
        response_format = make_json_schema_response_format(
            name="TurnOutput",
            schema=TurnOutput.model_json_schema(),
            description="Narration and world updates for a single RPG turn.",
        )
        data = self.llm_client.generate_json(system_prompt, user_prompt, response_format=response_format)
        output = validate_turn_output(data)
        output = self._enforce_no_first_mention(state, player_text, output, response_format)
        output = self._ensure_npc_dialogue(output, npc_id)
        output = self._enforce_world_roster(state, output, npc_id, response_format)
        state_non_move = apply_turn_output(state, output, npc_id)

        move_rejections = []
        legal_moves = []
        for move in output.world_updates.npc_moves:
            ok, reason = validate_npc_move(move, state_non_move, state_non_move.world)
            if ok:
                legal_moves.append(move)
                continue
            move_rejections.append(
                {
                    "type": "move_rejected",
                    "npc_id": move.npc_id,
                    "from_location": move.from_location,
                    "to_location": move.to_location,
                    "reason": reason,
                }
            )

        dialogue_by_id = {}
        for line in output.npc_dialogue:
            if not line.text:
                continue
            dialogue_by_id.setdefault(line.npc_id, []).append(line.text)
        allowed_moves, move_refusals = apply_agency_gate(
            legal_moves,
            state_non_move,
            state_non_move.world,
            player_text,
            dialogue_by_id,
        )
        updated_state = state_non_move.model_copy(deep=True)
        for move in allowed_moves:
            updated_state.npc_locations[move.npc_id] = move.to_location
        updated_state = GameState.model_validate(updated_state.model_dump())

        total_moves = len(output.world_updates.npc_moves)
        rejected_count = len(move_rejections)
        refused_count = len(move_refusals)
        applied_count = max(0, total_moves - rejected_count - refused_count)
        output = self._append_refusals(output, move_refusals, state.world)

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
            "move_rejections": move_rejections,
            "move_refusals": move_refusals,
            "move_applied_count": applied_count,
            "move_rejected_count": rejected_count,
            "move_refused_count": refused_count,
            "rag": rag_debug,
        }

        sessions_root = self.sessions_root or default_sessions_root(self.cfg)
        append_turn_log(updated_state.session_id, log_record, sessions_root)
        save_state(updated_state.session_id, updated_state, sessions_root)

        return updated_state, output, log_record

    def _build_rag_context(
        self,
        state: GameState,
        npc_id: str,
        player_text: str,
    ) -> tuple[str, dict]:
        if not self.cfg.rag.enabled:
            return "", {"enabled": False}
        sessions_root = self.sessions_root or default_sessions_root(self.cfg)
        store = InMemoryStore()
        index = RAGIndex(store)
        index.build_default(state.session_id, state.world)
        retriever = RAGRetriever(index)
        query_text = f"{player_text}\nlocation:{state.player_location}\nnpc:{npc_id}"
        rag_pack = retriever.get_forced_context_pack(
            session_id=state.session_id,
            world=state.world,
            state=state,
            npc_id=npc_id,
            sessions_root=sessions_root,
            last_n_summaries=self.cfg.rag.summary_window,
            top_k=self.cfg.rag.top_k,
            query_text=query_text,
        )
        rag_text = self._render_rag_pack(rag_pack)
        debug = {
            "enabled": True,
            "always_include_ids": [doc.id for doc in rag_pack["always_include"]],
            "retrieved_ids": [doc.id for doc in rag_pack["retrieved"]],
            **rag_pack["debug"],
        }
        return rag_text, debug

    def _render_rag_pack(self, rag_pack: dict) -> str:
        always = rag_pack.get("always_include", [])
        retrieved = rag_pack.get("retrieved", [])
        sections = []
        sections.append(self._render_section("WORLD BIBLE", _filter_docs(always, "world_bible")))
        sections.append(self._render_section("LOCATION", _filter_docs(always, "location")))
        sections.append(self._render_section("NPC PROFILE", _filter_docs(always, "npc_profile")))
        sections.append(self._render_section("NPC MEMORY", _filter_docs(always, "memory")))
        sections.append(self._render_section("RECENT SUMMARIES", _filter_docs(always, "summary")))
        sections.append(self._render_section("RETRIEVED MEMORIES", retrieved))
        return "\n\n".join([section for section in sections if section])

    def _render_section(self, title: str, docs: list[Document]) -> str:
        if not docs:
            return ""
        body = "\n".join([doc.text for doc in docs if doc.text])
        return f"=== {title} ===\n{body}"

    def _append_refusals(self, output: TurnOutput, refusals: list[dict], world: WorldSpec) -> TurnOutput:
        if not refusals:
            return output
        updated = output.model_copy(deep=True)
        lines = []
        for refusal in refusals:
            npc_id = refusal.get("npc_id", "npc")
            npc_name = self._npc_name(world, npc_id)
            reason = refusal.get("reason", "refused")
            lines.append(f"{npc_name} refuses to move: {reason}.")
        if updated.narration:
            updated.narration = updated.narration.rstrip() + " " + " ".join(lines)
        else:
            updated.narration = " ".join(lines)
        return updated

    def _npc_name(self, world: WorldSpec, npc_id: str) -> str:
        for npc in world.npcs:
            if npc.npc_id == npc_id:
                return npc.name
        return npc_id

    def _ensure_npc_dialogue(self, output: TurnOutput, npc_id: str) -> TurnOutput:
        if output.npc_dialogue:
            has_text = any(line.text.strip() for line in output.npc_dialogue)
            if has_text:
                return output
        if not output.narration.strip():
            return output
        updated = output.model_copy(deep=True)
        updated.npc_dialogue = [
            NPCDialogueLine(npc_id=npc_id, text=updated.narration.strip())
        ]
        updated.narration = ""
        return updated

    def _enforce_world_roster(
        self,
        state: GameState,
        output: TurnOutput,
        npc_id: str,
        response_format: dict,
    ) -> TurnOutput:
        roster_names = [npc.name for npc in state.world.npcs if npc.name]
        roster_professions = [npc.profession for npc in state.world.npcs if npc.profession]
        if not roster_names and not roster_professions:
            return output
        texts = []
        if output.narration:
            texts.append(output.narration)
        for line in output.npc_dialogue:
            if line.text:
                texts.append(line.text)
        offending = _detect_unknown_references(texts, roster_names, roster_professions)
        if not offending:
            return output
        rewrite_system = (
            "You are a JSON repair tool. Return ONLY valid JSON. "
            "Keep NPC/world consistency with the roster."
        )
        rewrite_user = (
            "The NPC must only reference known NPCs and professions from the roster.\n"
            f"Known NPC names: {roster_names}\n"
            f"Known professions: {roster_professions}\n"
            f"Offending terms: {sorted(offending)}\n\n"
            "Remove or replace offending terms. If a role does not exist, state that no such person exists.\n\n"
            f"Original JSON: {json.dumps(output.model_dump(), ensure_ascii=False)}"
        )
        fixed = self.llm_client.generate_json(rewrite_system, rewrite_user, response_format=response_format)
        fixed_output = validate_turn_output(fixed)
        fixed_output = self._ensure_npc_dialogue(fixed_output, npc_id)
        texts = []
        if fixed_output.narration:
            texts.append(fixed_output.narration)
        for line in fixed_output.npc_dialogue:
            if line.text:
                texts.append(line.text)
        still_offending = _detect_unknown_references(texts, roster_names, roster_professions)
        if still_offending:
            raise ValueError(f"unknown NPC/role references remain: {sorted(still_offending)}")
        return fixed_output

    def _enforce_no_first_mention(
        self,
        state: GameState,
        player_text: str,
        output: TurnOutput,
        response_format: dict,
    ) -> TurnOutput:
        tech_level = getattr(state.world.world_bible, "tech_level", "medieval") or "medieval"
        if tech_level != "medieval":
            return output
        npc_texts = [output.narration] + [line.text for line in output.npc_dialogue]
        new_terms = detect_first_mention(player_text, npc_texts, DEFAULT_ANACHRONISM_TERMS)
        if not new_terms:
            return output

        rewrite_system = (
            "You are a JSON repair tool. Return ONLY valid JSON. "
            "Remove out-of-setting terms unless the player said them first."
        )
        rewrite_user = (
            "Remove these terms from NPC output unless the player already said them:\n"
            f"{sorted(new_terms)}\n\n"
            f"Player text: {player_text}\n\n"
            "Do not introduce these terms. Keep the rest of the content consistent.\n\n"
            f"Original JSON: {json.dumps(output.model_dump(), ensure_ascii=False)}"
        )
        fixed = self.llm_client.generate_json(rewrite_system, rewrite_user, response_format=response_format)
        fixed_output = validate_turn_output(fixed)
        npc_texts = [fixed_output.narration] + [line.text for line in fixed_output.npc_dialogue]
        still_new = detect_first_mention(player_text, npc_texts, DEFAULT_ANACHRONISM_TERMS)
        if still_new:
            raise ValueError(f"first-mention anachronisms remain: {sorted(still_new)}")
        return fixed_output


def _filter_docs(docs: list[Document], doc_type: str) -> list[Document]:
    return [doc for doc in docs if doc.metadata.get("doc_type") == doc_type]


COMMON_ROLE_TERMS: Set[str] = {
    "铁匠",
    "药师",
    "医生",
    "牧师",
    "守卫",
    "队长",
    "猎人",
    "渔夫",
    "农夫",
    "工匠",
    "裁缝",
    "学者",
    "法师",
    "旅馆老板",
    "店主",
    "商人",
    "blacksmith",
    "healer",
    "doctor",
    "guard",
    "merchant",
}


def _detect_unknown_references(
    texts: List[str],
    roster_names: List[str],
    roster_professions: List[str],
) -> Set[str]:
    offending: Set[str] = set()
    name_set = {name.strip() for name in roster_names if name.strip()}
    prof_set = {prof.strip() for prof in roster_professions if prof.strip()}
    for text in texts:
        lower_text = text.lower()
        for match in re.findall(r"(?:名叫|名为|叫做|叫)([\\u4e00-\\u9fff]{2,4})", text):
            if match not in name_set:
                offending.add(match)
        for role in COMMON_ROLE_TERMS:
            if role.isascii():
                if role in lower_text:
                    if not any(role in prof.lower() for prof in prof_set):
                        offending.add(role)
                continue
            if role in text:
                if not any(role in prof for prof in prof_set):
                    offending.add(role)
    return offending
