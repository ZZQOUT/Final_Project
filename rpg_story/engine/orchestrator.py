"""Two-phase orchestration: WorldGen -> Turn pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import uuid

from rpg_story.config import Config
from rpg_story.engine.state import GameState
from rpg_story.models.turn import TurnResult
from rpg_story.engine.validators import validate_npc_move
from rpg_story.engine.agency import evaluate_move_acceptance
from rpg_story.llm.client import LLMClient
from rpg_story.llm.schemas import validate_turn_output
from rpg_story.persistence.store import save_state, append_turn_log
from rpg_story.rag.index import RAGIndex
from rpg_story.rag.retriever import Retriever, RetrievalPolicy
from rpg_story.world.consistency import WorldConsistencyGuard
from rpg_story.world.generator import WorldGenPipeline


class Orchestrator:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.llm = LLMClient(config)
        self.world_gen = WorldGenPipeline(self.llm)
        self.rag_index = RAGIndex()
        self.retriever = Retriever(RetrievalPolicy())

    def create_world(self, world_prompt: str) -> GameState:
        session_id = f"sess_{uuid.uuid4().hex[:8]}"
        world = self.world_gen.generate(world_prompt)
        state = GameState.from_world(world, session_id=session_id)
        # TODO: ingest world bible, locations, NPC profiles into RAG
        save_state(state)
        return state

    def run_turn(self, state: GameState, player_input: str, npc_target: str) -> TurnResult:
        guard = WorldConsistencyGuard(
            bible=state.world.bible,
            forbidden_terms=state.world.bible.do_not_mention,
        )
        # Strict retrieval policy: always include world bible + location + npc + recent summaries
        _context_docs = self.retriever.build_context(
            session_id=state.session_id,
            location_id=state.player_location,
            npc_id=npc_target,
        )
        prompt = {
            "player_input": player_input,
            "npc_target": npc_target,
            "player_location": state.player_location,
            "world_bible": state.world.bible.model_dump(),
            "context_docs": _context_docs,
        }
        raw = self.llm.generate_turn(prompt)
        output = validate_turn_output(raw)

        refused_moves = []
        # Apply NPC moves with legality + agency checks
        for move in output.world_updates.npc_moves:
            ok, reason = validate_npc_move(
                state=state,
                npc_id=move.npc_id,
                from_location=move.from_location,
                to_location=move.to_location,
            )
            if not ok:
                refused_moves.append({"npc_id": move.npc_id, "reason": reason})
                continue
            npc = next(n for n in state.world.npcs if n.npc_id == move.npc_id)
            decision = evaluate_move_acceptance(
                npc=npc,
                goal_alignment=0.5,
                risk=0.5,
            )
            if decision.accepted:
                state.npc_locations[move.npc_id] = move.to_location
            else:
                refused_moves.append({"npc_id": move.npc_id, "reason": decision.reason})

        # TODO: apply player_location updates and world facts
        narration = output.narration
        if guard.has_violation(narration):
            narration = guard.repair(self.llm, narration)

        record = {
            "session_id": state.session_id,
            "player_input": player_input,
            "npc_target": npc_target,
            "refused_moves": refused_moves,
            "raw_output": output.model_dump(),
        }
        append_turn_log(state.session_id, record)
        save_state(state)
        return TurnResult(narration=narration, updated_state=state, raw_response=output.model_dump())
