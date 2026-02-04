"""Single-turn CLI for Milestone 3 (no UI, no RAG)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json

from rpg_story.config import load_config
from rpg_story.llm.client import QwenOpenAICompatibleClient
from rpg_story.models.world import WorldSpec, WorldBibleRules, LocationSpec, NPCProfile, GameState
from rpg_story.persistence.store import generate_session_id, load_state, save_state, default_sessions_root
from rpg_story.engine.orchestrator import TurnPipeline


def _minimal_world() -> WorldSpec:
    locations = [
        LocationSpec(
            location_id="shop",
            name="Shop",
            kind="shop",
            description="A small shop with wooden shelves.",
            connected_to=["bridge"],
        ),
        LocationSpec(
            location_id="bridge",
            name="Broken Bridge",
            kind="bridge",
            description="A broken bridge over a misty river.",
            connected_to=["shop"],
        ),
    ]
    npc = NPCProfile(
        npc_id="npc_1",
        name="Mara",
        profession="Merchant",
        traits=["practical"],
        goals=["protect her goods"],
        starting_location="shop",
        obedience_level=0.5,
        stubbornness=0.5,
        risk_tolerance=0.5,
        disposition_to_player=0,
        refusal_style="firm but polite",
    )
    bible = WorldBibleRules(
        tech_level="medieval",
        magic_rules="low",
        tone="grounded",
    )
    return WorldSpec(
        world_id="world_demo",
        title="Demo World",
        world_bible=bible,
        locations=locations,
        npcs=[npc],
        starting_location="shop",
        starting_hook="A rumor spreads in the market.",
        initial_quest="Deliver a message across the river.",
    )


def _load_world(path: str) -> WorldSpec:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return WorldSpec.model_validate(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--session", default=None)
    parser.add_argument("--npc", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--world", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    sessions_root = default_sessions_root(cfg)

    if args.session:
        state = load_state(args.session, sessions_root)
    else:
        session_id = generate_session_id()
        world = _load_world(args.world) if args.world else _minimal_world()
        state = GameState(
            session_id=session_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            world=world,
            player_location=world.starting_location,
            npc_locations={npc.npc_id: npc.starting_location for npc in world.npcs},
        )
        save_state(session_id, state, sessions_root)

    client = QwenOpenAICompatibleClient(cfg)
    pipeline = TurnPipeline(cfg=cfg, llm_client=client, sessions_root=sessions_root)

    updated_state, output, _log = pipeline.run_turn(state, args.text, args.npc)
    print(output.narration)
    print(f"player_location: {updated_state.player_location}")
    if args.npc in updated_state.npc_locations:
        print(f"npc_location[{args.npc}]: {updated_state.npc_locations[args.npc]}")


if __name__ == "__main__":
    main()
