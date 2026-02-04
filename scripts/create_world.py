"""CLI world generation helper (Milestone 4)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rpg_story.config import load_config
from rpg_story.llm.client import QwenOpenAICompatibleClient, MockLLMClient
from rpg_story.world.generator import create_new_session


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--sessions-root", default=None)
    parser.add_argument("--worlds-root", default=None)
    args = parser.parse_args()

    world_prompt = args.prompt
    if not world_prompt:
        world_prompt = sys.stdin.read().strip()
    if not world_prompt:
        raise SystemExit("Provide --prompt or stdin")

    cfg = load_config(args.config)

    if args.mock:
        mock_world = (
            '{'
            '"world_id":"world_mock",'
            '"title":"Mock World",'
            '"world_bible":{"tech_level":"medieval","magic_rules":"low","tone":"grounded"},'
            '"locations":['
            ' {"location_id":"loc_001","name":"Town","kind":"town","description":"A small town.","connected_to":["loc_002"],"tags":[]},'
            ' {"location_id":"loc_002","name":"Bridge","kind":"bridge","description":"A broken bridge.","connected_to":["loc_001"],"tags":[]}'
            '],'
            '"npcs":['
            ' {"npc_id":"npc_001","name":"Mara","profession":"Merchant","traits":["practical"],"goals":["trade"],"starting_location":"loc_001",'
            '  "obedience_level":0.5,"stubbornness":0.5,"risk_tolerance":0.5,"disposition_to_player":0,"refusal_style":"polite"}'
            '],'
            '"starting_location":"loc_001",'
            '"starting_hook":"A rumor spreads.",'
            '"initial_quest":"Deliver a message."'
            '}'
        )
        llm = MockLLMClient([mock_world])
    else:
        llm = QwenOpenAICompatibleClient(cfg)

    sessions_root = Path(args.sessions_root) if args.sessions_root else None
    worlds_root = Path(args.worlds_root) if args.worlds_root else None

    session_id, world, state = create_new_session(
        cfg, llm, world_prompt, sessions_root=sessions_root, worlds_root=worlds_root
    )

    print(f"session_id: {session_id}")
    print(f"starting_location: {world.starting_location}")
    print(f"locations: {len(world.locations)}")
    print(f"npcs: {len(world.npcs)}")

    sessions_dir = sessions_root or cfg.app.sessions_dir
    worlds_dir = worlds_root or cfg.app.worlds_dir
    print(f"state.json: {Path(sessions_dir) / session_id / 'state.json'}")
    print(f"world.json: {Path(worlds_dir) / session_id / 'world.json'}")


if __name__ == "__main__":
    main()
