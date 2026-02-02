"""CLI entry point for the RPG prototype."""
from __future__ import annotations

import argparse

from rpg_story.config import load_config
from rpg_story.engine.orchestrator import Orchestrator
from rpg_story.engine.state import GameState


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    state = GameState.new_game()
    engine = Orchestrator(config)

    print("Type 'quit' to exit.")
    while True:
        player_input = input("You> ")
        if player_input.strip().lower() in {"quit", "exit"}:
            break
        npc_target = state.get_npcs_at_location(state.player_location)[0].npc_id
        result = engine.run_turn(
            state=state,
            player_input=player_input,
            npc_target=npc_target,
            player_location=state.player_location,
        )
        state = result.updated_state
        print(result.narration)


if __name__ == "__main__":
    main()
