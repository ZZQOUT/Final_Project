"""CLI world generation helper."""
from __future__ import annotations

import argparse

from rpg_story.config import load_config
from rpg_story.engine.orchestrator import Orchestrator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", default="data/sessions/latest_world.json")
    args = parser.parse_args()

    config = load_config(args.config)
    orch = Orchestrator(config)
    orch.create_world(args.prompt, save_path=args.out)
    print(f"World saved to {args.out}")


if __name__ == "__main__":
    main()
