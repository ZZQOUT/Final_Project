"""Game state data model with world generation support."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
import json

from rpg_story.world.schemas import WorldSpec, NPCProfile, LocationSpec


@dataclass
class GameState:
    world: WorldSpec
    player_location: str
    npc_locations: Dict[str, str]
    world_facts: List[str] = field(default_factory=list)
    timeline: List[str] = field(default_factory=list)
    quests: List[str] = field(default_factory=list)
    flags: Dict[str, bool] = field(default_factory=dict)
    inventory: List[str] = field(default_factory=list)
    session_id: str = ""

    @classmethod
    def from_world(cls, world: WorldSpec, session_id: str) -> "GameState":
        npc_locations = {npc.npc_id: npc.starting_location for npc in world.npcs}
        return cls(
            world=world,
            player_location=world.starting_location,
            npc_locations=npc_locations,
            session_id=session_id,
        )

    def get_npcs_at_location(self, location_id: str) -> List[NPCProfile]:
        return [npc for npc in self.world.npcs if self.npc_locations.get(npc.npc_id) == location_id]

    def get_location(self, location_id: str) -> LocationSpec:
        return next(loc for loc in self.world.locations if loc.location_id == location_id)

    def has_location(self, location_id: str) -> bool:
        return any(loc.location_id == location_id for loc in self.world.locations)

    def save(self, path: str | Path) -> None:
        data = {
            "world": self.world.model_dump(),
            "player_location": self.player_location,
            "npc_locations": self.npc_locations,
            "world_facts": self.world_facts,
            "timeline": self.timeline,
            "quests": self.quests,
            "flags": self.flags,
            "inventory": self.inventory,
            "session_id": self.session_id,
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "GameState":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        world = WorldSpec.model_validate(raw["world"])
        return GameState(
            world=world,
            player_location=raw["player_location"],
            npc_locations=raw["npc_locations"],
            world_facts=raw.get("world_facts", []),
            timeline=raw.get("timeline", []),
            quests=raw.get("quests", []),
            flags=raw.get("flags", {}),
            inventory=raw.get("inventory", []),
            session_id=raw.get("session_id", ""),
        )
