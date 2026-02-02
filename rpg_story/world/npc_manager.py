"""NPC registry and movement logic."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class NPC:
    npc_id: str
    name: str
    persona: str


@dataclass
class NPCRegistry:
    npcs: Dict[str, NPC] = field(default_factory=dict)
    npc_locations: Dict[str, str] = field(default_factory=dict)

    def get_npcs_at(self, location_id: str) -> List[NPC]:
        return [npc for npc_id, npc in self.npcs.items() if self.npc_locations.get(npc_id) == location_id]

    def move_npc(self, npc_id: str, to_location: str) -> None:
        self.npc_locations[npc_id] = to_location
