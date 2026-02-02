"""World map graph and movement logic."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class WorldMap:
    adjacency: Dict[str, List[str]] = field(default_factory=dict)

    def neighbors(self, location_id: str) -> List[str]:
        return self.adjacency.get(location_id, [])
