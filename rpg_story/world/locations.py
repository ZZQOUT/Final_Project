"""Location definitions for the RPG world."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Location:
    location_id: str
    name: str
    description: str
