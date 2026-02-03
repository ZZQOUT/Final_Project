"""Canonical data contracts for world and game state."""
from __future__ import annotations

from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field, ConfigDict, model_validator


class LocationSpec(BaseModel):
    """Location definition used for map connectivity."""

    model_config = ConfigDict(extra="forbid")

    location_id: str = Field(..., description="Stable id like loc_001")
    name: str
    kind: str = Field(..., description="Expected values: town, dungeon, forest, castle, bridge, ...")
    description: str
    connected_to: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class WorldBibleRules(BaseModel):
    """World rules and taboos."""

    model_config = ConfigDict(extra="forbid")

    tech_level: str
    magic_rules: str
    tone: str
    taboos: List[str] = Field(default_factory=list)
    do_not_mention: List[str] = Field(default_factory=list)
    anachronism_blocklist: List[str] = Field(default_factory=list)


class NPCProfile(BaseModel):
    """NPC personality and agency controls."""

    model_config = ConfigDict(extra="forbid")

    npc_id: str
    name: str
    profession: str
    traits: List[str]
    goals: List[str]
    starting_location: str
    obedience_level: float = Field(..., ge=0.0, le=1.0)
    stubbornness: float = Field(..., ge=0.0, le=1.0)
    risk_tolerance: float = Field(..., ge=0.0, le=1.0)
    disposition_to_player: int = Field(..., ge=-5, le=5)
    refusal_style: str


class WorldSpec(BaseModel):
    """World spec produced by WorldGen."""

    model_config = ConfigDict(extra="forbid")

    world_id: str
    title: str
    world_bible: WorldBibleRules
    locations: List[LocationSpec]
    npcs: List[NPCProfile]
    starting_location: str
    starting_hook: str
    initial_quest: str

    def location_ids(self) -> Set[str]:
        return {loc.location_id for loc in self.locations}

    def get_location(self, location_id: str) -> Optional[LocationSpec]:
        for loc in self.locations:
            if loc.location_id == location_id:
                return loc
        return None

    def validate_bidirectional_edges(self, strict: bool) -> None:
        """Optionally enforce bidirectional map edges when strict=True."""
        if not strict:
            return
        adjacency = {loc.location_id: set(loc.connected_to) for loc in self.locations}
        violations = []
        for a, neighbors in adjacency.items():
            for b in neighbors:
                if a not in adjacency.get(b, set()):
                    violations.append((a, b))
        if violations:
            pairs = ", ".join([f"{a}->{b}" for a, b in violations])
            raise ValueError(f"non-bidirectional edges found: {pairs}")

    @model_validator(mode="after")
    def _validate_world(self) -> "WorldSpec":
        loc_ids = self.location_ids()
        if len(loc_ids) != len(self.locations):
            raise ValueError("location_id must be unique")
        npc_ids = [npc.npc_id for npc in self.npcs]
        if len(set(npc_ids)) != len(npc_ids):
            raise ValueError("npc_id must be unique")
        if self.starting_location not in loc_ids:
            raise ValueError("starting_location must exist in locations")
        for npc in self.npcs:
            if npc.starting_location not in loc_ids:
                raise ValueError(f"NPC starting_location invalid: {npc.npc_id}")
        for loc in self.locations:
            bad = [target for target in loc.connected_to if target not in loc_ids]
            if bad:
                raise ValueError(f"connected_to invalid for {loc.location_id}: {bad}")
        return self

    def to_dict(self) -> Dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict) -> "WorldSpec":
        return cls.model_validate(data)


class GameState(BaseModel):
    """Global game state contract (single source of truth)."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    created_at: str
    world: WorldSpec
    player_location: str
    npc_locations: Dict[str, str]
    flags: Dict[str, bool] = Field(default_factory=dict)
    quests: Dict[str, str] = Field(default_factory=dict)
    inventory: Dict[str, int] = Field(default_factory=dict)
    recent_summaries: List[str] = Field(default_factory=list)
    last_turn_id: int = 0

    def npcs_at(self, location_id: str) -> List[str]:
        return [npc_id for npc_id, loc_id in self.npc_locations.items() if loc_id == location_id]

    def validate_references(self) -> None:
        """Validate npc_locations coverage and references.\n\n        Policy: every NPC in world.npcs must have a location entry.\n        """
        loc_ids = self.world.location_ids()
        npc_ids = {npc.npc_id for npc in self.world.npcs}
        # keys must be subset of npc_ids
        unknown = [npc_id for npc_id in self.npc_locations.keys() if npc_id not in npc_ids]
        if unknown:
            raise ValueError(f"npc_locations contains unknown npc_id: {unknown}")
        # require full coverage
        missing = [npc_id for npc_id in npc_ids if npc_id not in self.npc_locations]
        if missing:
            raise ValueError(f"npc_locations missing npc_id: {missing}")
        for npc_id, loc_id in self.npc_locations.items():
            if loc_id not in loc_ids:
                raise ValueError(f"npc_locations contains invalid location: {npc_id} -> {loc_id}")

    @model_validator(mode="after")
    def _validate_state(self) -> "GameState":
        loc_ids = self.world.location_ids()
        if self.player_location not in loc_ids:
            raise ValueError("player_location must exist in locations")
        self.validate_references()
        return self

    def to_dict(self) -> Dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict) -> "GameState":
        return cls.model_validate(data)
