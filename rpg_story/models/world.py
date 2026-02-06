"""Canonical data contracts for world and game state."""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator


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

    tech_level: Literal["medieval", "modern", "sci-fi"] = "medieval"
    narrative_language: Optional[Literal["zh", "en"]] = None
    magic_rules: str
    tone: str
    anachronism_policy: Optional[str] = None
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


class QuestSpec(BaseModel):
    """Quest specification generated as part of WorldSpec."""

    model_config = ConfigDict(extra="forbid")

    quest_id: str
    title: str
    category: Literal["main", "side"] = "side"
    description: str
    objective: str
    giver_npc_id: Optional[str] = None
    suggested_location: Optional[str] = None
    required_items: Dict[str, int] = Field(default_factory=dict)
    reward_items: Dict[str, int] = Field(default_factory=dict)
    reward_hint: Optional[str] = None

    @field_validator("required_items", "reward_items", mode="before")
    @classmethod
    def _normalize_required_items(cls, value):
        if value is None:
            return {}
        if isinstance(value, list):
            result: Dict[str, int] = {}
            for item in value:
                if not isinstance(item, dict):
                    continue
                key = item.get("item") or item.get("name")
                count = item.get("count") or item.get("qty") or item.get("quantity") or 1
                if not key:
                    continue
                try:
                    amount = int(float(count))
                except Exception:
                    amount = 1
                if amount > 0:
                    result[str(key)] = amount
            return result
        if isinstance(value, dict):
            result: Dict[str, int] = {}
            for key, raw in value.items():
                try:
                    amount = int(float(raw))
                except Exception:
                    continue
                if amount > 0:
                    result[str(key)] = amount
            return result
        return {}


class MapPosition(BaseModel):
    """Relative map coordinate for one location."""

    model_config = ConfigDict(extra="forbid")

    location_id: str
    x: float = Field(..., ge=0.0, le=100.0)
    y: float = Field(..., ge=0.0, le=100.0)


class QuestProgress(BaseModel):
    """Runtime quest progress tracked in GameState."""

    model_config = ConfigDict(extra="forbid")

    quest_id: str
    title: str
    category: Literal["main", "side"] = "side"
    status: Literal["available", "active", "completed", "failed"] = "available"
    objective: str = ""
    guidance: str = ""
    giver_npc_id: Optional[str] = None
    required_items: Dict[str, int] = Field(default_factory=dict)
    collected_items: Dict[str, int] = Field(default_factory=dict)
    reward_items: Dict[str, int] = Field(default_factory=dict)
    reward_hint: Optional[str] = None

    @field_validator("required_items", "collected_items", "reward_items", mode="before")
    @classmethod
    def _normalize_item_map(cls, value):
        if value is None:
            return {}
        if not isinstance(value, dict):
            return {}
        result: Dict[str, int] = {}
        for key, raw in value.items():
            try:
                amount = int(float(raw))
            except Exception:
                continue
            if amount >= 0:
                result[str(key)] = amount
        return result


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
    main_quest: Optional[QuestSpec] = None
    side_quests: List[QuestSpec] = Field(default_factory=list)
    map_layout: List[MapPosition] = Field(default_factory=list)

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
        npc_ids_set = set(npc_ids)
        quest_ids: Set[str] = set()
        if self.main_quest:
            if self.main_quest.category != "main":
                raise ValueError("main_quest.category must be 'main'")
            quest_ids.add(self.main_quest.quest_id)
            if self.main_quest.giver_npc_id and self.main_quest.giver_npc_id not in npc_ids_set:
                raise ValueError("main_quest.giver_npc_id must exist in npcs")
            if self.main_quest.suggested_location and self.main_quest.suggested_location not in loc_ids:
                raise ValueError("main_quest.suggested_location must exist in locations")
        for quest in self.side_quests:
            if quest.category != "side":
                raise ValueError(f"side_quests category must be 'side': {quest.quest_id}")
            if quest.quest_id in quest_ids:
                raise ValueError("quest_id must be unique across main_quest/side_quests")
            quest_ids.add(quest.quest_id)
            if quest.giver_npc_id and quest.giver_npc_id not in npc_ids_set:
                raise ValueError(f"Quest giver_npc_id invalid: {quest.quest_id}")
            if quest.suggested_location and quest.suggested_location not in loc_ids:
                raise ValueError(f"Quest suggested_location invalid: {quest.quest_id}")
        layout_ids = set()
        for node in self.map_layout:
            if node.location_id not in loc_ids:
                raise ValueError(f"map_layout contains unknown location_id: {node.location_id}")
            if node.location_id in layout_ids:
                raise ValueError(f"map_layout duplicates location_id: {node.location_id}")
            layout_ids.add(node.location_id)
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
    quest_journal: Dict[str, QuestProgress] = Field(default_factory=dict)
    main_quest_id: Optional[str] = None
    inventory: Dict[str, int] = Field(default_factory=dict)
    location_resource_stock: Dict[str, Dict[str, int]] = Field(default_factory=dict)
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
        if self.main_quest_id and self.main_quest_id not in self.quest_journal:
            raise ValueError("main_quest_id must exist in quest_journal")
        for loc_id, stock in self.location_resource_stock.items():
            if loc_id not in loc_ids:
                raise ValueError(f"location_resource_stock contains invalid location_id: {loc_id}")
            if not isinstance(stock, dict):
                raise ValueError("location_resource_stock values must be dictionaries")
            for item_name, item_count in stock.items():
                if int(item_count) < 0:
                    raise ValueError(
                        f"location_resource_stock count must be >= 0: {loc_id}.{item_name}"
                    )
        self.validate_references()
        return self

    def to_dict(self) -> Dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict) -> "GameState":
        return cls.model_validate(data)
