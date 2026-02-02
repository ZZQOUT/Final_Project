"""Pydantic schemas for world generation outputs."""
from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


LocationKind = Literal[
    "town",
    "dungeon",
    "landmark",
    "shop",
    "forest",
    "castle",
    "bridge",
    "ruins",
    "temple",
    "cave",
    "village",
]


class WorldBibleRules(BaseModel):
    tech_level: str = Field(..., description="Technology level, e.g., medieval")
    magic_rules: str = Field(..., description="Magic system constraints")
    tone: str = Field(..., description="Overall narrative tone")
    taboos: List[str] = Field(default_factory=list)
    do_not_mention: List[str] = Field(default_factory=list)
    allow_special_travel: bool = Field(False, description="Allows non-graph travel when true")


class LocationSpec(BaseModel):
    """Canonical location schema used for map connectivity."""

    location_id: str = Field(..., description="Stable id like loc_001")
    name: str
    kind: LocationKind
    description: str
    connected_to: List[str] = Field(default_factory=list, description="Adjacent location_ids")
    tags: List[str] = Field(default_factory=list)


class AgencyRules(BaseModel):
    obedience_level: float = Field(0.5, ge=0.0, le=1.0)
    stubbornness: float = Field(0.5, ge=0.0, le=1.0)
    risk_tolerance: float = Field(0.5, ge=0.0, le=1.0)


class NPCProfile(BaseModel):
    npc_id: str = Field(..., description="Stable id like npc_001")
    name: str
    profession: str
    role: str
    traits: List[str] = Field(default_factory=list)
    motivations: List[str] = Field(default_factory=list)
    goals: List[str] = Field(default_factory=list)
    disposition_to_player: int = Field(0, ge=-5, le=5, description="-5..+5")
    agency_rules: AgencyRules = Field(default_factory=AgencyRules)
    refusal_style: str = Field("polite refusal")
    starting_location: str


class PlayerProfile(BaseModel):
    name: str = "Player"
    archetype: str = "Adventurer"
    background: str = "Wanderer"


class WorldSpec(BaseModel):
    world_id: str = Field(..., description="Stable id like world_001")
    title: str
    bible: WorldBibleRules
    locations: List[LocationSpec]
    npcs: List[NPCProfile]
    starting_location: str
    hook: str
    initial_quest: str
    player_profile: Optional[PlayerProfile] = None

    @staticmethod
    def minimal_default() -> "WorldSpec":
        return WorldSpec(
            world_id="world_001",
            title="Fallback Realm",
            bible=WorldBibleRules(
                tech_level="medieval",
                magic_rules="Low magic; rituals are rare.",
                tone="grounded and adventurous",
                taboos=["modern technology"],
                do_not_mention=["smartphone", "internet", "politics"],
                allow_special_travel=False,
            ),
            locations=[
                LocationSpec(
                    location_id="loc_001",
                    name="Tavern",
                    kind="town",
                    description="A warm tavern with a crackling hearth.",
                    connected_to=["loc_002"],
                    tags=["safe", "social"],
                ),
                LocationSpec(
                    location_id="loc_002",
                    name="Broken Bridge",
                    kind="bridge",
                    description="A shattered bridge over a cold river.",
                    connected_to=["loc_001"],
                    tags=["ruin"],
                ),
            ],
            npcs=[
                NPCProfile(
                    npc_id="npc_001",
                    name="Bartender",
                    profession="Innkeeper",
                    role="Quest Giver",
                    traits=["gruff", "helpful"],
                    motivations=["keep peace"],
                    goals=["help travelers"],
                    disposition_to_player=0,
                    agency_rules=AgencyRules(obedience_level=0.6, stubbornness=0.4, risk_tolerance=0.3),
                    refusal_style="gruff but respectful",
                    starting_location="loc_001",
                )
            ],
            starting_location="loc_001",
            hook="A rumor of a lost relic reaches the tavern.",
            initial_quest="Find the relic beyond the broken bridge.",
            player_profile=PlayerProfile(),
        )
