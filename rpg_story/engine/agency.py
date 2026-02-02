"""NPC agency gate: decide if NPC accepts a move request."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from rpg_story.world.schemas import NPCProfile


@dataclass
class AgencyDecision:
    accepted: bool
    reason: str


def evaluate_move_acceptance(
    npc: NPCProfile,
    goal_alignment: float,
    risk: float,
    request_strength: float = 0.5,
) -> AgencyDecision:
    """Deterministic acceptance function.

    Inputs are normalized to [0..1].
    Disposition in NPCProfile is -5..+5 and is normalized to [0..1].
    """
    obedience = npc.agency_rules.obedience_level
    stubborn = npc.agency_rules.stubbornness
    risk_tol = npc.agency_rules.risk_tolerance
    disp_norm = (npc.disposition_to_player + 5) / 10.0

    score = (
        obedience * 0.35
        + goal_alignment * 0.30
        + request_strength * 0.15
        + disp_norm * 0.10
        - stubborn * 0.35
        - risk * (1.0 - risk_tol) * 0.40
    )
    accepted = score >= 0.2
    reason = "accepted" if accepted else "npc_refused"
    return AgencyDecision(accepted=accepted, reason=reason)
