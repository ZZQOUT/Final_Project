"""NPC agency gate for move compliance decisions."""
from __future__ import annotations

from typing import Dict, List, TypedDict

from rpg_story.models.turn import NPCMove
from rpg_story.models.world import GameState, WorldSpec, NPCProfile


class AgencyDecision(TypedDict):
    allowed: bool
    reason: str
    tags: List[str]


def decide_npc_move(
    move: NPCMove,
    state: GameState,
    world: WorldSpec,
    player_text: str | None = None,
) -> AgencyDecision:
    npc = _find_npc(world, move.npc_id)
    if npc is None:
        return {"allowed": False, "reason": "unknown npc", "tags": ["unknown_npc"]}

    base = 0.5
    score = base
    score += 0.35 * npc.obedience_level
    score -= 0.35 * npc.stubbornness
    score += 0.05 * (npc.disposition_to_player / 5.0)

    risk_penalty, risk_tags = _risk_alignment_penalty(npc, world, move.to_location)
    score += risk_penalty

    role_penalty, role_tags = _role_constraints_penalty(npc, state, move)
    score += role_penalty

    score = _clamp(score, 0.0, 1.0)
    threshold = _base_threshold(player_text)

    if score >= threshold:
        return {"allowed": True, "reason": "ok", "tags": []}

    tags = []
    if npc.stubbornness >= 0.7:
        tags.append("stubbornness")
    tags.extend(risk_tags)
    tags.extend(role_tags)
    if npc.disposition_to_player <= -2:
        tags.append("disposition")
    if not tags:
        tags = ["low_compliance"]

    reason = _refusal_reason(npc, risk_tags, role_tags)
    return {"allowed": False, "reason": reason, "tags": tags}


def apply_agency_gate(
    moves: List[NPCMove],
    state: GameState,
    world: WorldSpec,
    player_text: str,
    npc_dialogue_by_id: Dict[str, List[str]] | None = None,
) -> tuple[List[NPCMove], List[dict]]:
    allowed: List[NPCMove] = []
    events: List[dict] = []
    for move in moves:
        npc_texts = []
        if npc_dialogue_by_id:
            npc_texts = npc_dialogue_by_id.get(move.npc_id, [])
        if _npc_accepts_move(move, state, world, player_text, npc_texts):
            allowed.append(move)
            continue
        decision = decide_npc_move(move, state, world, player_text)
        if decision["allowed"]:
            allowed.append(move)
            continue
        events.append(
            {
                "type": "move_refused",
                "npc_id": move.npc_id,
                "from_location": move.from_location,
                "to_location": move.to_location,
                "reason": decision["reason"],
                "tags": decision["tags"],
            }
        )
    return allowed, events


def _find_npc(world: WorldSpec, npc_id: str) -> NPCProfile | None:
    for npc in world.npcs:
        if npc.npc_id == npc_id:
            return npc
    return None


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _base_threshold(player_text: str | None) -> float:
    threshold = 0.55
    if not player_text:
        return threshold
    text = player_text.lower()
    if any(token in text for token in ["must", "order", "now", "command", "immediately"]):
        threshold += 0.05
    return threshold


def _risk_alignment_penalty(
    npc: NPCProfile,
    world: WorldSpec,
    to_location: str,
) -> tuple[float, List[str]]:
    loc = world.get_location(to_location)
    if loc is None:
        return 0.0, []
    text = " ".join([loc.name, loc.kind, loc.description, " ".join(loc.tags)]).lower()
    risky_terms = ["forest", "ruins", "bandit", "dark", "cave", "dungeon", "bridge", "swamp"]
    risky = any(term in text for term in risky_terms)
    if not risky:
        return 0.05, ["risk_safe"]
    if npc.risk_tolerance >= 0.7:
        return 0.05, ["risk_tolerant"]
    if npc.risk_tolerance <= 0.3:
        return -0.2, ["risk"]
    return -0.1, ["risk"]


def _role_constraints_penalty(
    npc: NPCProfile,
    state: GameState,
    move: NPCMove,
) -> tuple[float, List[str]]:
    tags: List[str] = []
    prof = npc.profession.lower()
    traits = " ".join(npc.traits).lower()
    goals = " ".join(npc.goals).lower()

    anchored_roles = ["merchant", "shopkeeper", "innkeeper", "guard", "priest", "healer"]
    anchored_goals = ["protect", "guard", "keep shop", "keep", "watch", "avoid trouble"]
    anchored = any(role in prof for role in anchored_roles) or any(term in goals for term in anchored_goals)
    if "coward" in traits or "cautious" in traits:
        anchored = True
        tags.append("risk")

    if anchored and move.to_location != state.npc_locations.get(move.npc_id, ""):
        tags.append("role")
        return -0.25, tags
    return 0.0, tags


def _refusal_reason(npc: NPCProfile, risk_tags: List[str], role_tags: List[str]) -> str:
    if "role" in role_tags:
        return "Refused: guarding their post"
    if "risk" in risk_tags:
        return "Refused: too risky"
    if npc.disposition_to_player <= -2:
        return "Refused: doesn't trust the player"
    if npc.stubbornness >= 0.7:
        return "Refused: too stubborn"
    return "Refused: unwilling to comply"


def _npc_accepts_move(
    move: NPCMove,
    state: GameState,
    world: WorldSpec,
    player_text: str,
    npc_texts: List[str],
) -> bool:
    if not npc_texts:
        return False
    combined = " ".join([t for t in npc_texts if t]).strip()
    if not combined:
        return False
    if _npc_refuses_in_text(combined):
        return False
    loc = world.get_location(move.to_location)
    dest_names = {move.to_location}
    if loc and loc.name:
        dest_names.add(loc.name)
    dest_in_text = any(name and name in combined for name in dest_names)
    dest_in_player = any(name and name in player_text for name in dest_names) if player_text else False
    if not (dest_in_text or dest_in_player):
        return False
    return _npc_accepts_in_text(combined)


def _npc_accepts_in_text(text: str) -> bool:
    lower = text.lower()
    cues = [
        "我愿意",
        "我跟你去",
        "我会去",
        "我去",
        "我马上去",
        "我现在就去",
        "跟你走",
        "一起去",
        "好的",
        "可以",
        "没问题",
        "行",
        "当然",
        "let's go",
        "i will go",
        "i'll go",
        "i can go",
        "sure",
        "okay",
        "alright",
    ]
    for cue in cues:
        if cue.isascii():
            if cue in lower:
                return True
            continue
        if cue in text:
            return True
    return False


def _npc_refuses_in_text(text: str) -> bool:
    lower = text.lower()
    cues = [
        "不去",
        "不能",
        "不行",
        "不愿意",
        "没空",
        "拒绝",
        "做不到",
        "改天",
        "not going",
        "can't",
        "cannot",
        "won't",
        "refuse",
        "decline",
    ]
    for cue in cues:
        if cue.isascii():
            if cue in lower:
                return True
            continue
        if cue in text:
            return True
    return False
