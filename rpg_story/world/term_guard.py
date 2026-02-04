"""Guard against first-mention anachronisms in NPC output."""
from __future__ import annotations

from typing import Set
import re

DEFAULT_ANACHRONISM_TERMS = [
    "smartphone",
    "phone",
    "internet",
    "wifi",
    "wi-fi",
    "email",
    "app",
    "credit card",
    "gun",
    "ak-47",
    "gps",
    "browser",
    "website",
    "electricity",
]

_NORMALIZE_MAP = {
    "wi-fi": "wifi",
    "wifi": "wifi",
}


def extract_terms(text: str, terms: list[str]) -> Set[str]:
    if not text:
        return set()
    matches: Set[str] = set()
    for term in terms:
        if not term:
            continue
        normalized = _normalize_term(term)
        if normalized == "wifi":
            if re.search(r"\bwi[-\s]?fi\b", text, flags=re.IGNORECASE):
                matches.add("wifi")
            continue
        if _is_alnum(term):
            pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
            if pattern.search(text):
                matches.add(normalized)
        else:
            if term.lower() in text.lower():
                matches.add(normalized)
    return matches


def detect_first_mention(player_text: str, npc_texts: list[str], terms: list[str]) -> Set[str]:
    player_terms = extract_terms(player_text, terms)
    npc_terms: Set[str] = set()
    for text in npc_texts:
        npc_terms |= extract_terms(text, terms)
    return npc_terms - player_terms


def _normalize_term(term: str) -> str:
    lowered = term.lower()
    return _NORMALIZE_MAP.get(lowered, lowered)


def _is_alnum(term: str) -> bool:
    lowered = term.lower()
    return re.fullmatch(r"[a-z0-9]+", lowered) is not None
