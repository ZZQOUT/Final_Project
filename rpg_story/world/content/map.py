"""Lightweight helpers for map connectivity."""
from __future__ import annotations


def normalize_connections(connections: list[str]) -> list[str]:
    # Remove duplicates while preserving order
    seen = set()
    result = []
    for c in connections:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result
