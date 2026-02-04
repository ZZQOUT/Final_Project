"""Lightweight helpers for location content."""
from __future__ import annotations

import re


def create_location_id(index: int) -> str:
    return f"loc_{index:03d}"


def normalize_location_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip())
