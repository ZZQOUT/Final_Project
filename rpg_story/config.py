"""Configuration loader for the RPG prototype."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import os
import yaml
from dotenv import load_dotenv


@dataclass
class Config:
    model: Dict[str, Any]
    retrieval: Dict[str, Any]
    safety: Dict[str, Any]
    logging: Dict[str, Any]
    ui: Dict[str, Any]


def load_config(config_path: str | Path) -> Config:
    load_dotenv()
    path = Path(config_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    # Allow env overrides
    data.setdefault("model", {})
    data["model"].setdefault("api_key", os.getenv("LLM_API_KEY", ""))
    data["model"].setdefault("base_url", os.getenv("LLM_BASE_URL", ""))
    data["model"].setdefault("name", os.getenv("LLM_MODEL", ""))
    data["model"].setdefault("embedding_model", os.getenv("EMBEDDING_MODEL", ""))
    return Config(
        model=data.get("model", {}),
        retrieval=data.get("retrieval", {}),
        safety=data.get("safety", {}),
        logging=data.get("logging", {}),
        ui=data.get("ui", {}),
    )
