"""Configuration loader and typed config objects."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict
import os
import yaml


@dataclass(frozen=True)
class AppSection:
    name: str
    env: str
    data_dir: Path
    sessions_dir: Path
    worlds_dir: Path
    vectorstore_dir: Path


@dataclass(frozen=True)
class LLMSection:
    provider: str
    base_url: str
    model: str
    api_key_env: str
    api_key: str
    timeout_seconds: int
    max_retries: int
    temperature: float
    top_p: float


@dataclass(frozen=True)
class RAGSection:
    enabled: bool
    top_k: int
    summary_window: int


@dataclass(frozen=True)
class WorldGenSection:
    max_retries: int
    strict_consistency: bool
    enforce_bidirectional_edges: bool


@dataclass(frozen=True)
class LoggingSection:
    level: str
    log_jsonl: bool


@dataclass(frozen=True)
class AppConfig:
    app: AppSection
    llm: LLMSection
    rag: RAGSection
    worldgen: WorldGenSection
    logging: LoggingSection

    def resolve_paths(self, project_root: Path) -> "AppConfig":
        """Return a copy with app paths resolved to absolute paths."""
        app = self.app
        resolved = replace(
            app,
            data_dir=(project_root / app.data_dir).resolve() if not app.data_dir.is_absolute() else app.data_dir,
            sessions_dir=(project_root / app.sessions_dir).resolve()
            if not app.sessions_dir.is_absolute()
            else app.sessions_dir,
            worlds_dir=(project_root / app.worlds_dir).resolve() if not app.worlds_dir.is_absolute() else app.worlds_dir,
            vectorstore_dir=(project_root / app.vectorstore_dir).resolve()
            if not app.vectorstore_dir.is_absolute()
            else app.vectorstore_dir,
        )
        return replace(self, app=resolved)


def _load_env() -> None:
    """Load .env if python-dotenv is available. No-op if missing."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    if Path(".env").exists():
        load_dotenv(dotenv_path=Path(".env"), override=False)


def _require_section(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    if key not in cfg or not isinstance(cfg[key], dict):
        raise ValueError(f"Missing or invalid config section: {key}")
    return cfg[key]


def load_config(config_path: str = "configs/config.yaml") -> AppConfig:
    """Load YAML config, apply env overrides, return typed AppConfig."""
    _load_env()

    path = Path(config_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")

    app_cfg = _require_section(data, "app")
    llm_cfg = _require_section(data, "llm")
    rag_cfg = _require_section(data, "rag")
    worldgen_cfg = _require_section(data, "worldgen")
    logging_cfg = _require_section(data, "logging")

    # Environment overrides
    env_api_key = os.getenv("LLM_API_KEY", "")
    env_base_url = os.getenv("LLM_BASE_URL", "")
    env_model = os.getenv("LLM_MODEL", "")

    llm_base_url = env_base_url or str(llm_cfg.get("base_url", ""))
    llm_model = env_model or str(llm_cfg.get("model", ""))
    api_key_env = str(llm_cfg.get("api_key_env", "LLM_API_KEY"))
    api_key = env_api_key or os.getenv(api_key_env, "")

    app = AppSection(
        name=str(app_cfg.get("name", "app")),
        env=str(app_cfg.get("env", "dev")),
        data_dir=Path(str(app_cfg.get("data_dir", "data"))),
        sessions_dir=Path(str(app_cfg.get("sessions_dir", "data/sessions"))),
        worlds_dir=Path(str(app_cfg.get("worlds_dir", "data/worlds"))),
        vectorstore_dir=Path(str(app_cfg.get("vectorstore_dir", "data/vectorstore"))),
    )

    llm = LLMSection(
        provider=str(llm_cfg.get("provider", "openai_compatible")),
        base_url=llm_base_url,
        model=llm_model,
        api_key_env=api_key_env,
        api_key=api_key,
        timeout_seconds=int(llm_cfg.get("timeout_seconds", 30)),
        max_retries=int(llm_cfg.get("max_retries", 2)),
        temperature=float(llm_cfg.get("temperature", 0.7)),
        top_p=float(llm_cfg.get("top_p", 0.95)),
    )

    rag = RAGSection(
        enabled=bool(rag_cfg.get("enabled", True)),
        top_k=int(rag_cfg.get("top_k", 5)),
        summary_window=int(rag_cfg.get("summary_window", 3)),
    )

    worldgen = WorldGenSection(
        max_retries=int(worldgen_cfg.get("max_retries", 2)),
        strict_consistency=bool(worldgen_cfg.get("strict_consistency", True)),
        enforce_bidirectional_edges=bool(worldgen_cfg.get("enforce_bidirectional_edges", False)),
    )

    logging = LoggingSection(
        level=str(logging_cfg.get("level", "INFO")),
        log_jsonl=bool(logging_cfg.get("log_jsonl", True)),
    )

    return AppConfig(app=app, llm=llm, rag=rag, worldgen=worldgen, logging=logging)
