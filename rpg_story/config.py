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
    chunk_size_chars: int
    chunk_overlap_chars: int
    retrieval_backend: str
    embedding_provider: str
    embedding_model: str
    embedding_dim: int
    vector_weight: float
    lexical_weight: float
    recency_weight: float
    min_score: float


@dataclass(frozen=True)
class WorldGenSection:
    max_retries: int
    strict_consistency: bool
    enforce_bidirectional_edges: bool
    banned_keywords: list[str]
    strict_bidirectional_edges: bool
    max_rewrite_attempts: int
    locations_min: int
    locations_max: int
    npcs_min: int
    npcs_max: int


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
    if os.getenv("DISABLE_DOTENV") == "1":
        return
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    if Path(".env").exists():
        load_dotenv(dotenv_path=Path(".env"), override=False)


def _read_mapping_value(mapping: Any, key: str) -> Any:
    try:
        return mapping.get(key)
    except Exception:
        pass
    try:
        return mapping[key]
    except Exception:
        return None


def _as_non_empty_str(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _get_streamlit_secrets() -> Any | None:
    try:
        import streamlit as st  # type: ignore
    except Exception:
        return None
    try:
        return st.secrets
    except Exception:
        return None


def _streamlit_top_level(secrets_obj: Any | None, *keys: str) -> str:
    if secrets_obj is None:
        return ""
    for key in keys:
        if not key:
            continue
        value = _as_non_empty_str(_read_mapping_value(secrets_obj, key))
        if value:
            return value
    return ""


def _streamlit_section(secrets_obj: Any | None, section: str, *keys: str) -> str:
    if secrets_obj is None or not section:
        return ""
    section_obj = _read_mapping_value(secrets_obj, section)
    if section_obj is None:
        return ""
    for key in keys:
        if not key:
            continue
        value = _as_non_empty_str(_read_mapping_value(section_obj, key))
        if value:
            return value
    return ""


def _first_non_empty(*values: str) -> str:
    for value in values:
        text = str(value).strip()
        if text:
            return text
    return ""


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
    secrets_obj = _get_streamlit_secrets()

    # Environment overrides (base URL / model)
    env_base_url = _first_non_empty(
        os.getenv("LLM_BASE_URL", ""),
        os.getenv("BASE_URL", ""),
        _streamlit_top_level(secrets_obj, "LLM_BASE_URL", "BASE_URL"),
        _streamlit_section(secrets_obj, "llm", "base_url", "BASE_URL"),
        _streamlit_section(secrets_obj, "openai", "base_url", "BASE_URL"),
        _streamlit_section(secrets_obj, "dashscope", "base_url", "BASE_URL"),
    )
    env_model = _first_non_empty(
        os.getenv("LLM_MODEL", ""),
        os.getenv("MODEL_NAME", ""),
        _streamlit_top_level(secrets_obj, "LLM_MODEL", "MODEL_NAME"),
        _streamlit_section(secrets_obj, "llm", "model", "MODEL_NAME"),
        _streamlit_section(secrets_obj, "openai", "model", "MODEL_NAME"),
        _streamlit_section(secrets_obj, "dashscope", "model", "MODEL_NAME"),
    )

    llm_base_url = env_base_url or str(llm_cfg.get("base_url", ""))
    llm_model = env_model or str(llm_cfg.get("model", ""))
    api_key_env = str(llm_cfg.get("api_key_env", "DASHSCOPE_API_KEY"))
    api_key = _first_non_empty(
        os.getenv(api_key_env, ""),
        os.getenv("DASHSCOPE_API_KEY", "") if api_key_env != "DASHSCOPE_API_KEY" else "",
        os.getenv("OPENAI_API_KEY", ""),
        _streamlit_top_level(secrets_obj, api_key_env),
        _streamlit_top_level(secrets_obj, "DASHSCOPE_API_KEY") if api_key_env != "DASHSCOPE_API_KEY" else "",
        _streamlit_top_level(secrets_obj, "OPENAI_API_KEY"),
        _streamlit_section(secrets_obj, "llm", "api_key", "API_KEY"),
        _streamlit_section(secrets_obj, "openai", "api_key", "API_KEY"),
        _streamlit_section(secrets_obj, "dashscope", "api_key", "API_KEY"),
    )

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
        chunk_size_chars=int(rag_cfg.get("chunk_size_chars", 700)),
        chunk_overlap_chars=int(rag_cfg.get("chunk_overlap_chars", 120)),
        retrieval_backend=str(rag_cfg.get("retrieval_backend", "persistent_hybrid")),
        embedding_provider=str(rag_cfg.get("embedding_provider", "hashing")),
        embedding_model=str(rag_cfg.get("embedding_model", "")),
        embedding_dim=int(rag_cfg.get("embedding_dim", 384)),
        vector_weight=float(rag_cfg.get("vector_weight", 0.60)),
        lexical_weight=float(rag_cfg.get("lexical_weight", 0.35)),
        recency_weight=float(rag_cfg.get("recency_weight", 0.05)),
        min_score=float(rag_cfg.get("min_score", 0.03)),
    )

    worldgen = WorldGenSection(
        max_retries=int(worldgen_cfg.get("max_retries", 2)),
        strict_consistency=bool(worldgen_cfg.get("strict_consistency", True)),
        enforce_bidirectional_edges=bool(worldgen_cfg.get("enforce_bidirectional_edges", False)),
        banned_keywords=list(worldgen_cfg.get("banned_keywords", [])),
        strict_bidirectional_edges=bool(worldgen_cfg.get("strict_bidirectional_edges", False)),
        max_rewrite_attempts=int(worldgen_cfg.get("max_rewrite_attempts", 1)),
        locations_min=int(worldgen_cfg.get("locations_min", 3)),
        locations_max=int(worldgen_cfg.get("locations_max", 8)),
        npcs_min=int(worldgen_cfg.get("npcs_min", 2)),
        npcs_max=int(worldgen_cfg.get("npcs_max", 8)),
    )

    logging = LoggingSection(
        level=str(logging_cfg.get("level", "INFO")),
        log_jsonl=bool(logging_cfg.get("log_jsonl", True)),
    )

    return AppConfig(app=app, llm=llm, rag=rag, worldgen=worldgen, logging=logging)
