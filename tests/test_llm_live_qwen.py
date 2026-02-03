"""Live Qwen integration test.

Run manually:
RUN_LIVE_LLM_TESTS=1 PYTHONPATH=. python -m pytest -q tests/test_llm_live_qwen.py
Requires DASHSCOPE_API_KEY in environment or .env (gitignored).
"""
from __future__ import annotations

import os
import re
import pytest


# Skip by default unless explicitly enabled
if os.getenv("RUN_LIVE_LLM_TESTS") != "1":
    pytest.skip("Live LLM tests disabled (set RUN_LIVE_LLM_TESTS=1)", allow_module_level=True)

try:
    from openai import OpenAI  # noqa: F401
except Exception:
    pytest.skip("openai package not available", allow_module_level=True)

from rpg_story.config import load_config
from rpg_story.llm.client import QwenOpenAICompatibleClient

cfg = load_config("configs/config.yaml")
key_env = cfg.llm.api_key_env or "DASHSCOPE_API_KEY"
if not os.getenv(key_env):
    # After load_config(), .env should be loaded if present
    pytest.skip("Missing DASHSCOPE_API_KEY after load_config()", allow_module_level=True)


def test_live_qwen_generate_json():
    client = QwenOpenAICompatibleClient(cfg)

    system = "You are a helpful assistant. Output ONLY valid JSON."
    user = (
        "Return JSON with keys: ok (bool), model_used (string), "
        "nonce (8-digit numeric string). nonce must be random."
    )
    schema_hint = '{"ok": true, "model_used": "string", "nonce": "########"}'

    result = client.generate_json(system, user, schema_hint=schema_hint)
    assert isinstance(result, dict)
    assert "ok" in result
    assert "nonce" in result
    assert bool(result["ok"]) is True or bool(result["ok"]) is False
    assert re.match(r"^\d{8}$", str(result["nonce"]))
