import pytest

from rpg_story.config import load_config
from rpg_story.llm.client import (
    _extract_json,
    MockLLMClient,
    QwenOpenAICompatibleClient,
    OpenAI,
)


def test_extract_json():
    text = "prefix {\"a\": 1} suffix"
    extracted = _extract_json(text)
    assert extracted == '{"a": 1}'


def test_generate_json_valid_first():
    client = MockLLMClient(["{\"ok\": true}"])
    out = client.generate_json("sys", "user")
    assert out["ok"] is True
    assert client.calls == 1


def test_generate_json_repair_once():
    client = MockLLMClient(["not json", "{\"ok\": true}"])
    out = client.generate_json("sys", "user")
    assert out["ok"] is True
    assert client.calls == 2


def test_generate_json_repair_fails():
    client = MockLLMClient(["not json", "still not json"])
    with pytest.raises(ValueError):
        client.generate_json("sys", "user")
    assert client.calls == 2


def test_qwen_client_missing_api_key(monkeypatch):
    if OpenAI is None:
        pytest.skip("openai package not available")
    monkeypatch.setenv("DISABLE_DOTENV", "1")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    # In case a custom env var is set
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    cfg = load_config("configs/config.yaml")
    client = QwenOpenAICompatibleClient(cfg)
    with pytest.raises(ValueError):
        client.generate_json("sys", "user")
