"""Manual smoke test for Qwen DashScope OpenAI-compatible API."""
from __future__ import annotations

import os

from rpg_story.config import load_config
from rpg_story.llm.client import QwenOpenAICompatibleClient


def main() -> None:
    cfg = load_config("configs/config.yaml")
    if not os.getenv("DASHSCOPE_API_KEY") and cfg.llm.api_key_env == "DASHSCOPE_API_KEY":
        print("Set DASHSCOPE_API_KEY in .env or environment before running.")
        return

    client = QwenOpenAICompatibleClient(cfg)
    system = "You are a helpful assistant."
    user = (
        "Return ONLY valid JSON with keys ok (bool), model (string), nonce (string). "
        "nonce MUST be a random 8-digit numeric string DIFFERENT on each run. "
        "Do NOT use 12345678."
    )
    schema_hint = (
        'Return JSON only with keys {"ok": bool, "model": string, "nonce": string}. '
        'nonce must be an 8-digit numeric string and must not be "12345678".'
    )
    print("Using cfg.llm.model =", cfg.llm.model)
    result = client.generate_json(system, user, schema_hint=schema_hint)
    print(result)


if __name__ == "__main__":
    main()
