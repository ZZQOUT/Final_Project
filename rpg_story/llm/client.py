"""LLM API client (OpenAI-compatible, retryable, JSON-repairable)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json
import os
import random
import time
import warnings

from rpg_story.config import AppConfig

try:
    from openai import OpenAI
    from openai import APIConnectionError, APITimeoutError, RateLimitError, APIStatusError
except Exception:  # pragma: no cover - openai dependency missing at import time
    OpenAI = None  # type: ignore
    APIConnectionError = APITimeoutError = RateLimitError = APIStatusError = Exception  # type: ignore


def _extract_json(text: str) -> str | None:
    """Extract the first JSON object substring from text if present."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _truncate(text: str, n: int = 500) -> str:
    """Truncate text to n chars for error messages."""
    if len(text) <= n:
        return text
    return text[:n] + "..."


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse JSON from raw text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        extracted = _extract_json(text)
        if not extracted:
            return None
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            return None


class BaseLLMClient(ABC):
    """Base LLM client interface."""

    @abstractmethod
    def generate_text(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        schema_hint: str | None = None,
        response_format: dict | None = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


def make_json_schema_response_format(
    name: str,
    schema: dict,
    description: str | None = None,
    strict: bool = True,
) -> dict:
    rf = {"type": "json_schema", "json_schema": {"name": name, "schema": schema}, "strict": strict}
    if description:
        rf["json_schema"]["description"] = description
    return rf


class QwenOpenAICompatibleClient(BaseLLMClient):
    """Qwen client via DashScope OpenAI-compatible API."""

    def __init__(self, config: AppConfig) -> None:
        if OpenAI is None:
            raise ImportError("openai package is required")
        self.config = config
        self.api_key = self._read_api_key(config)
        self.base_url = config.llm.base_url
        self.model = config.llm.model
        if not self.base_url:
            raise ValueError("Missing LLM base_url (set LLM_BASE_URL or config llm.base_url)")
        if not self.model:
            raise ValueError("Missing LLM model (set LLM_MODEL or config llm.model)")
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @staticmethod
    def _read_api_key(config: AppConfig) -> str:
        api_key_env = config.llm.api_key_env or "DASHSCOPE_API_KEY"
        api_key = os.getenv(api_key_env, "")
        if not api_key and api_key_env != "DASHSCOPE_API_KEY":
            fallback = os.getenv("DASHSCOPE_API_KEY", "")
            if fallback:
                warnings.warn(
                    f"{api_key_env} not set; falling back to DASHSCOPE_API_KEY",
                    RuntimeWarning,
                )
                api_key = fallback
        return api_key

    def _require_api_key(self) -> None:
        if not self.api_key:
            raise ValueError("Missing DASHSCOPE_API_KEY (set it in .env or environment)")

    def _should_retry(self, exc: Exception) -> bool:
        if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError)):
            return True
        if isinstance(exc, APIStatusError):
            status = getattr(exc, "status_code", None)
            if status is not None and (status == 429 or 500 <= status <= 599):
                return True
        return False

    def _request_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        top_p: float,
        response_format: dict | None = None,
    ) -> str:
        self._require_api_key()
        max_retries = max(1, int(self.config.llm.max_retries))
        base_delay = 0.5
        cap_delay = 3.0
        for attempt in range(max_retries + 1):
            try:
                request_kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stream": False,
                }
                if response_format is not None:
                    request_kwargs["response_format"] = response_format
                completion = self._client.chat.completions.create(**request_kwargs)
                return completion.choices[0].message.content or ""
            except Exception as exc:  # pragma: no cover - external errors
                if attempt >= max_retries or not self._should_retry(exc):
                    raise
                delay = min(cap_delay, base_delay * (2 ** attempt) + random.uniform(0, base_delay))
                time.sleep(delay)
        return ""

    def generate_text(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        temp = self.config.llm.temperature if temperature is None else temperature
        top = self.config.llm.top_p if top_p is None else top_p
        return self._request_chat(messages, temp, top)

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        schema_hint: str | None = None,
        response_format: dict | None = None,
    ) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        temp = self.config.llm.temperature
        top = self.config.llm.top_p
        text = self._request_chat(messages, temp, top, response_format=response_format)
        parsed = _parse_json(text)
        if parsed is not None:
            return parsed

        # One-time repair attempt
        repair_system = "You are a JSON repair tool. Return ONLY valid JSON. No markdown. No commentary."
        repair_user = "Original text:\n" + text
        if schema_hint:
            repair_user += "\n\nSchema hint:\n" + schema_hint
        repair_messages = [
            {"role": "system", "content": repair_system},
            {"role": "user", "content": repair_user},
        ]
        repaired_text = self._request_chat(
            repair_messages,
            0.0,
            1.0,
            response_format=response_format,
        )
        repaired = _parse_json(repaired_text)
        if repaired is not None:
            return repaired

        preview = _truncate(text)
        raise ValueError(f"Invalid JSON after repair attempt. Preview: {preview}")


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for tests (offline)."""

    def __init__(self, outputs: List[str]) -> None:
        self.outputs = list(outputs)
        self.calls = 0
        self.last_schema_hint: str | None = None
        self.last_response_format: dict | None = None

    def generate_text(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        self.calls += 1
        if not self.outputs:
            raise RuntimeError("MockLLMClient has no more outputs")
        return self.outputs.pop(0)

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        schema_hint: str | None = None,
        response_format: dict | None = None,
    ) -> Dict[str, Any]:
        self.last_schema_hint = schema_hint
        self.last_response_format = response_format
        text = self.generate_text([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        parsed = _parse_json(text)
        if parsed is not None:
            return parsed

        repair_system = "You are a JSON repair tool. Return ONLY valid JSON. No markdown. No commentary."
        repair_user = "Original text:\n" + text
        if schema_hint:
            repair_user += "\n\nSchema hint:\n" + schema_hint
        repaired_text = self.generate_text([
            {"role": "system", "content": repair_system},
            {"role": "user", "content": repair_user},
        ])
        repaired = _parse_json(repaired_text)
        if repaired is not None:
            return repaired

        preview = _truncate(text)
        raise ValueError(f"Invalid JSON after repair attempt. Preview: {preview}")
