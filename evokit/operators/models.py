"""Language models used by LLM operators. No SDK dependencies."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from evokit.errors import UsageError


@dataclass(frozen=True)
class Completion:
    text: str
    usage: dict[str, float] = field(default_factory=dict)


class Model(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> Completion: ...

    async def chat(self, messages: list[dict[str, str]]) -> Completion:
        """The next assistant message of a conversation in OpenAI's format ({"role", "content"}).
        By default, the conversation joined into one prompt for `generate`."""
        return await self.generate("\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages))

    async def check(self) -> None:
        """Raise UsageError if the model cannot answer. A no-op by default."""


async def _ask_ok(model: "OpenAIModel | AnthropicModel") -> None:
    """One tiny request with the model's own settings; its cost is not recorded."""
    try:
        await model.generate("Reply with OK.")
    except Exception as exc:
        raise UsageError(f"{model.model} cannot answer: {exc}", hint=f"check ${model.api_key_env} and the model name") from None


def _cost(price: tuple[float, float] | None, tokens_in: int, tokens_out: int, reported: Any = None) -> dict[str, float]:
    """`reported` is a cost the endpoint returned itself (OpenRouter's `usage.cost`), used if there is no `price`."""
    usage: dict[str, float] = {"tokens": tokens_in + tokens_out, "input_tokens": tokens_in, "output_tokens": tokens_out}
    if price:
        usage["cost_usd"] = (tokens_in * price[0] + tokens_out * price[1]) / 1_000_000
    elif isinstance(reported, (int, float)) and not isinstance(reported, bool):
        usage["cost_usd"] = float(reported)
    return usage


def _post(url: str, headers: dict, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json", **headers})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{url} returned {exc.code}: {exc.read().decode(errors='replace')[:2000]}") from exc


class OpenAIModel(Model):
    """Any OpenAI-compatible /chat/completions endpoint. `price` is ($/Mtok in, $/Mtok out)."""

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        price: tuple[float, float] | None = None,
        system: str | None = None,
        timeout: float = 600,
        **params: Any,
    ):
        self.model, self.base_url, self.api_key_env = model, base_url.rstrip("/"), api_key_env
        self.price, self.system, self.timeout, self.params = price, system, timeout, params

    async def check(self) -> None:
        await _ask_ok(self)

    async def generate(self, prompt: str) -> Completion:
        return await self.chat([{"role": "user", "content": prompt}])

    async def chat(self, messages: list[dict[str, str]]) -> Completion:
        if self.system and not any(m["role"] == "system" for m in messages):
            messages = [{"role": "system", "content": self.system}, *messages]
        headers = {"authorization": f"Bearer {os.environ.get(self.api_key_env, '')}"}
        data = await asyncio.to_thread(
            _post, f"{self.base_url}/chat/completions", headers, {"model": self.model, "messages": messages, **self.params}, self.timeout
        )
        usage = data.get("usage") or {}
        return Completion(
            data["choices"][0]["message"]["content"] or "",
            _cost(self.price, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), usage.get("cost")),
        )


class AnthropicModel(Model):
    """Anthropic Messages API. `price` is ($/Mtok in, $/Mtok out)."""

    def __init__(
        self,
        model: str,
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_tokens: int = 16_000,
        price: tuple[float, float] | None = None,
        system: str | None = None,
        timeout: float = 600,
        base_url: str = "https://api.anthropic.com",
    ):
        self.model, self.api_key_env, self.max_tokens = model, api_key_env, max_tokens
        self.price, self.system, self.timeout, self.base_url = price, system, timeout, base_url.rstrip("/")

    async def check(self) -> None:
        await _ask_ok(self)

    async def generate(self, prompt: str) -> Completion:
        return await self.chat([{"role": "user", "content": prompt}])

    async def chat(self, messages: list[dict[str, str]]) -> Completion:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or self.system
        turns = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
        body: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens, "messages": turns}
        if system:
            body["system"] = system
        headers = {"x-api-key": os.environ.get(self.api_key_env, ""), "anthropic-version": "2023-06-01"}
        data = await asyncio.to_thread(_post, f"{self.base_url}/v1/messages", headers, body, self.timeout)
        text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
        usage = data.get("usage") or {}
        return Completion(text, _cost(self.price, usage.get("input_tokens", 0), usage.get("output_tokens", 0)))


class FunctionModel(Model):
    """Wrap `fn(prompt) -> str | Completion` (sync or async). For tests and custom clients."""

    def __init__(self, fn: Callable[[str], Any]):
        self.fn = fn

    async def generate(self, prompt: str) -> Completion:
        result = self.fn(prompt)
        if inspect.isawaitable(result):
            result = await result
        return result if isinstance(result, Completion) else Completion(str(result))
