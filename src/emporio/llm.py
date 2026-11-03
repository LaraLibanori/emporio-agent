"""Adaptadores de LLM com function calling.

Formato interno de mensagens = formato "chat completions" da OpenAI, que virou
padrão de fato (Gemini, Groq, OpenRouter, Ollama, vLLM expõem endpoints
compatíveis). Assim, UM adaptador cobre quase todos os provedores. A Anthropic
tem adaptador próprio que converte para o formato de content blocks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .config import Settings


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)

    def as_message(self) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                }
                for tc in self.tool_calls
            ]
        return msg


class LLM(Protocol):
    def chat(self, system: str, messages: list[dict], tools: list[dict], force_text: bool = False) -> LLMResponse:
        """force_text=True: ferramentas continuam declaradas (exigido quando o histórico
        tem tool calls), mas o modelo é obrigado a responder em texto."""
        ...


def _parse_args(raw: str | dict | None) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"_raw": raw}


class OpenAICompatibleLLM:
    def __init__(self, settings: Settings):
        from openai import OpenAI

        self.client = OpenAI(api_key=settings.api_key or "not-needed", base_url=settings.base_url)
        self.model, self.temperature = settings.model, settings.temperature

    def chat(self, system: str, messages: list[dict], tools: list[dict], force_text: bool = False) -> LLMResponse:
        kwargs = {}
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
            if force_text:
                kwargs["tool_choice"] = "none"
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "system", "content": system}, *messages],
            **kwargs,
        )
        msg = resp.choices[0].message
        calls = [
            ToolCall(tc.id or f"call_{i}", tc.function.name, _parse_args(tc.function.arguments))
            for i, tc in enumerate(msg.tool_calls or [])
        ]
        usage = {}
        if resp.usage:
            usage = {"input": resp.usage.prompt_tokens, "output": resp.usage.completion_tokens}
        return LLMResponse(msg.content, calls, usage)


class AnthropicLLM:
    def __init__(self, settings: Settings):
        import anthropic

        self.client = anthropic.Anthropic(api_key=settings.api_key)
        self.model, self.temperature = settings.model, settings.temperature

    @staticmethod
    def _convert(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            elif m["role"] == "assistant" and m.get("tool_calls"):
                blocks: list[dict] = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                for tc in m["tool_calls"]:
                    blocks.append(
                        {"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"],
                         "input": _parse_args(tc["function"]["arguments"])}
                    )
                out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": m["role"], "content": m.get("content") or ""})
        return out

    def chat(self, system: str, messages: list[dict], tools: list[dict], force_text: bool = False) -> LLMResponse:
        kwargs = {}
        if tools:
            kwargs["tools"] = [
                {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools
            ]
            if force_text:
                kwargs["tool_choice"] = {"type": "none"}
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=self.temperature,
            system=system,
            messages=self._convert(messages),
            **kwargs,
        )
        text = "".join(b.text for b in resp.content if b.type == "text") or None
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
        return LLMResponse(text, calls, {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens})


class ScriptedLLM:
    """LLM falso para testes: recebe uma função (system, messages) -> LLMResponse."""

    def __init__(self, fn: Callable[[str, list[dict]], LLMResponse]):
        self.fn = fn
        self.calls: list[list[dict]] = []
        self.force_text_flags: list[bool] = []

    def chat(self, system: str, messages: list[dict], tools: list[dict], force_text: bool = False) -> LLMResponse:
        self.calls.append(list(messages))
        self.force_text_flags.append(force_text)
        return self.fn(system, messages)


def build_llm(settings: Settings | None = None) -> LLM:
    settings = settings or Settings()
    if settings.provider == "anthropic":
        return AnthropicLLM(settings)
    if settings.provider in ("openai", "openai-compatible", "gemini", "groq", "ollama", "openrouter"):
        return OpenAICompatibleLLM(settings)
    raise ValueError(f"LLM_PROVIDER desconhecido: {settings.provider}")
