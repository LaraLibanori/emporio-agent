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
    raw_parts: list[dict] | None = None  # Gemini: partes originais (com thoughtSignature)

    def as_message(self) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.raw_parts:
            msg["_gemini_parts"] = self.raw_parts
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


def _public(m: dict) -> dict:
    """Remove campos internos (ex.: _gemini_parts) antes de enviar a outro provedor."""
    return {k: v for k, v in m.items() if not k.startswith("_")}


def _openai_function(spec: dict) -> dict:
    """Gemini (endpoint compatível) rejeita `parameters` com `properties` vazio (HTTP 400).
    Para ferramentas sem argumentos, omitimos `parameters` (aceito por OpenAI e Gemini)."""
    if not spec.get("parameters", {}).get("properties"):
        return {"name": spec["name"], "description": spec["description"]}
    return spec


def describe_error(e: Exception) -> str:
    """Extrai a mensagem real do provedor (o SDK às vezes mostra só 'Error code: 400')."""
    body = getattr(e, "body", None)
    if body:
        if isinstance(body, list) and body:
            body = body[0]
        if isinstance(body, dict):
            err = body.get("error", body)
            if isinstance(err, dict):
                return f"{getattr(e, 'status_code', '')} {err.get('status', '')} {err.get('message', err)}".strip()
            return f"{getattr(e, 'status_code', '')} {err}".strip()
    resp = getattr(e, "response", None)
    if resp is not None:
        try:
            return f"{resp.status_code} {resp.text[:500]}"
        except Exception:
            pass
    return f"{type(e).__name__}: {e}"


class OpenAICompatibleLLM:
    def __init__(self, settings: Settings):
        from openai import OpenAI

        self.client = OpenAI(api_key=settings.api_key or "not-needed", base_url=settings.base_url)
        self.model, self.temperature = settings.model, settings.temperature

    def chat(self, system: str, messages: list[dict], tools: list[dict], force_text: bool = False) -> LLMResponse:
        kwargs = {}
        if tools:
            kwargs["tools"] = [{"type": "function", "function": _openai_function(t)} for t in tools]
            if force_text:
                kwargs["tool_choice"] = "none"
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "system", "content": system}, *[_public(m) for m in messages]],
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


class GeminiLLM:
    """API nativa do Google Gemini (REST generateContent), sem SDK extra.

    Por que além do endpoint compatível com OpenAI? Algumas chaves do Google
    (ex.: as novas chaves "AQ.") funcionam na API nativa mas não no endpoint
    /openai. A API nativa também devolve `thoughtSignature` nos function calls,
    que precisa voltar no histórico para os modelos Gemini com raciocínio.
    """

    DEFAULT_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, settings: Settings, client=None):
        import httpx

        base = settings.base_url or self.DEFAULT_URL
        if "/openai" in base:  # .env antigo apontando para o endpoint compatível
            base = self.DEFAULT_URL
        self.url = f"{base.rstrip('/')}/models/{settings.model}:generateContent"
        self.headers = {"x-goog-api-key": settings.api_key or "", "content-type": "application/json"}
        self.temperature = settings.temperature
        self.http = client or httpx.Client(timeout=60)

    @staticmethod
    def _declaration(spec: dict) -> dict:
        d = {"name": spec["name"], "description": spec["description"]}
        if spec.get("parameters", {}).get("properties"):
            d["parameters"] = spec["parameters"]
        return d

    @staticmethod
    def _convert(messages: list[dict]) -> list[dict]:
        contents: list[dict] = []

        def push(role: str, part: dict):
            if contents and contents[-1]["role"] == role and role == "user" and "functionResponse" in part \
                    and all("functionResponse" in p for p in contents[-1]["parts"]):
                contents[-1]["parts"].append(part)
            else:
                contents.append({"role": role, "parts": [part]})

        for m in messages:
            if m["role"] == "user":
                push("user", {"text": m.get("content") or ""})
            elif m["role"] == "assistant":
                if m.get("_gemini_parts"):
                    contents.append({"role": "model", "parts": m["_gemini_parts"]})
                    continue
                parts = [{"text": m["content"]}] if m.get("content") else []
                for tc in m.get("tool_calls") or []:
                    parts.append({"functionCall": {"name": tc["function"]["name"],
                                                   "args": _parse_args(tc["function"]["arguments"])}})
                contents.append({"role": "model", "parts": parts or [{"text": ""}]})
            elif m["role"] == "tool":
                try:
                    resp = json.loads(m["content"])
                except (json.JSONDecodeError, TypeError):
                    resp = {"resultado": m["content"]}
                if not isinstance(resp, dict):
                    resp = {"resultado": resp}
                push("user", {"functionResponse": {"name": m.get("name", "tool"), "response": resp}})
        return contents

    def chat(self, system: str, messages: list[dict], tools: list[dict], force_text: bool = False) -> LLMResponse:
        body: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": self._convert(messages),
            "generationConfig": {"temperature": self.temperature},
        }
        if tools:
            body["tools"] = [{"functionDeclarations": [self._declaration(t) for t in tools]}]
            body["toolConfig"] = {"functionCallingConfig": {"mode": "NONE" if force_text else "AUTO"}}
        r = self.http.post(self.url, headers=self.headers, json=body)
        if r.status_code >= 400:
            raise GeminiError(r.status_code, r.text)
        data = r.json()
        cands = data.get("candidates") or []
        parts = (cands[0].get("content") or {}).get("parts", []) if cands else []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought")) or None
        calls = [
            ToolCall(f"call_{i}", p["functionCall"]["name"], dict(p["functionCall"].get("args") or {}))
            for i, p in enumerate(parts) if "functionCall" in p
        ]
        meta = data.get("usageMetadata") or {}
        resp = LLMResponse(text, calls, {"input": meta.get("promptTokenCount", 0),
                                         "output": meta.get("candidatesTokenCount", 0)})
        resp.raw_parts = [p for p in parts if not p.get("thought")] if calls else None
        return resp


class GeminiError(Exception):
    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        try:
            self.body = json.loads(text)
        except json.JSONDecodeError:
            self.body = text[:500]
        super().__init__(f"Gemini HTTP {status_code}")


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
    # Gemini sempre pela API nativa (também quando o .env aponta para o endpoint /openai)
    if settings.provider == "gemini" or "generativelanguage.googleapis.com" in (settings.base_url or ""):
        return GeminiLLM(settings)
    if settings.provider in ("openai", "openai-compatible", "groq", "ollama", "openrouter"):
        return OpenAICompatibleLLM(settings)
    raise ValueError(f"LLM_PROVIDER desconhecido: {settings.provider}")
