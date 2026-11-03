"""Infra mínima de ferramentas: registro + esquema JSON + execução segura."""

from __future__ import annotations

import inspect
import json
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from ..config import Settings, now
from ..data import Store, get_store

log = logging.getLogger(__name__)


@dataclass
class ToolContext:
    """Estado disponível para as ferramentas (injetado, não vem do LLM)."""

    session_id: str = "default"
    store: Store = field(default_factory=get_store)
    settings: Settings = field(default_factory=Settings)
    clock: Callable[[], datetime] = now

    @property
    def state_dir(self) -> Path:
        self.settings.state_dir.mkdir(parents=True, exist_ok=True)
        return self.settings.state_dir


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., dict]

    def spec(self) -> dict:
        """Formato neutro (nome, descrição, JSON Schema) — cada provedor adapta."""
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def run(self, ctx: ToolContext, args: dict | str | None) -> dict:
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except json.JSONDecodeError:
                return {"erro": "argumentos inválidos (JSON malformado)"}
        args = dict(args or {})
        accepted = set(inspect.signature(self.fn).parameters) - {"ctx"}
        unknown = set(args) - accepted
        clean = {k: v for k, v in args.items() if k in accepted}
        try:
            result = self.fn(ctx, **clean)
        except TypeError as e:  # parâmetro obrigatório ausente, tipo errado...
            return {"erro": f"chamada inválida: {e}"}
        except Exception as e:  # nunca derrubar a conversa por causa de uma tool
            log.error("tool %s falhou: %s\n%s", self.name, e, traceback.format_exc())
            return {"erro": "falha interna ao consultar o sistema; peça desculpas e ofereça atendimento humano"}
        if unknown:
            result = {**result, "_aviso": f"parâmetros ignorados: {sorted(unknown)}"}
        return result


REGISTRY: dict[str, Tool] = {}


def tool(name: str, description: str, parameters: dict[str, Any]):
    def deco(fn):
        REGISTRY[name] = Tool(name, description.strip(), parameters, fn)
        return fn

    return deco


def brl(value: float) -> str:
    s = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"
