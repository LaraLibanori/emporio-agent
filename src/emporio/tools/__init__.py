"""Ferramentas do agente. Importar este pacote registra todas elas em REGISTRY."""

from . import catalog, commerce, orders, store  # noqa: F401  (registro por efeito colateral)
from .base import REGISTRY, Tool, ToolContext

__all__ = ["REGISTRY", "Tool", "ToolContext"]
