"""Configuração central (lida de variáveis de ambiente / .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

try:  # .env é opcional
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("AGENT_DATA_DIR", PROJECT_ROOT / "data"))
STORE_TZ = ZoneInfo("America/Campo_Grande")


@dataclass(frozen=True)
class Settings:
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai").lower())
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gemini-2.5-flash"))
    api_key: str | None = field(default_factory=lambda: os.getenv("LLM_API_KEY") or None)
    base_url: str | None = field(default_factory=lambda: os.getenv("LLM_BASE_URL") or None)
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.2")))
    max_tool_rounds: int = field(default_factory=lambda: int(os.getenv("AGENT_MAX_TOOL_ROUNDS", "6")))
    history_turns: int = field(default_factory=lambda: int(os.getenv("AGENT_HISTORY_TURNS", "12")))
    state_dir: Path = field(
        default_factory=lambda: Path(os.getenv("AGENT_STATE_DIR", PROJECT_ROOT / "var"))
    )


def now() -> datetime:
    """Data/hora "atual" da loja.

    Pode ser fixada com AGENT_NOW (útil para demos e testes, já que os pedidos
    do dataset são de 2025/2026 e as regras de prazo dependem da data).
    """
    fixed = os.getenv("AGENT_NOW", "").strip()
    if fixed:
        dt = datetime.fromisoformat(fixed)
        if dt.hour == 0 and dt.minute == 0 and "T" not in fixed:
            dt = dt.replace(hour=10)  # só a data -> assume meio da manhã
        return dt.replace(tzinfo=STORE_TZ) if dt.tzinfo is None else dt.astimezone(STORE_TZ)
    return datetime.now(STORE_TZ)
