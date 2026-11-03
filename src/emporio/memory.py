"""Persistência do histórico de conversa (SQLite).

Por que persistir? Atendimento por mensagem é assíncrono: o cliente manda o
número do pedido agora e volta horas depois perguntando "e aí, chegou?". Uma
sessão por cliente (ex.: telefone do WhatsApp) mantém o contexto entre
reinícios do processo. SQLite = zero infraestrutura; em produção trocaria por
Postgres/Redis atrás da mesma interface.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class ConversationStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    message TEXT NOT NULL)"""
            )
            c.execute("CREATE INDEX IF NOT EXISTS ix_session ON messages(session_id, id)")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def load(self, session_id: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT message FROM messages WHERE session_id=? ORDER BY id", (session_id,)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def append(self, session_id: str, messages: list[dict]) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with self._conn() as c:
            c.executemany(
                "INSERT INTO messages(session_id, created_at, message) VALUES (?,?,?)",
                [(session_id, ts, json.dumps(m, ensure_ascii=False)) for m in messages],
            )

    def clear(self, session_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM messages WHERE session_id=?", (session_id,))


def window(messages: list[dict], max_user_turns: int) -> list[dict]:
    """Mantém só as últimas N interações, cortando SEMPRE no início de uma mensagem
    do usuário (nunca deixa um resultado de ferramenta órfão sem a chamada)."""
    user_idx = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if len(user_idx) <= max_user_turns:
        return messages
    return messages[user_idx[-max_user_turns]:]
