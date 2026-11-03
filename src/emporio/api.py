"""API HTTP mínima — o formato que um webhook de WhatsApp/chat chamaria.

    uvicorn emporio.api:app --reload
    curl -X POST localhost:8000/chat -H 'content-type: application/json' \\
         -d '{"session_id": "5567999990000", "message": "Vocês abrem sábado?"}'
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .agent import Agent

app = FastAPI(title="Empório da Música — Agente de Atendimento", version="0.1.0")
_agent: Agent | None = None


def get_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = Agent()
    return _agent


class ChatIn(BaseModel):
    session_id: str = Field(..., description="Identificador da conversa (ex.: telefone do cliente)")
    message: str = Field(..., min_length=1, max_length=2000)
    debug: bool = False


class ChatOut(BaseModel):
    reply: str
    tools_used: list[str]
    trace: list[dict] | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatOut)
def chat(body: ChatIn) -> ChatOut:
    r = get_agent().reply(body.session_id, body.message)
    return ChatOut(
        reply=r.text,
        tools_used=[s.tool for s in r.trace],
        trace=[s.__dict__ for s in r.trace] if body.debug else None,
    )


@app.delete("/chat/{session_id}")
def reset(session_id: str) -> dict:
    get_agent().reset(session_id)
    return {"status": "reset"}
