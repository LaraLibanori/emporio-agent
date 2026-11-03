"""Loop do agente: function calling "na mão" (sem framework).

    usuário ─▶ LLM ─┬─▶ resposta final ─▶ usuário
                    └─▶ tool_calls ─▶ executa ferramentas ─▶ LLM (repete, até N rodadas)

Por que sem LangChain/LangGraph? O fluxo é um único agente com ~8 ferramentas.
Um loop explícito de ~60 linhas é mais fácil de ler, depurar e testar, e não
esconde o que vai no prompt. Se o projeto crescer (vários agentes, aprovação
humana no meio do fluxo), migrar para LangGraph é natural.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from .config import Settings
from .llm import LLM, build_llm
from .memory import ConversationStore, window
from .prompts import SYSTEM_PROMPT, turn_context
from .tools import REGISTRY, ToolContext

log = logging.getLogger(__name__)

FALLBACK_REPLY = (
    "Opa, tive um probleminha técnico pra consultar o sistema agora 😕 "
    "Você pode tentar de novo em instantes ou falar com a equipe pelo (67) 3341-4444."
)


@dataclass
class TraceStep:
    tool: str
    args: dict
    result: dict
    ms: int


@dataclass
class AgentReply:
    text: str
    trace: list[TraceStep] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


class Agent:
    def __init__(
        self,
        llm: LLM | None = None,
        settings: Settings | None = None,
        store: ConversationStore | None = None,
        tool_context_factory=None,
    ):
        self.settings = settings or Settings()
        self.llm = llm or build_llm(self.settings)
        self.history = store or ConversationStore(self.settings.state_dir / "conversations.db")
        self.tools = list(REGISTRY.values())
        self._ctx_factory = tool_context_factory or (lambda sid: ToolContext(session_id=sid, settings=self.settings))

    def reply(self, session_id: str, user_message: str) -> AgentReply:
        ctx = self._ctx_factory(session_id)
        past = window(self.history.load(session_id), self.settings.history_turns)
        new_msgs: list[dict] = [{"role": "user", "content": user_message}]
        system = f"{SYSTEM_PROMPT}\n\n{turn_context(ctx.clock())}"
        trace: list[TraceStep] = []
        usage = {"input": 0, "output": 0}

        text = None
        specs = [t.spec() for t in self.tools]
        for round_ in range(self.settings.max_tool_rounds + 1):
            # última rodada: força o modelo a responder em texto com o que já tem
            last_round = round_ == self.settings.max_tool_rounds
            try:
                resp = self.llm.chat(system, past + new_msgs, specs, force_text=last_round)
            except Exception as e:  # provedor fora, rate limit, etc.
                log.exception("falha no LLM: %s", e)
                text = FALLBACK_REPLY
                break
            for k, v in resp.usage.items():
                usage[k] = usage.get(k, 0) + v
            if not resp.tool_calls or last_round:
                text = (resp.content or "").strip()
                break  # a mensagem final é gravada abaixo (nunca gravamos assistant vazio)
            new_msgs.append(resp.as_message())
            for tc in resp.tool_calls:
                t0 = time.perf_counter()
                tool = REGISTRY.get(tc.name)
                result = tool.run(ctx, tc.arguments) if tool else {"erro": f"ferramenta '{tc.name}' não existe"}
                trace.append(TraceStep(tc.name, tc.arguments, result, int((time.perf_counter() - t0) * 1000)))
                new_msgs.append(
                    {"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                     "content": json.dumps(result, ensure_ascii=False)}
                )

        if not text:
            # estourou rodadas ou modelo respondeu vazio: não deixar o cliente no vácuo
            text = (
                "Deixa eu confirmar isso com a equipe e já te retorno, tudo bem? "
                "Se preferir, fale com a gente no (67) 3341-4444."
            )
        new_msgs.append({"role": "assistant", "content": text})

        self.history.append(session_id, new_msgs)
        return AgentReply(text, trace, usage)

    def reset(self, session_id: str) -> None:
        self.history.clear(session_id)
