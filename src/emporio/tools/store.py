"""Ferramentas institucionais: políticas (RAG), horário/aberto agora, chamados."""

from __future__ import annotations

import json
import uuid

from ..hours import DIAS, HORARIO_PADRAO, holidays, store_status  # noqa: F401  (holidays: reexportado)
from ..policies import get_policy_index
from .base import ToolContext, tool


@tool(
    "status_loja_agora",
    """Diz se a loja está aberta AGORA (considera dia da semana, horário e feriados) e
quando reabre. Use para "vocês estão abertos?", "posso passar aí hoje?" e para avisar
o horário de retorno da equipe humana quando a loja estiver fechada.""",
    {"type": "object", "properties": {}},
)
def status_loja_agora(ctx: ToolContext) -> dict:
    dt = ctx.clock()
    st = store_status(dt)
    out = {
        "agora": f"{DIAS[dt.weekday()]}, {dt.strftime('%d/%m/%Y %H:%M')} (horário de Campo Grande)",
        "aberta_agora": st.is_open,
        "horario_padrao": HORARIO_PADRAO,
    }
    if st.holiday:
        out["feriado_hoje"] = st.holiday
    if st.is_open:
        out["fecha_hoje_as"] = st.closes_at
    else:
        out["reabre"] = st.reopens_label
    out["obs"] = "em datas especiais (Black Friday, Aniversário, Natal) o horário pode ser estendido"
    return out


@tool(
    "consultar_politicas",
    """Busca trechos do Manual de Políticas da loja. Use para: endereço/contato, horários,
formas de pagamento, trocas/devoluções, frete/entregas, rastreamento, promoções
(regras), garantia, privacidade/LGPD, escopo (o que a loja vende).
Responda com base APENAS nos trechos retornados; se não cobrirem a dúvida, diga que vai
confirmar com a equipe.""",
    {
        "type": "object",
        "properties": {"pergunta": {"type": "string", "description": "Dúvida do cliente em linguagem natural"}},
        "required": ["pergunta"],
    },
)
def consultar_politicas(ctx: ToolContext, pergunta: str) -> dict:
    hits = get_policy_index().search(pergunta, k=3)
    if not hits:
        return {"trechos": [], "aviso": "nenhum trecho relevante no manual"}
    return {"trechos": [c.render() for c, _ in hits]}


@tool(
    "registrar_chamado",
    """Registra um chamado para a equipe humana (reclamação, pedido de atendimento humano,
cancelamento, problema com entrega, devolução aprovada para dar andamento, dúvida não
coberta). Retorna protocolo. Prazo de retorno: 24 horas úteis. Antes de chamar, tenha
um resumo claro e um contato (nome + telefone ou e-mail) do cliente.""",
    {
        "type": "object",
        "properties": {
            "tipo": {
                "type": "string",
                "enum": ["reclamacao", "atendimento_humano", "cancelamento", "devolucao_troca", "entrega", "outro"],
            },
            "resumo": {"type": "string", "description": "Resumo objetivo do caso"},
            "contato_cliente": {"type": "string", "description": "Nome e telefone/e-mail do cliente"},
            "numero_pedido": {"type": "integer", "description": "Se houver"},
        },
        "required": ["tipo", "resumo", "contato_cliente"],
    },
)
def registrar_chamado(
    ctx: ToolContext, tipo: str, resumo: str, contato_cliente: str, numero_pedido: int | None = None
) -> dict:
    protocol = f"EM-{ctx.clock().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    rec = {
        "protocolo": protocol,
        "criado_em": ctx.clock().isoformat(),
        "sessao": ctx.session_id,
        "tipo": tipo,
        "resumo": resumo,
        "contato": contato_cliente,
        "pedido": numero_pedido,
    }
    with open(ctx.state_dir / "chamados.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    status = status_loja_agora(ctx)
    return {
        "protocolo": protocol,
        "prazo_retorno": "até 24 horas úteis",
        "equipe_disponivel_agora": status["aberta_agora"],
        **({"equipe_retorna": status["reabre"]} if not status["aberta_agora"] else {}),
    }
