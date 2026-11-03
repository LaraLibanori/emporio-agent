"""Ferramentas institucionais: políticas (RAG), horário/aberto agora, chamados."""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timedelta

from ..policies import get_policy_index
from .base import ToolContext, tool

WEEK_HOURS = {  # weekday() -> (abre, fecha)
    0: ("09:00", "18:00"), 1: ("09:00", "18:00"), 2: ("09:00", "18:00"),
    3: ("09:00", "18:00"), 4: ("09:00", "18:00"), 5: ("09:00", "13:00"),
}
DIAS = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]


def _easter(y: int) -> date:  # algoritmo anônimo gregoriano
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(y, month, day)


def holidays(y: int) -> dict[date, str]:
    """Feriados nacionais + MS + Campo Grande (o manual diz 'Feriados: fechado')."""
    fixed = {
        (1, 1): "Confraternização Universal", (4, 21): "Tiradentes", (5, 1): "Dia do Trabalho",
        (6, 13): "Santo Antônio (padroeiro de Campo Grande)", (8, 26): "Aniversário de Campo Grande",
        (9, 7): "Independência", (10, 11): "Criação do Estado de MS", (10, 12): "Nossa Senhora Aparecida",
        (11, 2): "Finados", (11, 15): "Proclamação da República", (11, 20): "Consciência Negra",
        (12, 25): "Natal",
    }
    out = {date(y, m, d): n for (m, d), n in fixed.items()}
    e = _easter(y)
    out[e - timedelta(days=47)] = "Carnaval"
    out[e - timedelta(days=2)] = "Sexta-feira Santa"
    out[e + timedelta(days=60)] = "Corpus Christi"
    return out


def _next_opening(dt: datetime) -> datetime:
    d = dt
    for _ in range(14):
        hrs = WEEK_HOURS.get(d.weekday())
        if hrs and d.date() not in holidays(d.year):
            opens = d.replace(hour=int(hrs[0][:2]), minute=0, second=0, microsecond=0)
            if opens > dt:
                return opens
        d = (d + timedelta(days=1)).replace(hour=0, minute=0)
    return d


@tool(
    "status_loja_agora",
    """Diz se a loja está aberta AGORA (considera dia da semana, horário e feriados) e
quando reabre. Use para "vocês estão abertos?", "posso passar aí hoje?" e para avisar
o horário de retorno da equipe humana quando a loja estiver fechada.""",
    {"type": "object", "properties": {}},
)
def status_loja_agora(ctx: ToolContext) -> dict:
    dt = ctx.clock()
    hol = holidays(dt.year).get(dt.date())
    hrs = WEEK_HOURS.get(dt.weekday())
    is_open = False
    if hrs and not hol:
        o = dt.replace(hour=int(hrs[0][:2]), minute=int(hrs[0][3:]), second=0)
        c = dt.replace(hour=int(hrs[1][:2]), minute=int(hrs[1][3:]), second=0)
        is_open = o <= dt < c
    out = {
        "agora": f"{DIAS[dt.weekday()]}, {dt.strftime('%d/%m/%Y %H:%M')} (horário de Campo Grande)",
        "aberta_agora": is_open,
        "horario_padrao": "segunda a sexta 09:00–18:00; sábado 09:00–13:00; domingos e feriados fechado",
    }
    if hol:
        out["feriado_hoje"] = hol
    if is_open:
        out["fecha_hoje_as"] = hrs[1]
    else:
        nxt = _next_opening(dt)
        out["reabre"] = f"{DIAS[nxt.weekday()]}, {nxt.strftime('%d/%m')} às {nxt.strftime('%H:%M')}"
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
