"""Ferramentas de pedidos: status (com verificação de identidade) e devolução/troca.

Privacidade (LGPD, seção 9 do manual): dados de pedido só são revelados se o
cliente informar o NÚMERO DO PEDIDO + E-MAIL ou TELEFONE do titular. Em caso de
falha, a mensagem é genérica (não confirma se o pedido existe), para não vazar
informação a quem tenta adivinhar números de pedido.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from ..data import ORDER_STATUS_LABELS, digits, payment_label
from .base import ToolContext, brl, tool

VERIFY_PARAMS = {
    "numero_pedido": {"type": "integer", "description": "Número do pedido informado pelo cliente"},
    "email_ou_telefone": {
        "type": "string",
        "description": "E-mail ou telefone cadastrado do titular do pedido, exatamente como o cliente informou",
    },
}

MAX_FAILED_ATTEMPTS = 3
_failed_attempts: dict[str, int] = {}


def _verify(ctx: ToolContext, numero_pedido, email_ou_telefone) -> tuple[pd.Series | None, dict | None]:
    if _failed_attempts.get(ctx.session_id, 0) >= MAX_FAILED_ATTEMPTS:
        return None, {
            "verificado": False,
            "bloqueado": True,
            "mensagem": "Muitas tentativas sem sucesso. Por segurança, oriente o cliente a falar com a "
            "equipe pelo telefone/WhatsApp ou e-mail da loja.",
        }
    fail = {
        "verificado": False,
        "mensagem": "Não foi possível localizar um pedido com esse número para esse e-mail/telefone. "
        "Peça ao cliente para conferir os dados (sem confirmar se o pedido existe).",
    }
    try:
        oid = int(str(numero_pedido).strip().lstrip("#"))
    except ValueError:
        return None, {"verificado": False, "mensagem": "Número de pedido inválido."}
    orders, customers = ctx.store.orders, ctx.store.customers
    if oid not in orders.index or not email_ou_telefone:
        _failed_attempts[ctx.session_id] = _failed_attempts.get(ctx.session_id, 0) + 1
        return None, fail
    order = orders.loc[oid]
    cust = customers.loc[order["customer_id"]]
    given = str(email_ou_telefone).strip().lower()
    ok = given == cust["email_norm"]
    d = digits(given)
    if not ok and len(d) >= 8:
        ok = cust["phone_digits"].endswith(d[-9:]) or d.endswith(cust["phone_digits"])
    if not ok:
        _failed_attempts[ctx.session_id] = _failed_attempts.get(ctx.session_id, 0) + 1
        return None, fail
    _failed_attempts.pop(ctx.session_id, None)
    return order, None


def _items(ctx: ToolContext, oid: int) -> list[dict]:
    it = ctx.store.order_items[ctx.store.order_items["order_id"] == oid]
    p = ctx.store.products
    return [
        {"produto": p.loc[r.product_id, "name"], "quantidade": int(r.quantity), "categoria": p.loc[r.product_id, "category"]}
        for r in it.itertuples()
    ]


def _fmt(d) -> str | None:
    return None if pd.isna(d) else pd.Timestamp(d).strftime("%d/%m/%Y")


@tool(
    "consultar_pedido",
    """Consulta status, itens, pagamento, rastreio e previsão de entrega de um pedido.
EXIGE o número do pedido E o e-mail ou telefone do titular. Se o cliente não informou
os dois, peça antes de chamar. Nunca revele dados de pedido sem verificação.""",
    {"type": "object", "properties": VERIFY_PARAMS, "required": ["numero_pedido", "email_ou_telefone"]},
)
def consultar_pedido(ctx: ToolContext, numero_pedido, email_ou_telefone: str) -> dict:
    order, err = _verify(ctx, numero_pedido, email_ou_telefone)
    if err:
        return err
    today = ctx.clock().date()
    cust = ctx.store.customers.loc[order["customer_id"]]
    out = {
        "verificado": True,
        "pedido": int(order["order_id"]),
        "cliente_primeiro_nome": cust["name"].split()[0],
        "data_pedido": _fmt(order["order_date"]),
        "status": ORDER_STATUS_LABELS.get(order["status"], order["status"]),
        "itens": _items(ctx, int(order["order_id"])),
        "total": brl(order["total_brl"]),
        "pagamento": payment_label(order["payment_method"]),
    }
    if order["tracking_code"]:
        out["codigo_rastreio"] = order["tracking_code"]
        out["onde_rastrear"] = "site dos Correios ou da Jadlog"
    if not pd.isna(order["estimated_delivery"]):
        out["previsao_entrega"] = _fmt(order["estimated_delivery"])
        if order["status"] == "shipped" and order["estimated_delivery"].date() < today:
            out["alerta"] = (
                "previsão de entrega já passou e o pedido ainda consta como enviado: reconheça o atraso, "
                "peça desculpas e ofereça abrir chamado para a equipe verificar com a transportadora"
            )
    if order["status"] in ("pending", "confirmed"):
        days = (today - order["order_date"].date()).days
        out["dias_desde_o_pedido"] = days
        if days > 10:
            out["alerta"] = (
                f"pedido parado há {days} dias sem envio: reconheça a demora e ofereça abrir chamado "
                "para a equipe verificar"
            )
    if order["notes"]:
        out["observacao"] = order["notes"]
    return out


def _days(a: date, b: date) -> int:
    return (b - a).days


@tool(
    "avaliar_devolucao_ou_troca",
    """Aplica as regras do manual (seções 4 e 8) a um pedido real e calcula prazos:
arrependimento (7 dias após recebimento), troca por preferência (7 dias), troca por
defeito (30 dias da compra) e garantia legal (90 dias do recebimento).
EXIGE número do pedido + e-mail/telefone do titular. Use quando o cliente quiser
devolver, trocar ou relatar defeito em um pedido.""",
    {
        "type": "object",
        "properties": {
            **VERIFY_PARAMS,
            "motivo": {
                "type": "string",
                "enum": ["arrependimento", "preferencia", "defeito"],
                "description": "arrependimento = desistiu da compra; preferencia = trocar cor/modelo; "
                "defeito = problema de fabricação",
            },
        },
        "required": ["numero_pedido", "email_ou_telefone", "motivo"],
    },
)
def avaliar_devolucao_ou_troca(ctx: ToolContext, numero_pedido, email_ou_telefone: str, motivo: str) -> dict:
    order, err = _verify(ctx, numero_pedido, email_ou_telefone)
    if err:
        return err
    today = ctx.clock().date()
    status = order["status"]
    base = {"verificado": True, "pedido": int(order["order_id"]), "motivo": motivo, "hoje": today.strftime("%d/%m/%Y")}

    if status == "cancelled":
        return {**base, "elegivel": False, "explicacao": "O pedido já está cancelado; não há o que devolver."}
    if status in ("pending", "confirmed"):
        return {
            **base,
            "elegivel": None,
            "explicacao": "O pedido ainda não foi enviado. O manual não trata cancelamento antes do envio: "
            "ofereça abrir um chamado para a equipe analisar o cancelamento.",
            "acao_sugerida": "registrar_chamado (tipo=cancelamento)",
        }
    if status == "shipped":
        return {
            **base,
            "elegivel": None,
            "explicacao": "O pedido ainda está em transporte. Os prazos de arrependimento e troca contam a "
            "partir do recebimento. Se chegar avariado, o cliente deve RECUSAR o recebimento e contatar a "
            "loja imediatamente (seção 5.2).",
        }

    # delivered — data de recebimento real não existe no dataset: usamos a previsão como proxy
    purchase = order["order_date"].date()
    received = order["estimated_delivery"].date() if not pd.isna(order["estimated_delivery"]) else purchase
    since_received, since_purchase = _days(received, today), _days(purchase, today)
    out = {
        **base,
        "data_compra": purchase.strftime("%d/%m/%Y"),
        "data_recebimento_considerada": received.strftime("%d/%m/%Y"),
        "suposicao": "data de recebimento estimada pela previsão de entrega do pedido; se o cliente "
        "recebeu em outra data, a equipe confirma",
        "dias_desde_recebimento": since_received,
        "dias_desde_compra": since_purchase,
        "condicoes_gerais": [
            "itens personalizados/com ajuste sob encomenda (setup, regulagem especial) não são elegíveis",
            "itens de liquidação com aviso de 'venda final' não são elegíveis",
        ],
    }

    if motivo == "arrependimento":
        limit = 7
        ok = since_received <= limit
        out.update(
            elegivel=ok,
            regra="4.1 Direito de arrependimento: até 7 dias corridos após o recebimento, sem justificativa.",
            prazo_final=(received.fromordinal(received.toordinal() + limit)).strftime("%d/%m/%Y"),
        )
        if ok:
            out["como_funciona"] = [
                "produto na embalagem original, sem sinais de uso, com acessórios e manuais",
                "frete de devolução por conta da loja",
                f"reembolso na mesma forma de pagamento ({payment_label(order['payment_method'])}) em até 10 dias úteis",
            ]
        else:
            out["alternativas"] = (
                "prazo de arrependimento encerrado; se houver DEFEITO, avaliar troca por defeito/garantia"
            )
    elif motivo == "preferencia":
        limit = 7
        ok = since_received <= limit
        out.update(
            elegivel=ok,
            regra="4.3 Troca por preferência (cor/modelo): até 7 dias, produto perfeito e na embalagem "
            "original, sujeito à disponibilidade; diferenças de valor cobradas ou reembolsadas.",
            prazo_final=(received.fromordinal(received.toordinal() + limit)).strftime("%d/%m/%Y"),
        )
    else:  # defeito
        if since_purchase <= 30:
            out.update(elegivel=True, regra="4.2 Troca por defeito de fabricação: até 30 dias corridos após a compra.")
        elif since_received <= 90:
            out.update(
                elegivel=True,
                via="garantia",
                regra="Passaram os 30 dias de troca direta na loja (4.2), mas o item está dentro da garantia "
                "legal de 90 dias (8.1). Acionar garantia com o fabricante; a loja pode intermediar.",
            )
        else:
            out.update(
                elegivel=False,
                via="garantia do fabricante",
                regra="Fora da troca na loja (30 dias) e da garantia legal (90 dias). Pode haver garantia do "
                "fabricante (6 meses a 2 anos, ver certificado do produto); a loja pode intermediar (4.2/8.2).",
            )
        out["nao_cobre"] = "mau uso, quedas, umidade excessiva, modificações não autorizadas, desgaste natural (8.3)"
    return out
