"""Pagamento e frete: regras do manual implementadas em código.

LLMs erram aritmética e esquecem exceções ("PIX não acumula com promoção",
"parcela mínima por faixa"). Por isso essas regras são calculadas aqui e o
modelo apenas comunica o resultado.
"""

from __future__ import annotations

from ..data import normalize
from .base import ToolContext, brl, tool
from .catalog import detalhes_produto

PIX_DISCOUNT = 0.05
INSTALLMENT_RULES = [  # (de, até, parcela mínima) — seção 3.1
    (1, 3, 50.0),
    (4, 6, 80.0),
    (7, 12, 100.0),
]
COMBINED_PAYMENT_MIN = 2000.0

METRO_CAMPO_GRANDE = {"campo grande"}  # manual não lista municípios da região metropolitana
FREE_SHIPPING_MIN = 500.0
METRO_FEE = 35.0
OVERSIZED_CATEGORIES = {"Baterias e Percussão", "Teclados e Pianos", "Cordas Orquestrais"}


def installment_table(total: float) -> list[dict]:
    rows = []
    for lo, hi, min_parcel in INSTALLMENT_RULES:
        for n in range(lo, hi + 1):
            parcel = total / n
            if parcel + 1e-9 >= min_parcel:
                rows.append({"parcelas": n, "valor_parcela": brl(round(parcel, 2))})
    return rows


@tool(
    "calcular_pagamento",
    """Calcula as condições de pagamento (PIX, débito, boleto, crédito em até 12x) para um
produto do catálogo OU para um valor. Aplica: desconto PIX de 5% (não cumulativo com
promoção), parcela mínima por faixa e regra de pagamento combinado. Use sempre que o
cliente perguntar sobre parcelamento, desconto à vista ou "quanto fica no PIX".""",
    {
        "type": "object",
        "properties": {
            "produto": {"type": "string", "description": "Nome ou id do produto (preferível: preço vem do sistema)"},
            "quantidade": {"type": "integer", "description": "Quantidade (padrão 1)"},
            "valor": {"type": "number", "description": "Valor em reais, se não for um produto específico"},
            "valor_ja_promocional": {
                "type": "boolean",
                "description": "Só com 'valor': true se o valor já tem desconto promocional",
            },
        },
    },
)
def calcular_pagamento(
    ctx: ToolContext,
    produto: str | None = None,
    quantidade: int = 1,
    valor: float | None = None,
    valor_ja_promocional: bool = False,
) -> dict:
    qty = max(1, int(quantidade or 1))
    info: dict = {}
    if produto:
        found = detalhes_produto(ctx, produto)
        if not found.get("encontrado"):
            return found
        pid = found["produto"]["id"]
        row = ctx.store.products.loc[pid]
        info["produto"] = row["name"]
        if row["availability"] != "disponivel":
            info["atencao"] = f"produto {found['produto']['disponibilidade']}; simulação apenas informativa"
        promo = bool(row["promo_discount"])
        total = float(row["final_price"]) * qty
        if promo:
            info["promocao"] = f"{row['promo_name']} -{row['promo_discount']}% (de {brl(row['price_brl'])} por {brl(row['final_price'])})"
    elif valor is not None:
        total, promo = float(valor), bool(valor_ja_promocional)
    else:
        return {"erro": "informe 'produto' ou 'valor'"}

    if total <= 0:
        return {"erro": "valor deve ser positivo"}

    pix = total if promo else round(total * (1 - PIX_DISCOUNT), 2)
    table = installment_table(total)
    out = {
        **info,
        "quantidade": qty,
        "valor_total": brl(total),
        "pix": {
            "valor": brl(pix),
            "obs": "sem desconto adicional: promoções não acumulam com o desconto PIX"
            if promo
            else "5% de desconto à vista no PIX",
        },
        "debito": brl(total),
        "boleto": f"{brl(total)} (compensação em até 3 dias úteis)",
        "credito": {
            "max_parcelas_sem_juros": table[-1]["parcelas"] if table else 1,
            "opcoes": table if table else [{"parcelas": 1, "valor_parcela": brl(total)}],
            "regra": "sem juros; parcela mínima R$50 (1–3x), R$80 (4–6x), R$100 (7–12x)",
        },
        "pagamento_combinado": "permitido (ex.: PIX + cartão)" if total > COMBINED_PAYMENT_MIN
        else f"não disponível (só para compras acima de {brl(COMBINED_PAYMENT_MIN)})",
    }
    if total < 50:
        out["credito"]["opcoes"] = [{"parcelas": 1, "valor_parcela": brl(total)}]
        out["credito"]["max_parcelas_sem_juros"] = 1
    return out


@tool(
    "estimar_frete",
    """Informa regras de frete e prazo para a cidade do cliente (seção 5 do manual).
Campo Grande: grátis acima de R$500, senão R$35, 1–3 dias úteis por motoboy.
Outras cidades: Correios/Jadlog com valor calculado pelo CEP (a equipe cota).""",
    {
        "type": "object",
        "properties": {
            "cidade": {"type": "string", "description": "Cidade de entrega"},
            "valor_pedido": {"type": "number", "description": "Valor dos produtos em reais (opcional)"},
            "produto": {"type": "string", "description": "Produto (opcional, para checar se é de grande porte)"},
        },
        "required": ["cidade"],
    },
)
def estimar_frete(ctx: ToolContext, cidade: str, valor_pedido: float | None = None, produto: str | None = None) -> dict:
    out: dict = {"cidade": cidade}
    if produto:
        found = detalhes_produto(ctx, produto)
        if found.get("encontrado"):
            row = ctx.store.products.loc[found["produto"]["id"]]
            valor_pedido = valor_pedido or float(row["final_price"])
            if row["category"] in OVERSIZED_CATEGORIES:
                out["grande_porte"] = "instrumento de grande porte: pode exigir frete especial com cotação individual"
    if normalize(cidade).split("-")[0].strip() in METRO_CAMPO_GRANDE:
        out["modalidade"] = "motoboy próprio (cliente é contatado por telefone antes da entrega)"
        out["prazo"] = "1 a 3 dias úteis"
        if valor_pedido is None:
            out["valor"] = f"grátis para pedidos acima de {brl(FREE_SHIPPING_MIN)}; abaixo disso {brl(METRO_FEE)}"
        else:
            out["valor"] = "grátis" if valor_pedido > FREE_SHIPPING_MIN else brl(METRO_FEE)
    else:
        out["modalidades"] = [
            {"nome": "PAC (Correios)", "prazo": "5 a 12 dias úteis"},
            {"nome": "SEDEX (Correios)", "prazo": "2 a 5 dias úteis"},
            {"nome": "Jadlog (.package)", "prazo": "3 a 8 dias úteis"},
        ]
        out["valor"] = "calculado pelo CEP, peso e dimensões; a equipe faz a cotação (não informe valor estimado)"
        out["seguro"] = "todos os envios têm seguro contra extravio e avaria, com rastreamento"
        out["observacao"] = (
            "o manual não define quais municípios compõem a região metropolitana; se a cidade for "
            "vizinha a Campo Grande, confirme com a equipe"
        )
    return out
