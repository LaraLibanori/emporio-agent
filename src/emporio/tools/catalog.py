"""Ferramentas de catálogo: busca, detalhes, alternativas."""

from __future__ import annotations

import re

import pandas as pd
from rapidfuzz import fuzz, process

from ..data import AVAILABILITY_LABELS, STRINGS_LABELS, normalize
from .base import ToolContext, brl, tool

SEARCH_STOP = set(
    normalize(
        "quero queria procuro procurando tem tem? voces vcs algum alguma um uma de do da "
        "para pra com ate até reais r$ preco barato opcoes opções disponivel disponiveis "
        "modelo modelos instrumento instrumentos o a os as que me mostra mostre quais"
    ).split()
)

# Termo do cliente -> nome de categoria
CATEGORY_ALIASES = {
    "guitarra": "Guitarras", "guitarras": "Guitarras",
    "baixo": "Baixos", "baixos": "Baixos", "contrabaixo": "Baixos",
    "bateria": "Baterias e Percussão", "baterias": "Baterias e Percussão", "percussao": "Baterias e Percussão",
    "teclado": "Teclados e Pianos", "teclados": "Teclados e Pianos", "piano": "Teclados e Pianos",
    "sintetizador": "Teclados e Pianos", "synth": "Teclados e Pianos",
    "violao": "Violões", "violoes": "Violões",
    "ukulele": "Ukuleles", "ukuleles": "Ukuleles", "uke": "Ukuleles",
    "flauta": "Instrumentos de Sopro (Madeiras)", "saxofone": "Instrumentos de Sopro (Madeiras)",
    "sax": "Instrumentos de Sopro (Madeiras)", "clarinete": "Instrumentos de Sopro (Madeiras)",
    "trompete": "Instrumentos de Sopro (Metais)", "trombone": "Instrumentos de Sopro (Metais)",
    "tuba": "Instrumentos de Sopro (Metais)",
    "violino": "Cordas Orquestrais", "viola": "Cordas Orquestrais", "violoncelo": "Cordas Orquestrais",
    "cello": "Cordas Orquestrais",
}

ACCESSORY_TERMS = {
    "corda", "cordas", "palheta", "palhetas", "cabo", "cabos", "pedal", "pedais",
    "amplificador", "amplificadores", "amp", "case", "cases", "capa", "capas",
    "afinador", "afinadores", "correia", "estante", "suporte", "baqueta", "baquetas",
    "encordoamento", "boquilha", "fone", "microfone",
}


def _looks_like_accessory(terms: list[str]) -> bool:
    """'cordas para violão', 'pedal de guitarra' -> acessório.
    '7 cordas', 'violão cordas de aço' -> característica de instrumento."""
    if any(t.isdigit() for t in terms):
        return False
    acc_pos = [i for i, t in enumerate(terms) if t in ACCESSORY_TERMS]
    if not acc_pos:
        return False
    inst_pos = [i for i, t in enumerate(terms) if t in CATEGORY_ALIASES]
    # acessório citado antes do instrumento (ou sem instrumento) = cliente quer o acessório
    return not inst_pos or acc_pos[0] < inst_pos[0]


def _resolve_category(ctx: ToolContext, categoria: str | None) -> str | None:
    if not categoria:
        return None
    key = normalize(categoria)
    if key in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[key]
    names = ctx.store.categories["name"].tolist()
    match = process.extractOne(key, {n: normalize(n) for n in names}, scorer=fuzz.WRatio)
    return match[2] if match and match[1] >= 70 else None


def product_card(row: pd.Series, detailed: bool = False) -> dict:
    """Representação de produto enviada ao LLM: só o necessário, já formatado."""
    card = {
        "id": int(row["product_id"]),
        "nome": row["name"],
        "categoria": row["category"],
        "disponibilidade": AVAILABILITY_LABELS[row["availability"]],
        "pode_comprar": row["availability"] == "disponivel",
        "preco": brl(row["price_brl"]),
    }
    if row["promo_discount"]:
        card["promocao_ativa"] = {
            "campanha": row["promo_name"],
            "desconto": f"{row['promo_discount']}%",
            "preco_original": brl(row["price_brl"]),
            "preco_promocional": brl(row["final_price"]),
        }
    if row["availability"] == "disponivel" and row["stock_quantity"] <= 2:
        card["observacao_estoque"] = "últimas unidades"
    specs = dict(row["specs"] or {})
    if "strings" in specs and specs["strings"] in STRINGS_LABELS:
        specs["strings"] = STRINGS_LABELS[specs["strings"]]
    if "electronics" in specs:
        specs["electronics"] = "sim (eletroacústico)" if specs["electronics"] == "yes" else "não"
    card["specs"] = specs
    if detailed:
        if row["description_reliable"]:
            card["descricao"] = row["description"]
        else:
            card["descricao"] = None
            card["nota_interna"] = (
                "descrição do cadastro inconsistente com o produto; use apenas nome e specs"
            )
    return card


def _name_key(text: str) -> str:
    """'Giannini GF-3D' -> 'giannini gf3d' (hífens/pontuação não atrapalham o match)."""
    return re.sub(r"[^a-z0-9 ]", "", normalize(text))


def _model_key(text: str) -> str:
    """Como _name_key, mas junta código de modelo digitado com espaço:
    'takamine gd 20' -> 'takamine gd20', 'yamaha c 40' -> 'yamaha c40'."""
    return re.sub(r"\b([a-z]{1,4}) (\d{1,4}[a-z]?)\b", r"\1\2", _name_key(text))


def rank_names(ctx: ToolContext, query: str) -> list[tuple[int, float, float]]:
    """[(product_id, token_set_ratio, WRatio)] em ordem decrescente.
    token_set_ratio = 100 quando a consulta está contida no nome ("takamine gd20");
    WRatio desempata e tolera erros de digitação."""
    qk = _model_key(query)
    rows = []
    for pid, name in ctx.store.products["name"].items():
        nk = _model_key(name)
        score = (fuzz.token_set_ratio(qk, nk) + fuzz.WRatio(qk, nk)) / 2
        rows.append((int(pid), round(score, 1), _coverage(qk, nk)))
    return sorted(rows, key=lambda r: (r[2], r[1]), reverse=True)


NAME_NOISE = {"violao", "guitarra", "baixo", "ukulele", "teclado", "bateria", "o", "a", "de", "do", "da", "modelo"}


def _coverage(query_key: str, name_key: str) -> float:
    """Fração dos termos da consulta presentes no nome (tolerando erro de digitação e prefixo).
    Evita que 'Gibson SG' seja resolvido como 'Gibson Les Paul' só por causa da marca."""
    q = [t for t in query_key.split() if t not in NAME_NOISE]
    if not q:
        return 0.0
    words = name_key.split()
    hit = sum(1 for t in q if any(w.startswith(t) or fuzz.ratio(t, w) >= 80 for w in words))
    return round(hit / len(q), 2)


def _alternatives(ctx: ToolContext, row: pd.Series, k: int = 3) -> list[dict]:
    p = ctx.store.products
    cand = p[(p["category_id"] == row["category_id"]) & (p["availability"] == "disponivel")]
    cand = cand[cand["product_id"] != row["product_id"]]
    if row["specs"].get("strings") in ("nylon", "steel"):
        same = cand[cand["specs"].map(lambda s: s.get("strings")) == row["specs"]["strings"]]
        cand = same if len(same) else cand
    # prioriza o mesmo "tipo" (7 cordas, barítono, eletroacústico...) e depois preço parecido
    brand_tokens = set(_name_key(row["name"]).split()[:1])
    ref = set(_name_key(row["name"]).split()) - brand_tokens - {"natural", "preto", "sunburst"}
    overlap = cand["name"].map(lambda n: len(ref & set(_name_key(n).split())))
    cand = cand.assign(_o=overlap, _d=(cand["final_price"] - row["final_price"]).abs())
    cand = cand.sort_values(["_o", "_d"], ascending=[False, True])
    return [product_card(r) for _, r in cand.head(k).iterrows()]


@tool(
    "buscar_produtos",
    """Busca instrumentos no catálogo com filtros. Use para perguntas abertas como
"quais violões até R$1000?", "tem guitarra Fender?", "ukulele para iniciante".
Retorna preço atual (e promocional, se houver) e disponibilidade REAL do estoque.
Por padrão só retorna itens que podem ser comprados agora.""",
    {
        "type": "object",
        "properties": {
            "termo": {"type": "string", "description": "Palavras-chave: marca, modelo, tipo, madeira, cor... (opcional)"},
            "categoria": {
                "type": "string",
                "description": "Guitarras, Baixos, Baterias e Percussão, Teclados e Pianos, Violões, Ukuleles, "
                "Instrumentos de Sopro (Madeiras), Instrumentos de Sopro (Metais), Cordas Orquestrais",
            },
            "preco_min": {"type": "number", "description": "Preço mínimo em reais (considera preço promocional)"},
            "preco_max": {"type": "number", "description": "Preço máximo em reais (considera preço promocional)"},
            "cordas": {"type": "string", "enum": ["nylon", "aço"], "description": "Tipo de corda (violões/ukuleles)"},
            "eletroacustico": {"type": "boolean", "description": "true = só com captação; false = só acústicos"},
            "incluir_indisponiveis": {"type": "boolean", "description": "Incluir esgotados/descontinuados/em breve"},
            "ordenar": {"type": "string", "enum": ["menor_preco", "maior_preco", "relevancia"]},
            "limite": {"type": "integer", "description": "Máx. de itens (padrão 8, máx 15)"},
        },
    },
)
def buscar_produtos(
    ctx: ToolContext,
    termo: str | None = None,
    categoria: str | None = None,
    preco_min: float | None = None,
    preco_max: float | None = None,
    cordas: str | None = None,
    eletroacustico: bool | None = None,
    incluir_indisponiveis: bool = False,
    ordenar: str = "menor_preco",
    limite: int = 8,
) -> dict:
    from .commerce import to_number  # import tardio: commerce importa catalog

    try:
        preco_min, preco_max = to_number(preco_min, "preco_min"), to_number(preco_max, "preco_max")
        limite = int(to_number(limite, "limite") or 8)
    except ValueError as e:
        return {"erro": str(e)}
    p = ctx.store.products
    df = p.copy()
    notes: list[str] = []

    terms = [t for t in normalize(termo or "").split() if t not in SEARCH_STOP]
    if _looks_like_accessory(terms):
        acc = sorted(set(terms) & ACCESSORY_TERMS)
        return {
            "resultados": [],
            "aviso_escopo": f"'{', '.join(acc)}' é acessório. A loja vende exclusivamente instrumentos "
            "musicais (não vende cordas, palhetas, cabos, cases, pedais, amplificadores etc.).",
        }
    # termo que é na verdade uma categoria ("violão nylon") -> vira filtro
    for t in list(terms):
        if t in CATEGORY_ALIASES:
            categoria = categoria or CATEGORY_ALIASES[t]
            terms.remove(t)
            break
    # "corda(s)" sem ser acessório ("7 cordas", "cordas de nylon") não é palavra discriminante
    terms = [t for t in terms if t not in {"corda", "cordas"} or any(c.isdigit() for c in termo or "")]

    cat = _resolve_category(ctx, categoria)
    if categoria and not cat:
        notes.append(f"categoria '{categoria}' não reconhecida; busca feita em todo o catálogo")
    if cat:
        df = df[df["category"] == cat]
        if df.empty:
            return {
                "resultados": [],
                "categoria": cat,
                "aviso": f"No momento não temos nenhum produto da categoria '{cat}' no catálogo.",
            }

    if cordas:
        want = "steel" if normalize(cordas).startswith("aco") else "nylon"
        df = df[df["specs"].map(lambda s: s.get("strings")) == want]
    if eletroacustico is not None:
        df = df[df["specs"].map(lambda s: s.get("electronics")) == ("yes" if eletroacustico else "no")]
    if preco_min is not None:
        df = df[df["final_price"] >= float(preco_min)]
    if preco_max is not None:
        df = df[df["final_price"] <= float(preco_max)]

    if terms:
        # todos os termos precisam aparecer (com tolerância a erro de digitação)
        def match(text: str) -> float:
            words = text.split()
            scores = []
            for t in terms:
                if t in text:
                    scores.append(100)
                else:
                    best = process.extractOne(t, words, scorer=fuzz.ratio)
                    scores.append(best[1] if best else 0)
            return min(scores)

        df = df.assign(_score=df["search_text"].map(match))
        df = df[df["_score"] >= 80]

    unavailable = df[df["availability"] != "disponivel"]
    if not incluir_indisponiveis:
        df = df[df["availability"] == "disponivel"]
        if len(unavailable):
            notes.append(
                f"{len(unavailable)} item(ns) que atendem ao filtro estão indisponíveis "
                "(esgotado/descontinuado/em breve) e foram omitidos"
            )

    if ordenar == "maior_preco":
        df = df.sort_values("final_price", ascending=False)
    elif ordenar == "relevancia" and "_score" in df:
        df = df.sort_values(["_score", "final_price"], ascending=[False, True])
    else:
        df = df.sort_values("final_price")

    limite = max(1, min(limite, 15))
    out = {
        "total_encontrado": int(len(df)),
        "mostrando": int(min(len(df), limite)),
        "resultados": [product_card(r) for _, r in df.head(limite).iterrows()],
    }
    if cat:
        out["categoria"] = cat
    if notes:
        out["observacoes"] = notes
    if not len(df):
        out["sugestao"] = "nada encontrado; tente afrouxar filtros (preço, termo) ou outra categoria"
    return out


@tool(
    "detalhes_produto",
    """Consulta UM produto específico pelo nome (aceita nome aproximado/erros de digitação)
ou pelo id. Use para "quanto custa o X?", "tem o X em estoque?", "quais as specs do X?".
Se o produto estiver esgotado/descontinuado/em breve, já retorna alternativas disponíveis.""",
    {
        "type": "object",
        "properties": {
            "produto": {"type": "string", "description": "Nome (ex.: 'Takamine GD20') ou id numérico"},
        },
        "required": ["produto"],
    },
)
def detalhes_produto(ctx: ToolContext, produto: str) -> dict:
    p = ctx.store.products
    q = str(produto).strip()
    row = None
    if q.isdigit() and int(q) in p.index:
        row = p.loc[int(q)]
    else:
        ranked = rank_names(ctx, q)
        top = ranked[0]
        if top[2] < 1.0 or top[1] < 60:
            return {
                "encontrado": False,
                "mensagem": f"Nenhum produto corresponde a '{q}' no catálogo.",
                "parecidos": [
                    p.loc[pid, "name"] for pid, _, _ in sorted(ranked, key=lambda r: r[1], reverse=True)[:3]
                ],
            }
        tied = [r for r in ranked if r[2] == top[2] and r[1] >= top[1] - 3]
        if len(tied) > 1:
            return {
                "encontrado": False,
                "ambiguo": True,
                "mensagem": "Mais de um produto corresponde; pergunte ao cliente qual deles.",
                "candidatos": [product_card(p.loc[pid]) for pid, _, _ in tied[:6]],
                **({"total_candidatos": len(tied)} if len(tied) > 6 else {}),
            }
        row = p.loc[top[0]]

    out = {"encontrado": True, "produto": product_card(row, detailed=True)}
    if row["availability"] != "disponivel":
        out["alternativas_disponiveis"] = _alternatives(ctx, row)
    return out
