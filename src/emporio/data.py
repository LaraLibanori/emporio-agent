"""Carregamento e tratamento dos dados operacionais (CSVs).

Decisões de tratamento (detalhadas no README):
- `specs` (JSON em texto) vira dicionário.
- Disponibilidade é derivada de `status` + `stock_quantity` em um único campo.
- Promoções: só valem as com `is_active == 1`. O preço final é calculado aqui,
  nunca pelo LLM.
- Descrições que citam uma marca diferente da marca do nome do produto
  (ex.: "Music Man Bass 1X" descrito como "Fender Jazz Bass") são marcadas
  como não confiáveis e não são mostradas ao cliente; usamos nome + specs.
- Telefones/e-mails são normalizados para a verificação de identidade.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .config import DATA_DIR

KNOWN_BRANDS = [
    "fender", "gibson", "ibanez", "prs", "music man", "yamaha", "pearl", "tama",
    "korg", "nord", "roland", "tagima", "giannini", "rozini", "shelby", "crafter",
    "martin", "taylor", "takamine", "kala", "ohana", "kalani",
]

AVAILABILITY_LABELS = {
    "disponivel": "disponível",
    "esgotado": "temporariamente esgotado",
    "descontinuado": "descontinuado (fora do catálogo)",
    "em_breve": "em breve (lançamento, ainda não disponível para venda)",
}

ORDER_STATUS_LABELS = {
    "pending": "aguardando confirmação de pagamento",
    "confirmed": "pagamento confirmado, em separação",
    "shipped": "enviado (em transporte)",
    "delivered": "entregue",
    "cancelled": "cancelado",
}

PAYMENT_LABELS = {
    "pix": "PIX",
    "debit": "cartão de débito",
    "boleto": "boleto bancário",
}

STRINGS_LABELS = {"nylon": "nylon", "steel": "aço"}


def normalize(text: str) -> str:
    """minúsculas + sem acentos + espaços simples (para busca)."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text.lower()).strip()


def digits(text: str) -> str:
    return re.sub(r"\D", "", str(text))


def payment_label(code: str) -> str:
    if code in PAYMENT_LABELS:
        return PAYMENT_LABELS[code]
    m = re.fullmatch(r"credit_(\d+)x", code or "")
    if m:
        return f"cartão de crédito em {m.group(1)}x"
    return code


def _brands_in(text: str) -> set[str]:
    t = normalize(text)
    return {b for b in KNOWN_BRANDS if re.search(rf"\b{re.escape(b)}\b", t)}


def description_is_reliable(name: str, description: str) -> bool:
    """Falso se a descrição cita marca(s) que não aparecem no nome."""
    name_brands = _brands_in(name)
    desc_brands = _brands_in(description)
    if not desc_brands:
        return True
    return desc_brands <= name_brands


def _availability(status: str, stock: int) -> str:
    if status == "discontinued":
        return "descontinuado"
    if status == "coming_soon":
        return "em_breve"
    return "disponivel" if stock > 0 else "esgotado"


@dataclass
class Store:
    products: pd.DataFrame
    categories: pd.DataFrame
    customers: pd.DataFrame
    orders: pd.DataFrame
    order_items: pd.DataFrame
    promotions: pd.DataFrame


def load_store(data_dir: Path | str = DATA_DIR) -> Store:
    data_dir = Path(data_dir)
    categories = pd.read_csv(data_dir / "categories.csv")
    products = pd.read_csv(data_dir / "products.csv")
    promotions = pd.read_csv(data_dir / "promotions.csv")
    customers = pd.read_csv(data_dir / "customers.csv", dtype={"phone": str})
    orders = pd.read_csv(
        data_dir / "orders.csv",
        dtype={"tracking_code": str, "notes": str},
        parse_dates=["order_date", "estimated_delivery"],
    )
    order_items = pd.read_csv(data_dir / "order_items.csv")

    # --- produtos -------------------------------------------------------
    products["specs"] = products["specs"].apply(lambda s: json.loads(s) if isinstance(s, str) else {})
    products["stock_quantity"] = products["stock_quantity"].fillna(0).astype(int)
    products = products.merge(
        categories.rename(columns={"name": "category", "description": "category_description"}),
        on="category_id",
        how="left",
    )
    products["availability"] = [
        _availability(s, q) for s, q in zip(products["status"], products["stock_quantity"])
    ]
    products["description_reliable"] = [
        description_is_reliable(n, d) for n, d in zip(products["name"], products["description"])
    ]

    active = promotions[promotions["is_active"] == 1]
    # Se houver mais de uma promoção ativa para o mesmo produto, vale a maior.
    best = active.sort_values("discount_percent", ascending=False).drop_duplicates("product_id")
    products = products.merge(
        best[["product_id", "discount_percent", "description"]].rename(
            columns={"description": "promo_name", "discount_percent": "promo_discount"}
        ),
        on="product_id",
        how="left",
    )
    products["promo_discount"] = products["promo_discount"].fillna(0).astype(int)
    products["final_price"] = (products["price_brl"] * (1 - products["promo_discount"] / 100)).round(2)
    products["search_text"] = products.apply(_search_text, axis=1)
    products = products.set_index("product_id", drop=False).sort_index()

    # --- clientes / pedidos ----------------------------------------------
    customers["email_norm"] = customers["email"].str.strip().str.lower()
    customers["phone_digits"] = customers["phone"].map(digits)
    orders["notes"] = orders["notes"].fillna("")
    orders["tracking_code"] = orders["tracking_code"].fillna("")

    return Store(
        products=products,
        categories=categories,
        customers=customers.set_index("customer_id", drop=False),
        orders=orders.set_index("order_id", drop=False),
        order_items=order_items,
        promotions=promotions,
    )


def _search_text(row: pd.Series) -> str:
    specs = row["specs"] or {}
    parts = [row["name"], row["category"]]
    parts += [str(v) for v in specs.values()]
    if specs.get("strings") == "steel":
        parts.append("aco folk")
    if specs.get("strings") == "nylon":
        parts.append("nylon classico")
    if specs.get("electronics") == "yes":
        parts.append("eletroacustico eletrico captacao")
    if row["description_reliable"]:
        parts.append(row["description"])
    return normalize(" ".join(parts))


@lru_cache(maxsize=1)
def get_store() -> Store:
    return load_store()
