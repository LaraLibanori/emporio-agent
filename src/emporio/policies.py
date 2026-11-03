"""Recuperação de trechos do Manual de Políticas (RAG léxico com BM25).

Por que BM25 e não embeddings?
- O manual tem ~8 páginas e ~25 seções. É um corpus pequeno, estruturado e
  escrito no mesmo vocabulário que o cliente usa ("troca", "frete", "PIX").
- BM25 roda offline, é determinístico, não precisa de modelo de embedding
  nem de banco vetorial, e é trivial de testar.
- Chunking por *seção* do manual (e não por número fixo de tokens) mantém
  regras completas juntas e permite citar a seção na resposta.
A interface `PolicyIndex.search` permite trocar por busca vetorial/híbrida
sem mexer no agente, caso o manual cresça.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .config import DATA_DIR
from .data import normalize

POLICY_PDF = DATA_DIR / "politicas_da_loja.pdf"

_HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2})?)\.?\s+(.+)$")
_NOISE = re.compile(
    r"^(Empório da Música Manual de Políticas e Procedimentos|Página \d+)$"
)

STOPWORDS = set(
    normalize(
        "a o e é de da do das dos em no na nos nas um uma uns umas para por com "
        "sem que se ao aos à às ou como mais mas meu minha seu sua eu voce vocês "
        "vcs vc qual quais quanto quando onde ja já posso pode tem ter ser foi "
        "isso esse essa este esta oi ola olá bom dia boa tarde noite obrigado "
        "obrigada por favor pra pro gostaria queria quero saber sobre loja"
    ).split()
)

# Expansão de consulta: vocabulário do cliente -> vocabulário do manual.
QUERY_EXPANSIONS = {
    ("devolv", "devoluc", "arrepend", "reembols", "estorn", "desist"): "devolucao arrependimento reembolso",
    ("troc",): "trocas troca preferencia defeito",
    ("horari", "abre", "abert", "fecha", "funcion", "expedient", "domingo", "sabado", "feriad"): "horario funcionamento",
    ("endereco", "endereço", "localiz", "onde fica", "fica onde", "chegar"): "endereco dados empresa",
    ("telefon", "whats", "contato", "email", "e-mail", "ligar"): "telefone whatsapp email contato",
    ("parcel", "vezes", "cartao", "credito", "debito", "pix", "boleto", "pagar", "pagament", "juros"): "pagamento parcelamento",
    ("frete", "entreg", "envi", "prazo", "chega", "correio", "sedex", "jadlog", "motoboy"): "frete entrega prazo",
    ("rastre", "rastrei", "codigo"): "rastreamento codigo",
    ("garant", "defeit", "quebr", "estrag", "problema"): "garantia defeito",
    ("promoc", "descont", "black", "oferta", "cupom", "liquidac"): "promocoes descontos",
    ("dados", "lgpd", "privacid", "excluir"): "privacidade protecao dados lgpd",
    ("acessori", "corda", "palheta", "cabo", "pedal", "amplificador", "case", "capa"): "acessorios exclusivamente instrumentos",
    ("reclam", "insatisf", "pessimo", "absurdo"): "reclamacao",
}


def _tokens(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", normalize(text))
    # "stemming" leve por prefixo: devolução/devolver -> devol, parcelas/parcelamento -> parce
    return [t[:6] if len(t) > 6 else t for t in toks if t not in STOPWORDS and len(t) > 1]


def expand_query(query: str) -> str:
    q = normalize(query)
    extra = [exp for keys, exp in QUERY_EXPANSIONS.items() if any(k in q for k in keys)]
    return " ".join([q, *extra])


@dataclass
class PolicyChunk:
    section: str  # ex.: "4.1"
    title: str  # ex.: "Política de Trocas e Devoluções > Direito de Arrependimento"
    text: str

    def render(self) -> str:
        return f"[Seção {self.section} — {self.title}]\n{self.text.strip()}"


def parse_policy_pdf(path: Path = POLICY_PDF) -> list[PolicyChunk]:
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]

    chunks: list[PolicyChunk] = [
        PolicyChunk("0", "Capa — contato e endereço", "\n".join(l for l in pages[0].splitlines() if l.strip()))
    ]
    parents: dict[str, str] = {}
    cur_sec, cur_title, buf = None, None, []

    def flush():
        if cur_sec is not None and "".join(buf).strip():
            chunks.append(PolicyChunk(cur_sec, cur_title, "\n".join(buf)))

    for page in pages[1:]:
        for line in page.splitlines():
            line = line.strip()
            if not line or _NOISE.match(line):
                continue
            m = _HEADING.match(line)
            if m and len(m.group(2)) <= 50 and ":" not in m.group(2) and not line.endswith("."):
                flush()
                sec, title = m.group(1), m.group(2).strip()
                if "." in sec:
                    title = f"{parents.get(sec.split('.')[0], '')} > {title}".strip(" >")
                else:
                    parents[sec] = title
                cur_sec, cur_title, buf = sec, title, []
            else:
                buf.append(line)
    flush()
    return chunks


class PolicyIndex:
    """BM25 (Okapi) sobre os trechos do manual."""

    def __init__(self, chunks: list[PolicyChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [_tokens(f"{c.title} {c.title} {c.text}") for c in chunks]  # título com peso 2
        self.tf = [Counter(d) for d in self.docs]
        self.avgdl = sum(map(len, self.docs)) / max(len(self.docs), 1)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def _score(self, q: list[str], i: int) -> float:
        tf, dl = self.tf[i], len(self.docs[i])
        s = 0.0
        for t in q:
            if t in tf:
                f = tf[t]
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def search(self, query: str, k: int = 3) -> list[tuple[PolicyChunk, float]]:
        q = _tokens(expand_query(query))
        scored = sorted(((self._score(q, i), i) for i in range(len(self.chunks))), reverse=True)
        return [(self.chunks[i], round(s, 2)) for s, i in scored[:k] if s > 0]


@lru_cache(maxsize=1)
def get_policy_index() -> PolicyIndex:
    return PolicyIndex(parse_policy_pdf())
