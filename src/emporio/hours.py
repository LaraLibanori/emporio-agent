"""Horário de funcionamento e feriados (seção 2 do manual), em código determinístico.

Usado pela ferramenta `status_loja_agora` e pelo contexto injetado a cada turno:
assim o modelo sempre sabe se a loja está aberta, sem depender de lembrar de
chamar a ferramenta (seção 7.1: fora do expediente, informar o horário de retorno).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

WEEK_HOURS = {  # weekday() -> (abre, fecha)
    0: ("09:00", "18:00"), 1: ("09:00", "18:00"), 2: ("09:00", "18:00"),
    3: ("09:00", "18:00"), 4: ("09:00", "18:00"), 5: ("09:00", "13:00"),
}
DIAS = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]
HORARIO_PADRAO = "segunda a sexta 09:00–18:00; sábado 09:00–13:00; domingos e feriados fechado"


def _easter(y: int) -> date:  # algoritmo anônimo gregoriano
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
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


def _at(dt: datetime, hhmm: str) -> datetime:
    return dt.replace(hour=int(hhmm[:2]), minute=int(hhmm[3:]), second=0, microsecond=0)


def next_opening(dt: datetime) -> datetime:
    d = dt
    for _ in range(14):
        hrs = WEEK_HOURS.get(d.weekday())
        if hrs and d.date() not in holidays(d.year):
            opens = _at(d, hrs[0])
            if opens > dt:
                return opens
        d = (d + timedelta(days=1)).replace(hour=0, minute=0)
    return d


@dataclass(frozen=True)
class StoreStatus:
    now: datetime
    is_open: bool
    holiday: str | None
    closes_at: str | None  # "18:00" se aberta
    reopens: datetime | None  # próxima abertura se fechada

    @property
    def reopens_label(self) -> str | None:
        if not self.reopens:
            return None
        r = self.reopens
        return f"{DIAS[r.weekday()]}, {r.strftime('%d/%m')} às {r.strftime('%H:%M')}"

    def summary(self) -> str:
        if self.is_open:
            return f"loja ABERTA agora (fecha hoje às {self.closes_at})"
        motivo = f"feriado: {self.holiday}" if self.holiday else "fora do horário"
        return f"loja FECHADA agora ({motivo}); a equipe volta {self.reopens_label}"


def store_status(dt: datetime) -> StoreStatus:
    hol = holidays(dt.year).get(dt.date())
    hrs = WEEK_HOURS.get(dt.weekday())
    is_open = bool(hrs and not hol and _at(dt, hrs[0]) <= dt < _at(dt, hrs[1]))
    return StoreStatus(
        now=dt,
        is_open=is_open,
        holiday=hol,
        closes_at=hrs[1] if is_open and hrs else None,
        reopens=None if is_open else next_opening(dt),
    )
