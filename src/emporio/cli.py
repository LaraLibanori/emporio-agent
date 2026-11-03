"""Interface de linha de comando (chat no terminal).

    python -m emporio.cli                 # nova conversa
    python -m emporio.cli --session 6799  # retoma conversa persistida
    python -m emporio.cli --debug         # mostra as ferramentas chamadas

Comandos no chat: /sair, /reset, /debug
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid

from rich.console import Console
from rich.panel import Panel

from .agent import Agent
from .config import Settings, missing_key_message

console = Console()


def main() -> None:
    ap = argparse.ArgumentParser(description="Chat com a Mel, assistente da Empório da Música")
    ap.add_argument("--session", default=None, help="id da sessão (ex.: telefone do cliente)")
    ap.add_argument("--debug", action="store_true", help="mostrar chamadas de ferramentas")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    if msg := missing_key_message(Settings()):
        console.print(f"[bold red]⚠ {msg}[/]")
        raise SystemExit(1)
    session = args.session or f"cli-{uuid.uuid4().hex[:8]}"
    debug = args.debug
    agent = Agent()
    console.print(
        Panel.fit(
            f"🎸 [bold]Empório da Música[/] — assistente virtual\n"
            f"sessão: [cyan]{session}[/]   comandos: /sair /reset /debug",
            border_style="magenta",
        )
    )
    while True:
        try:
            msg = console.input("[bold green]você ›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            continue
        if msg in ("/sair", "/exit", "/quit"):
            break
        if msg == "/reset":
            agent.reset(session)
            console.print("[dim]histórico apagado[/]")
            continue
        if msg == "/debug":
            debug = not debug
            console.print(f"[dim]debug {'ligado' if debug else 'desligado'}[/]")
            continue
        with console.status("Mel está digitando..."):
            r = agent.reply(session, msg)
        if debug:
            for s in r.trace:
                console.print(
                    f"[dim]⚙ {s.tool}({json.dumps(s.args, ensure_ascii=False)}) "
                    f"→ {json.dumps(s.result, ensure_ascii=False)[:300]} ({s.ms}ms)[/]"
                )
        console.print(f"[bold magenta]Mel ›[/] {r.text}\n")
    console.print("[dim]até mais! 🎶[/]")


if __name__ == "__main__":
    main()
