"""Roda os cenários contra o LLM configurado no .env e gera relatório + exemplos.

    python evals/run_evals.py                 # todos os cenários
    python evals/run_evals.py --only 04       # filtra por id
    python evals/run_evals.py --examples      # também (re)gera examples/*.md

Saída: evals/report.md (pass/fail por verificação) e, com --examples,
examples/<id>.md com a conversa real + ferramentas chamadas.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from evals.scenarios import SCENARIOS, Scenario  # noqa: E402


def run_scenario(sc: Scenario, state_dir: Path):
    os.environ["AGENT_NOW"] = sc.now
    from emporio.agent import Agent
    from emporio.config import Settings
    from emporio.tools import orders

    orders._failed_attempts.clear()
    settings = Settings(state_dir=state_dir)
    agent = Agent(settings=settings)
    session = f"eval-{sc.id}-{int(time.time())}"
    results, transcript = [], []
    for i, turn in enumerate(sc.turns, 1):
        t0 = time.perf_counter()
        r = agent.reply(session, turn.user)
        dt = time.perf_counter() - t0
        used = [s.tool for s in r.trace]
        checks = []
        for t in turn.tools:
            checks.append((f"chamou `{t}`", t in used))
        for t in turn.no_tools:
            checks.append((f"não chamou `{t}`", t not in used))
        for rx in turn.must:
            checks.append((f"contém /{rx}/", bool(re.search(rx, r.text, re.I))))
        for rx in turn.must_not:
            checks.append((f"não contém /{rx}/", not re.search(rx, r.text, re.I)))
        results.append((i, checks))
        transcript.append((turn.user, r, dt))
    return results, transcript


def write_example(sc: Scenario, transcript, model: str) -> None:
    lines = [
        f"# {sc.title}",
        "",
        f"> Conversa real gerada por `evals/run_evals.py --examples` · modelo `{model}` · "
        f"data simulada da loja: `{sc.now}`",
        "",
    ]
    for user, r, dt in transcript:
        lines += [f"**👤 Cliente:** {user}", ""]
        for s in r.trace:
            args = json.dumps(s.args, ensure_ascii=False)
            lines.append(f"<sub>⚙️ `{s.tool}({args})`</sub>  ")
        if r.trace:
            lines.append("")
        lines += [f"**🎸 Mel:** {r.text}", "", f"<sub>⏱ {dt:.1f}s</sub>", "", "---", ""]
    (ROOT / "examples" / f"{sc.id}.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--examples", action="store_true")
    args = ap.parse_args()

    from emporio.config import Settings

    model = f"{Settings().provider}:{Settings().model}"
    state = ROOT / "var" / "evals"
    state.mkdir(parents=True, exist_ok=True)
    total = passed = 0
    report = [f"# Relatório de evals\n\nModelo: `{model}`\n"]
    for sc in SCENARIOS:
        if args.only and args.only not in sc.id:
            continue
        print(f"▶ {sc.id} ...", flush=True)
        results, transcript = run_scenario(sc, state)
        report.append(f"## {sc.id} — {sc.title}\n")
        for i, checks in results:
            for name, ok in checks:
                total += 1
                passed += ok
                report.append(f"- turno {i}: {'✅' if ok else '❌'} {name}")
                if not ok:
                    print(f"   ❌ turno {i}: {name}")
        report.append("\n<details><summary>respostas</summary>\n")
        for user, r, _ in transcript:
            report.append(f"- **cliente:** {user}\n  - **tools:** {[s.tool for s in r.trace]}\n  - **Mel:** {r.text}\n")
        report.append("</details>\n")
        if args.examples and sc.example:
            write_example(sc, transcript, model)
    report.insert(1, f"**Resultado: {passed}/{total} verificações ({passed / max(total, 1):.0%})**\n")
    (ROOT / "evals" / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\n{passed}/{total} verificações passaram — ver evals/report.md")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
