"""Testes do loop do agente com um LLM roteirizado (sem rede)."""

import json

from emporio.agent import FALLBACK_REPLY, Agent
from emporio.config import Settings
from emporio.llm import AnthropicLLM, LLMResponse, ScriptedLLM, ToolCall
from emporio.memory import ConversationStore, window


def make_agent(tmp_path, ctx, fn):
    llm = ScriptedLLM(fn)
    agent = Agent(
        llm=llm,
        settings=Settings(state_dir=tmp_path),
        store=ConversationStore(tmp_path / "c.db"),
        tool_context_factory=lambda sid: ctx,
    )
    return agent, llm


def test_loop_executa_ferramenta_e_devolve_resultado_ao_modelo(tmp_path, ctx):
    def fn(system, msgs):
        if msgs[-1]["role"] == "user":
            return LLMResponse(None, [ToolCall("c1", "detalhes_produto", {"produto": "Takamine GD20"})])
        assert msgs[-1]["role"] == "tool"
        data = json.loads(msgs[-1]["content"])
        return LLMResponse(f"O {data['produto']['nome']} sai por {data['produto']['preco']}!")

    agent, _ = make_agent(tmp_path, ctx, fn)
    r = agent.reply("s1", "Quanto custa o Takamine GD20?")
    assert "R$ 2.199,00" in r.text
    assert [s.tool for s in r.trace] == ["detalhes_produto"]


def test_historico_persiste_entre_turnos(tmp_path, ctx):
    seen = []

    def fn(system, msgs):
        seen.append([m["role"] for m in msgs])
        return LLMResponse("ok")

    agent, _ = make_agent(tmp_path, ctx, fn)
    agent.reply("s1", "oi")
    agent.reply("s1", "tudo bem?")
    assert seen[-1] == ["user", "assistant", "user"]
    agent.reply("outra", "oi")
    assert seen[-1] == ["user"]  # sessões isoladas


def test_ferramenta_inexistente_nao_quebra(tmp_path, ctx):
    calls = iter([LLMResponse(None, [ToolCall("x", "hackear_sistema", {})]), LLMResponse("desculpe")])
    agent, _ = make_agent(tmp_path, ctx, lambda s, m: next(calls))
    r = agent.reply("s", "oi")
    assert r.text == "desculpe" and "erro" in r.trace[0].result


def test_falha_do_provedor_vira_mensagem_amigavel(tmp_path, ctx):
    def boom(system, msgs):
        raise RuntimeError("503")

    agent, _ = make_agent(tmp_path, ctx, boom)
    assert agent.reply("s", "oi").text == FALLBACK_REPLY
    assert agent.history.load("s")[-1]["role"] == "assistant"


def test_limite_de_rodadas(tmp_path, ctx):
    agent, llm = make_agent(
        tmp_path, ctx, lambda s, m: LLMResponse(None, [ToolCall("c", "status_loja_agora", {})])
    )
    r = agent.reply("s", "loop")
    assert "equipe" in r.text and len(llm.calls) == agent.settings.max_tool_rounds + 1


def test_contexto_de_data_vai_no_system_prompt(tmp_path, ctx):
    captured = {}

    def fn(system, msgs):
        captured["s"] = system
        return LLMResponse("ok")

    agent, _ = make_agent(tmp_path, ctx, fn)
    agent.reply("s", "oi")
    assert "25/03/2026" in captured["s"] and "Mel" in captured["s"]


def test_janela_de_historico_corta_em_mensagem_de_usuario():
    msgs = []
    for i in range(5):
        msgs += [
            {"role": "user", "content": str(i)},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "type": "function",
                                                                     "function": {"name": "x", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "a", "content": "{}"},
            {"role": "assistant", "content": "r"},
        ]
    w = window(msgs, 2)
    assert w[0] == {"role": "user", "content": "3"} and len(w) == 8


def test_conversao_para_anthropic_agrupa_tool_results():
    msgs = [
        {"role": "user", "content": "oi"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
            {"id": "2", "type": "function", "function": {"name": "b", "arguments": '{"x": 1}'}},
        ]},
        {"role": "tool", "tool_call_id": "1", "content": "{}"},
        {"role": "tool", "tool_call_id": "2", "content": "{}"},
    ]
    out = AnthropicLLM._convert(msgs)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert len(out[2]["content"]) == 2 and out[1]["content"][1]["input"] == {"x": 1}


def test_resposta_vazia_nao_corrompe_historico(tmp_path, ctx):
    agent, _ = make_agent(tmp_path, ctx, lambda s, m: LLMResponse(None))
    r = agent.reply("s", "oi")
    hist = agent.history.load("s")
    assert r.text and hist[-1]["content"] == r.text
    assert all(m.get("content") or m.get("tool_calls") for m in hist if m["role"] == "assistant")


def test_ultima_rodada_forca_texto_e_nao_executa_ferramenta(tmp_path, ctx):
    def fn(s, m):
        return LLMResponse("parcial", [ToolCall("c", "registrar_chamado",
                                                {"tipo": "outro", "resumo": "x", "contato_cliente": "y"})])

    agent, llm = make_agent(tmp_path, ctx, fn)
    r = agent.reply("s", "loop")
    assert llm.force_text_flags[-1] is True and llm.force_text_flags.count(True) == 1
    assert len(r.trace) == agent.settings.max_tool_rounds  # a última rodada não executou nada


def test_api_chat(tmp_path, ctx):
    from fastapi.testclient import TestClient

    import emporio.api as api

    api._agent, _ = make_agent(tmp_path, ctx, lambda s, m: LLMResponse("Olá!"))
    c = TestClient(api.app)
    assert c.post("/chat", json={"session_id": "67999", "message": "oi"}).json()["reply"] == "Olá!"
    assert c.post("/chat", json={"session_id": "67999", "message": "   "}).status_code == 422
    assert c.post("/chat", json={"session_id": "", "message": "oi"}).status_code == 422
