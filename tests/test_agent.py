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


def test_ferramenta_sem_parametros_compativel_com_gemini():
    from emporio.llm import _openai_function
    from emporio.tools import REGISTRY

    f = _openai_function(REGISTRY["status_loja_agora"].spec())
    assert "parameters" not in f
    assert "parameters" in _openai_function(REGISTRY["detalhes_produto"].spec())


def test_describe_error_mostra_mensagem_do_provedor():
    from emporio.llm import describe_error

    class E(Exception):
        status_code = 400
        body = {"error": {"status": "INVALID_ARGUMENT", "message": "API key not valid."}}

    assert "API key not valid" in describe_error(E())


# ---------------- adaptador nativo do Gemini (HTTP simulado) ----------------

def _gemini(handler):
    import httpx

    from emporio.llm import GeminiLLM

    s = Settings(provider="gemini", model="gemini-2.5-flash", api_key="AQ.teste",
                 base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    return GeminiLLM(s, client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_gemini_function_call_e_thought_signature(tmp_path, ctx):
    import httpx

    sent = []

    def handler(req):
        body = json.loads(req.content)
        sent.append(body)
        assert req.url.path == "/v1beta/models/gemini-2.5-flash:generateContent"  # ignora /openai do .env antigo
        assert req.headers["x-goog-api-key"] == "AQ.teste"
        decls = {d["name"]: d for d in body["tools"][0]["functionDeclarations"]}
        assert "parameters" not in decls["status_loja_agora"]
        if len(sent) == 1:
            return httpx.Response(200, json={"candidates": [{"content": {"role": "model", "parts": [
                {"functionCall": {"name": "detalhes_produto", "args": {"produto": "Takamine GD20"}},
                 "thoughtSignature": "SIG123"}]}}], "usageMetadata": {"promptTokenCount": 10}})
        model_turn, tool_turn = body["contents"][-2], body["contents"][-1]
        assert model_turn["parts"][0]["thoughtSignature"] == "SIG123"
        assert tool_turn["parts"][0]["functionResponse"]["name"] == "detalhes_produto"
        preco = tool_turn["parts"][0]["functionResponse"]["response"]["produto"]["preco"]
        return httpx.Response(200, json={"candidates": [{"content": {"role": "model", "parts": [
            {"text": f"Sai por {preco}!"}]}}]})

    agent = Agent(llm=_gemini(handler), settings=Settings(state_dir=tmp_path),
                  store=ConversationStore(tmp_path / "c.db"), tool_context_factory=lambda sid: ctx)
    r = agent.reply("s", "Quanto custa o Takamine GD20?")
    assert r.text == "Sai por R$ 2.199,00!"
    assert sent[0]["toolConfig"]["functionCallingConfig"]["mode"] == "AUTO"
    # segundo turno reaproveita histórico persistido (com a assinatura) sem erro
    agent.reply("s", "obrigado")
    assert any("SIG123" in json.dumps(c) for c in sent[-1]["contents"])


def test_gemini_erro_legivel():
    import httpx

    from emporio.llm import describe_error

    llm = _gemini(lambda req: httpx.Response(404, json={"error": {"code": 404, "status": "NOT_FOUND",
                                                                  "message": "models/x is not found"}}))
    try:
        llm.chat("s", [{"role": "user", "content": "oi"}], [])
        raise AssertionError("deveria falhar")
    except Exception as e:
        assert "NOT_FOUND" in describe_error(e) and "not found" in describe_error(e)


def test_build_llm_usa_gemini_nativo_para_url_do_google():
    from emporio.llm import GeminiLLM, OpenAICompatibleLLM, build_llm

    g = Settings(provider="openai", model="m", api_key="k",
                 base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    assert isinstance(build_llm(g), GeminiLLM)
    assert isinstance(build_llm(Settings(provider="gemini", model="m", api_key="k")), GeminiLLM)
    assert isinstance(build_llm(Settings(provider="openai", model="m", api_key="k", base_url=None)), OpenAICompatibleLLM)


def test_gemini_retry_em_timeout_e_fallback_thinking(monkeypatch):
    import httpx

    import emporio.llm as llm_mod

    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    calls = []

    def handler(req):
        body = json.loads(req.content)
        calls.append(body)
        if len(calls) == 1:
            raise httpx.ReadTimeout("lento")
        if len(calls) == 2:
            assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}
            return httpx.Response(400, json={"error": {"message": "thinking_level is not supported"}})
        assert "thinkingConfig" not in body["generationConfig"]
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    assert _gemini(handler).chat("s", [{"role": "user", "content": "oi"}], []).content == "ok"
    assert len(calls) == 3


# ---------------- regressões da revisão (nov/2026) ----------------

def test_mensagens_simultaneas_da_mesma_sessao_sao_serializadas(tmp_path, ctx):
    """WhatsApp: o cliente manda 2 mensagens seguidas; a 2ª resposta precisa ver a 1ª."""
    import threading
    import time

    seen: list[list[str]] = []
    first_started = threading.Event()

    def fn(system, msgs):
        seen.append([m["content"] for m in msgs if m["role"] == "user"])
        first_started.set()
        time.sleep(0.2)  # simula latência do modelo
        return LLMResponse("ok")

    agent, _ = make_agent(tmp_path, ctx, fn)
    t1 = threading.Thread(target=agent.reply, args=("s", "oi"))
    t1.start()
    first_started.wait(2)
    agent.reply("s", "quero um violão")
    t1.join()
    assert seen[1] == ["oi", "quero um violão"]


def test_config_padrao_coerente_com_o_readme(monkeypatch):
    for var in ("LLM_PROVIDER", "LLM_TEMPERATURE", "LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    s = Settings()
    assert s.provider == "gemini" and s.temperature == 1.0 and s.model
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert Settings().temperature == 0.2
    monkeypatch.setenv("LLM_TEMPERATURE", "0.5")
    assert Settings().temperature == 0.5
