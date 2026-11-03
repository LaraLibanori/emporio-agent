"""Testes determinísticos das ferramentas (sem LLM)."""

import json

from emporio.policies import get_policy_index
from emporio.tools import REGISTRY


def run(ctx, name, **args):
    return REGISTRY[name].run(ctx, args)


# ---------------- catálogo ----------------

def test_violoes_ate_1000_so_disponiveis_e_ordenados(ctx):
    r = run(ctx, "buscar_produtos", categoria="Violões", preco_max=1000, limite=15)
    precos = [float(x["preco"].replace("R$ ", "").replace(".", "").replace(",", ".")) for x in r["resultados"]]
    assert r["total_encontrado"] == 12
    assert precos == sorted(precos) and max(precos) <= 1000
    assert all(x["pode_comprar"] for x in r["resultados"])
    assert "observacoes" in r  # GF-3D (esgotado) e SN-7C (descontinuado) foram omitidos


def test_filtro_de_preco_usa_preco_promocional(ctx):
    # Ohana CK-20: 549 cheio, 439,20 na promoção
    r = run(ctx, "buscar_produtos", categoria="ukulele", preco_max=450, limite=15)
    assert 121 in [x["id"] for x in r["resultados"]]


def test_acessorio_fora_do_escopo(ctx):
    assert "aviso_escopo" in run(ctx, "buscar_produtos", termo="cordas para violão")
    assert "aviso_escopo" in run(ctx, "buscar_produtos", termo="pedal de guitarra")
    # mas "7 cordas" é característica de instrumento
    r = run(ctx, "buscar_produtos", termo="violão 7 cordas")
    assert r["total_encontrado"] == 4


def test_categoria_sem_produtos(ctx):
    r = run(ctx, "buscar_produtos", termo="saxofone")
    assert r["resultados"] == [] and "não temos" in r["aviso"]


def test_detalhes_por_nome_com_erro_de_digitacao(ctx):
    r = run(ctx, "detalhes_produto", produto="takamine gd20")
    assert r["produto"]["id"] == 95 and r["produto"]["preco"] == "R$ 2.199,00"
    assert run(ctx, "detalhes_produto", produto="martn d-28e")["produto"]["id"] == 108
    assert run(ctx, "detalhes_produto", produto="giannini gf3d")["produto"]["id"] == 96


def test_nome_ambiguo_e_inexistente(ctx):
    assert run(ctx, "detalhes_produto", produto="yamaha")["ambiguo"]
    r = run(ctx, "detalhes_produto", produto="Gibson SG")
    assert r["encontrado"] is False and "Gibson Les Paul Standard '50s" in r["parecidos"]


def test_indisponivel_traz_alternativas_do_mesmo_tipo(ctx):
    r = run(ctx, "detalhes_produto", produto="Shelby SN-7C")
    assert r["produto"]["pode_comprar"] is False
    assert all("7 Cordas" in a["nome"] for a in r["alternativas_disponiveis"])


def test_descricao_nao_confiavel_nao_vai_para_o_cliente(ctx):
    r = run(ctx, "detalhes_produto", produto="Music Man Bass 1X")
    assert r["produto"]["descricao"] is None and "nota_interna" in r["produto"]


# ---------------- pagamento / frete ----------------

def test_pix_nao_acumula_com_promocao(ctx):
    r = run(ctx, "calcular_pagamento", produto="Ohana CK-20")
    assert r["pix"]["valor"] == "R$ 439,20"
    sem_promo = run(ctx, "calcular_pagamento", produto="Takamine GD20")
    assert sem_promo["pix"]["valor"] == "R$ 2.089,05"  # 2199 * 0,95


def test_parcelamento_respeita_minimo_por_faixa(ctx):
    r = run(ctx, "calcular_pagamento", valor=439.20)
    assert r["credito"]["max_parcelas_sem_juros"] == 5  # 6x = 73,20 < 80
    r = run(ctx, "calcular_pagamento", valor=2199)
    assert r["credito"]["max_parcelas_sem_juros"] == 12  # 12x = 183,25
    r = run(ctx, "calcular_pagamento", valor=700)
    assert r["credito"]["max_parcelas_sem_juros"] == 7   # 7x = 100; 8x = 87,5 < 100


def test_pagamento_combinado_so_acima_de_2000(ctx):
    assert "permitido" in run(ctx, "calcular_pagamento", valor=2500)["pagamento_combinado"]
    assert "não disponível" in run(ctx, "calcular_pagamento", valor=1500)["pagamento_combinado"]


def test_frete(ctx):
    assert run(ctx, "estimar_frete", cidade="Campo Grande", valor_pedido=600)["valor"] == "grátis"
    assert run(ctx, "estimar_frete", cidade="campo grande", valor_pedido=300)["valor"] == "R$ 35,00"
    r = run(ctx, "estimar_frete", cidade="Dourados", produto="Pearl Kit 2")
    assert "grande_porte" in r and len(r["modalidades"]) == 3


# ---------------- pedidos ----------------

def test_pedido_exige_verificacao(ctx):
    ok = run(ctx, "consultar_pedido", numero_pedido=8, email_ou_telefone="anacarol.ferreira@coldmail.com")
    assert ok["verificado"] and ok["codigo_rastreio"] == "BRJL5544332BR"
    fail = run(ctx, "consultar_pedido", numero_pedido=8, email_ou_telefone="outra@pessoa.com")
    assert fail["verificado"] is False
    inexistente = run(ctx, "consultar_pedido", numero_pedido=999, email_ou_telefone="outra@pessoa.com")
    assert inexistente["mensagem"] == fail["mensagem"]  # não revela se o pedido existe
    assert "Ana" not in json.dumps(fail)


def test_pedido_aceita_telefone_em_varios_formatos(ctx):
    for tel in ["(67) 99654-7821", "67996547821", "99654-7821", "+55 67 99654 7821"]:
        assert run(ctx, "consultar_pedido", numero_pedido=8, email_ou_telefone=tel)["verificado"], tel


def test_bloqueio_apos_tentativas(ctx):
    for _ in range(3):
        run(ctx, "consultar_pedido", numero_pedido=8, email_ou_telefone="chute@x.com")
    r = run(ctx, "consultar_pedido", numero_pedido=8, email_ou_telefone="anacarol.ferreira@coldmail.com")
    assert r.get("bloqueado") and "Ana" not in str(r)


def test_alerta_de_atraso(ctx):
    r = run(ctx, "consultar_pedido", numero_pedido=8, email_ou_telefone="anacarol.ferreira@coldmail.com")
    assert "alerta" in r  # previsão 28/02/2026 < 25/03/2026


def test_arrependimento_dentro_e_fora_do_prazo(ctx_at):
    # pedido 7: entregue, previsão 17/02/2026
    dentro = run(ctx_at("2026-02-20"), "avaliar_devolucao_ou_troca", numero_pedido=7,
                 email_ou_telefone="leticia.rocha@jmail.com", motivo="arrependimento")
    assert dentro["elegivel"] is True and dentro["prazo_final"] == "24/02/2026"
    fora = run(ctx_at("2026-03-01"), "avaliar_devolucao_ou_troca", numero_pedido=7,
               email_ou_telefone="leticia.rocha@jmail.com", motivo="arrependimento")
    assert fora["elegivel"] is False


def test_defeito_troca_garantia_fabricante(ctx_at):
    args = dict(numero_pedido=6, email_ou_telefone="gabriel.araujo@jmail.com", motivo="defeito")
    # compra 18/01/2026, recebimento ~01/02/2026
    assert "via" not in run(ctx_at("2026-02-10"), "avaliar_devolucao_ou_troca", **args)
    assert run(ctx_at("2026-04-01"), "avaliar_devolucao_ou_troca", **args)["via"] == "garantia"
    late = run(ctx_at("2026-11-02"), "avaliar_devolucao_ou_troca", **args)
    assert late["elegivel"] is False and late["via"] == "garantia do fabricante"


def test_devolucao_de_pedido_nao_entregue(ctx):
    r = run(ctx, "avaliar_devolucao_ou_troca", numero_pedido=13,
            email_ou_telefone="fernanda.ribeiro@coldmail.com", motivo="arrependimento")
    assert r["elegivel"] is None and "cancelamento" in r["acao_sugerida"]


# ---------------- loja / políticas ----------------

def test_status_loja(ctx_at):
    assert run(ctx_at("2026-03-25T10:00"), "status_loja_agora")["aberta_agora"] is True   # quarta
    sab = run(ctx_at("2026-03-28T14:00"), "status_loja_agora")                            # sábado tarde
    assert sab["aberta_agora"] is False and "segunda" in sab["reabre"]
    fer = run(ctx_at("2026-11-02T10:00"), "status_loja_agora")                            # Finados
    assert fer["aberta_agora"] is False and fer["feriado_hoje"] == "Finados"
    sexta_santa = run(ctx_at("2026-04-03T10:00"), "status_loja_agora")
    assert sexta_santa["feriado_hoje"] == "Sexta-feira Santa"


def test_rag_politicas_recupera_secao_certa():
    ix = get_policy_index()
    casos = {
        "Me arrependi da compra, posso devolver?": "4.1",
        "qual o endereço da loja?": "1.2",
        "vocês abrem domingo?": "2",
        "posso parcelar em 10 vezes?": "3.1",
        "meu violão veio com defeito": "4.2",
        "quanto tempo demora a entrega para Dourados?": "5.2",
        "o código de rastreio chega por onde?": "5.3",
        "o desconto do pix vale junto com a promoção?": "6.2",
        "como peço a exclusão dos meus dados?": "9",
        "vocês vendem cordas?": "1",
    }
    for q, sec in casos.items():
        top = [c.section for c, _ in ix.search(q, k=2)]
        assert sec in top, (q, top)


def test_chamado_gera_protocolo_e_persiste(ctx):
    r = run(ctx, "registrar_chamado", tipo="reclamacao", resumo="atraso", contato_cliente="Ana 67 99999")
    assert r["protocolo"].startswith("EM-")
    assert (ctx.state_dir / "chamados.jsonl").read_text().count(r["protocolo"]) == 1


def test_ferramenta_robusta_a_argumentos_ruins(ctx):
    assert "erro" in REGISTRY["detalhes_produto"].run(ctx, {})
    assert "erro" in REGISTRY["detalhes_produto"].run(ctx, "{json quebrado")
    r = REGISTRY["detalhes_produto"].run(ctx, {"produto": "Takamine GD20", "inventado": 1})
    assert r["encontrado"] and "_aviso" in r


# ---------------- regressões da revisão de código ----------------

def test_frete_campo_grande_em_varios_formatos(ctx):
    for c in ["Campo Grande/MS", "Campo Grande, MS", "campo grande ms", "Campo Grande - MS", "CAMPO GRANDE"]:
        assert run(ctx, "estimar_frete", cidade=c, valor_pedido=600)["valor"] == "grátis", c
    assert "modalidades" in run(ctx, "estimar_frete", cidade="Campo Grande do Sul, PR")


def test_numero_de_pedido_como_float_ou_texto(ctx):
    for n in [8.0, "8", "#8", "8.0"]:
        assert run(ctx, "consultar_pedido", numero_pedido=n,
                   email_ou_telefone="anacarol.ferreira@coldmail.com")["verificado"], n
    assert run(ctx, "consultar_pedido", numero_pedido=8.5, email_ou_telefone="x")["verificado"] is False


def test_bloqueio_nao_zera_com_acerto_e_vale_por_pedido(ctx_at):
    ctx = ctx_at("2026-03-25T10:00", session="atacante")
    mine = dict(numero_pedido=8, email_ou_telefone="anacarol.ferreira@coldmail.com")
    run(ctx, "consultar_pedido", numero_pedido=1, email_ou_telefone="chute1@x.com")
    run(ctx, "consultar_pedido", numero_pedido=1, email_ou_telefone="chute2@x.com")
    assert run(ctx, "consultar_pedido", **mine)["verificado"]       # acerto não zera
    run(ctx, "consultar_pedido", numero_pedido=1, email_ou_telefone="chute3@x.com")
    assert run(ctx, "consultar_pedido", **mine).get("bloqueado")
    # trocar de sessão não libera chutes no mesmo pedido (limite por pedido = 5)
    for i in range(5):
        run(ctx_at("2026-03-25T10:00", session=f"s{i}"), "consultar_pedido",
            numero_pedido=3, email_ou_telefone=f"c{i}@x.com")
    r = run(ctx_at("2026-03-25T10:00", session="nova"), "consultar_pedido",
            numero_pedido=3, email_ou_telefone="isabela.moreira@jmail.com")
    assert r.get("bloqueado")
    # janela de 1h expira
    later = ctx_at("2026-03-25T11:30", session="atacante")
    assert run(later, "consultar_pedido", **mine)["verificado"]


def test_entradas_numericas_em_texto(ctx):
    r = run(ctx, "buscar_produtos", categoria="violão", preco_max="1.000,00")
    assert r["total_encontrado"] == 12
    assert "erro" in run(ctx, "buscar_produtos", preco_max="mil")
    assert "erro" in run(ctx, "calcular_pagamento", valor=float("nan"))
    assert "erro" in run(ctx, "calcular_pagamento", valor=100, quantidade=-2)
    assert run(ctx, "estimar_frete", cidade="Campo Grande", produto="Kalani KAL-700T", quantidade=2)["valor"] == "grátis"
