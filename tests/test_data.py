from emporio.data import description_is_reliable, get_store


def test_integridade_e_derivados():
    s = get_store()
    assert len(s.products) == 65
    assert set(s.order_items["product_id"]) <= set(s.products.index)
    p = s.products
    assert p.loc[96, "availability"] == "esgotado"          # active, estoque 0
    assert p.loc[136, "availability"] == "descontinuado"    # status manda sobre estoque 13
    assert p.loc[130, "availability"] == "em_breve"


def test_promocoes_ativas_e_preco_final():
    p = get_store().products
    assert set(p[p["promo_discount"] > 0].index) == {90, 94, 121, 127}
    assert p.loc[121, "final_price"] == 439.20  # 549 - 20%
    assert p.loc[96, "promo_discount"] == 0     # promoções do 96 estão inativas


def test_descricoes_inconsistentes_sao_marcadas():
    p = get_store().products
    for pid in (135, 136, 137, 139, 140, 142, 144):
        assert not p.loc[pid, "description_reliable"], pid
    assert p.loc[95, "description_reliable"]
    assert p.loc[138, "description_reliable"]  # Music Man StingRay é coerente
    assert description_is_reliable("Fender Telecaster", "O autêntico twang da Fender")
