# Auditoria dos dados e das políticas

Feita antes de escrever as ferramentas, com pandas (CSVs) e `pdftotext` (PDF).
Cada achado vem com a decisão tomada pelo agente. As decisões estão no código
em `src/emporio/data.py` e `src/emporio/tools/`.

**Integridade referencial:** todos os vínculos batem (order_items → products/orders,
orders → customers, products → categories, promotions → products). Não há registros órfãos.

## categories.csv
| Achado | Decisão |
|---|---|
| As categorias 6 (Sopro Madeiras), 7 (Sopro Metais) e 8 (Cordas Orquestrais) não têm produtos | A busca devolve uma mensagem explícita de "sem itens nessa categoria". O agente nunca inventa produtos |

## products.csv (65 itens, ids 81–145)
| Achado | Decisão |
|---|---|
| Ids 135–144: a descrição contradiz o nome ou as specs (ex.: 135 "Music Man", mas a descrição diz "Fender Jazz Bass"; 137 "Yamaha", mas a descrição diz "Ibanez 5 cordas" e as specs dizem 4; 142 "Korg", mas a descrição diz "Roland 88 teclas" e as specs dizem 61) | Uma heurística de marca (`description_is_reliable`) marca a descrição como não confiável. Nesses casos o agente usa só nome + specs, e a descrição sai da busca e da resposta |
| Erro de digitação "Cedr" no nome do 84 | A busca e a resolução de nomes usam fuzzy matching (rapidfuzz) |
| 136 está `discontinued`, mas com estoque 13. 130 está `coming_soon`, com estoque 2 | O **status prevalece** sobre o estoque. Os dois aparecem como indisponíveis para compra |
| 96 (active) e 113 estão com estoque 0 | Aparecem como "esgotado". A ferramenta já devolve alternativas similares (política 7.3) |
| `specs` tem 5 esquemas diferentes, e a chave `strings` às vezes é o tipo (nylon/steel), às vezes a quantidade | As specs são tratadas como dicionário livre. Os filtros de cordas e de captação só se aplicam a violões e ukuleles |

## orders.csv / order_items.csv
| Achado | Decisão |
|---|---|
| Pedido 3: total 3450, quando a soma dos itens dá 3498. Pedido 20: total 1400, quando a soma dá 1488 | O agente informa o `total_brl` gravado e não recalcula |
| O desconto de PIX e o frete nunca aparecem nos totais | `total_brl` é tratado como valor faturado. O agente não explica a diferença |
| Produto 130 (`coming_soon`) aparece em pedidos | É dado histórico. O status do pedido é mostrado como está |
| Pedido 8 está `shipped` com previsão 2026-02-28, já vencida. Há pedidos pending/confirmed de fev–mar/2026 | A ferramenta calcula `previsao_vencida` / dias parados. O agente reconhece o atraso e oferece abrir chamado para um humano |
| Parcelas abaixo do mínimo da política nos pedidos 9, 16 e 18 | Isso é histórico. Simulações novas seguem a seção 3.1 |

## promotions.csv
| Achado | Decisão |
|---|---|
| Há mais de uma promoção para os produtos 93, 96 e 99 (todas inativas) | Só vale `is_active = 1`. Se houver mais de uma ativa, vale a maior (não são cumulativas) |
| São 4 promoções ativas (127, 121, 90, 94), nenhuma com data de validade | O preço promocional é sempre mostrado com o preço original e o percentual (6.2) |
| Uma promoção inativa citada pelo cliente | O agente diz com transparência que ela não está vigente (7.3) |

## customers.csv
| Achado | Decisão |
|---|---|
| O e-mail `bruno.martins@jmail.com` aparece nos ids 11 e 43 | A verificação exige **nº do pedido + e-mail ou telefone do titular daquele pedido**. Assim a ambiguidade não aparece |
| Há homônimos (Diego Castro: 13 e 47) | O nome nunca é usado como prova de identidade |

## Políticas (PDF)
| Achado | Decisão |
|---|---|
| O telefone na capa e na seção 1.2 é (67) 3341-4444. Na seção 7, o WhatsApp é (67) 3321-4500 | Uso o 3341-4444, que aparece duas vezes e está rotulado "Telefone / WhatsApp". A divergência fica registrada |
| O e-mail tem acento na seção 1.2 (`empóriodamusica`), o que é inválido | Uso `contato@emporiodamusica.com.br` |
| Parcelamento: a tabela diz "parcela mínima R$100", mas a seção 3.1 tem faixas, e a faixa "até 3x sem mínimo (exceto abaixo de R$50)" é ambígua | Vale a seção 3.1 (mais específica): parcela ≥ R$50 em 1–3x, ≥ R$80 em 4–6x, ≥ R$100 em 7–12x |
| O manual fala em "mais de 300 instrumentos", inclusive sopro, orquestrais e pianos digitais | **Os dados mandam:** o agente só oferece o que está no catálogo |
| A política de devolução conta 7 dias "após o recebimento", mas o dataset não tem data de entrega real | `estimated_delivery` é usado como proxy da data de recebimento (suposição documentada) |
