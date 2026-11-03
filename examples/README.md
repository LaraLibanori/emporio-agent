# Exemplos de conversa

Os arquivos `0X_*.md` desta pasta são conversas **reais** com o agente. Eles são gerados por:

```bash
python evals/run_evals.py --examples
```

O script roda os cenários marcados com `example=True` em `evals/scenarios.py` contra o LLM
configurado no `.env`. Cada turno registra a mensagem do cliente, as ferramentas chamadas
(com argumentos) e a resposta da Mel. A data da loja é fixada por cenário para que as regras
de prazo sejam reproduzíveis.
