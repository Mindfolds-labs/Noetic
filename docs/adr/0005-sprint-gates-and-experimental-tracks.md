# ADR 0005 — Gates por sprint e trilhas experimentais

## Status
Accepted

## Contexto
Precisamos introduzir governança de rollout para:

1. Qualidade de texto+IPA+conceito sem regressão.
2. Evolução de atenção associativa com ganho ou neutralidade controlada.
3. Ativação de multimodal apenas após estabilidade textual.
4. RIVE/PGE em trilha experimental behind flag com benchmark dedicado.

## Decisão

### Gate Sprint 1
Aprovado somente se todas as checagens abaixo passarem:
- tokenizer sem regressão
- alinhamento IPA sem regressão
- normalização conceitual sem regressão

### Gate Sprint 2
Mantém requisitos do Sprint 1 e adiciona:
- associação por attention com ganho mensurável **ou** neutralidade controlada

### Multimodal
`enable_multimodal` só deve ser considerado para ativação quando os gates do sprint atual estiverem aprovados.

### RIVE/PGE
RIVE/PGE segue trilha experimental sob `enable_experimental_rive_pge`, com benchmark isolado em `scripts/eval/benchmark_rive_pge_experimental.py`.

## Consequências
- Menor risco de regressão ao liberar novas capacidades.
- Clareza operacional para evolução incremental por sprint.
- Isolamento de experimentos de alto risco (RIVE/PGE).
