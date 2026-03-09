# ADR 0004 — Associative Attention v1

- **Status**: Proposto
- **Data**: 2026-03-09

## Contexto
A arquitetura alvo prevê mecanismos associativos (atenção/memória) e integração com RIVE. Porém, esses blocos ainda não devem substituir o fluxo principal sem controle de risco.

## Decisão
Introduzir a trilha associativa v1 totalmente protegida por flags:
- `enable_associative_attention`
- `enable_associative_memory`
- `enable_rive_adapter`

Todas desligadas por padrão.

## Alternativas consideradas
1. Lançar apenas atenção associativa sem memória.
2. Lançar memória associativa primeiro e adiar atenção.
3. Integrar RIVE adapter sem flags para acelerar validação.

## Riscos
- Combinação de múltiplas flags pode gerar matriz grande de cenários.
- Custo computacional elevado em protótipos associativos.
- Dificuldade de atribuir causalidade de ganhos/perdas entre módulos.

## Plano de rollback
1. Desativar as três flags associativas no runtime.
2. Rodar baseline sem blocos associativos em treino e inferência.
3. Congelar experimentos associativos até nova rodada de ADR.

## Métricas de sucesso
- Ganho de qualidade em benchmark-alvo sem degradar baseline global.
- Custos adicionais de memória/latência dentro de orçamento definido por experimento.
- Reprodutibilidade dos resultados em no mínimo 3 execuções com seeds distintas.
