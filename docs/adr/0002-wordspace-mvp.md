# ADR 0002 — WordSpace MVP

- **Status**: Proposto
- **Data**: 2026-03-09

## Contexto
A base atual usa WordPiece + sinais fonéticos, mas a camada de representação semântica por espaço de palavras (WordSpace) ainda não foi introduzida. Precisamos testar essa hipótese sem quebrar a baseline já validada.

## Decisão
Implementar WordSpace como experimento incremental atrás da flag `enable_wordspace`, mantendo o pipeline atual como padrão.

## Alternativas consideradas
1. Ativar WordSpace por padrão imediatamente.
2. Adiar WordSpace até concluir os módulos de memória/atenção associativa.
3. Implementar WordSpace como branch separado sem integração no runtime principal.

## Riscos
- Aumento de complexidade de treino sem ganho claro em tarefas curtas.
- Regressão de latência/inferência por novas projeções.
- Dificuldade de debug caso WordSpace e IPA side-channel sejam ativados juntos.

## Plano de rollback
1. Desativar `enable_wordspace` em configuração central.
2. Reexecutar jobs com baseline atual para confirmar restauração de métricas.
3. Remover wiring experimental em scripts se houver custo operacional alto.

## Métricas de sucesso
- Não regressão de acurácia versus baseline (Δ <= 0.5 pp em validação).
- Overhead de tempo por batch <= 10%.
- Estabilidade de treino (sem NaN/divergência em 3 seeds consecutivas).
