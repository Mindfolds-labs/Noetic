# ADR 0003 — IPA Side-Channel

- **Status**: Proposto
- **Data**: 2026-03-09

## Contexto
O sistema já possui pistas fonéticas, mas falta um canal explícito e controlável para injetar IPA como side-channel em tarefas de treino/execução.

## Decisão
Expor o canal IPA como recurso opcional via `enable_ipa_channel`, preservando o comportamento baseline quando desligado.

## Alternativas consideradas
1. Fundir IPA diretamente no embedding principal sem flag.
2. Manter IPA apenas para análise offline (sem uso em runtime).
3. Usar somente regras ortográficas sem IPA explícito.

## Riscos
- Superdependência em qualidade de G2P para línguas com alta ambiguidade.
- Maior custo de memória por token.
- Interação inesperada com futuras camadas associativas.

## Plano de rollback
1. Desligar `enable_ipa_channel` na configuração.
2. Reverter scripts para execução sem side-channel.
3. Confirmar equivalência de outputs com baseline de referência.

## Métricas de sucesso
- Melhora em robustez a variações ortográficas/fonéticas nas suítes de teste.
- Regressão de throughput <= 8%.
- Ganho consistente em pelo menos 2 cenários de avaliação fonética.
