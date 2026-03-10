# 0009 — Roteamento determinístico de backends G2P e robustez a dependências opcionais

## Contexto
O módulo `noetic_pawp.g2p` suportava seleção implícita de backend com prioridade fixa e cache sobre `(text, lang, backend)` apenas em nível superficial da API pública.

## Problema técnico
1. **Prioridade rígida** dificultava reprodutibilidade entre ambientes (ex.: máquinas com/sem `epitran`).
2. **Dependências opcionais** exigem tratamento específico para ausência (`ModuleNotFoundError`) sem mascarar falhas reais de execução.
3. **Determinismo de cache** é requisito para experimentos comparáveis em pipeline multimodal; a chave deve refletir backend efetivo.

## Correção proposta
- Introduzir interface formal `G2PBackend` e backends explícitos:
  - `EpitranBackend`
  - `EspeakBackend`
  - `HeuristicFallbackBackend`
- Tornar a prioridade configurável em `PAWPConfig.g2p_backend_priority`.
- Implementar roteador por prioridade com fallback determinístico.
- Garantir que o cache de superfície dependa de `(text, lang, backend_name)`.
- Manter `word_to_ipa` como façade para retrocompatibilidade.

## Impacto esperado
- **Maior estabilidade experimental:** mesmo texto/idioma com mesmo backend gera saída reprodutível.
- **Portabilidade:** ambientes sem `epitran` funcionam via `espeak`/heurística sem `except Exception` amplo.
- **Compatibilidade:** chamadas legadas de `word_to_ipa` permanecem válidas.

## Riscos e mitigação
- Mudança de prioridade pode alterar IPA gerado; mitigado por configuração explícita e testes unitários de roteamento.
