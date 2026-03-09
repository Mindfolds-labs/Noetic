# ADR 0001 — Escopo da v1 do PAWP

- **Status**: Aceito
- **Data**: 2026-03-08

## Contexto
A proposta PAWP pode crescer para múltiplas frentes (tokenização, visão, geometria, memória). Isso reduz testabilidade inicial.

## Decisão
Na v1, PAWP será limitado a:

1. WordPiece como base de segmentação textual;
2. enriquecimento fonético (IPA) por token;
3. alinhamento explícito subword ↔ IPA;
4. compatibilidade com modelos posteriores sem mudar arquitetura central.

Fora do escopo v1:

- visão/RIVE;
- atenção geométrica;
- memória associativa;
- substituição total de IDs por vetores geométricos.

## Consequências

- ganho de clareza experimental;
- baseline simples de comparar com WordPiece puro;
- facilita paper inicial com hipótese falsificável.
