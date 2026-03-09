# Arquitetura PAWP v0.3

## Fluxo textual

1. Normalização Unicode.
2. Tokenização WordPiece.
3. G2P heurístico para IPA.
4. Alinhamento subword↔IPA.
5. Geração de `PAWPToken` enriquecido.

```text
Unicode -> WordPiece -> G2P(IPA) -> Align -> PAWPToken -> Embeddings -> Noetic/Transformer/GNC
```

## Fluxo de áudio (planejado)

```text
Áudio -> Encoder acústico -> Unidades fonéticas -> Align com subwords -> PAWPToken -> Modelo
```

## Formato de token

```python
PAWPToken(
    wp_piece: str,
    wp_id: int,
    ipa_units: list[str],
    phoneme_spans: list[tuple[int, int]],
    root_tag: str | None,
    lang: str | None,
    cn: list[float] | None,  # placeholder
)
```

## Baselines recomendados

- Baseline A: WordPiece puro.
- Baseline B: WordPiece + root_tag.
- Baseline C: WordPiece + IPA.
- Baseline D: PAWP completo (IPA + alinhamento + root_tag).

## Limitações atuais

- G2P ainda heurístico (não linguístico completo).
- Alinhamento proporcional pode falhar em casos irregulares.
- `cn` ainda é placeholder para fase futura.
