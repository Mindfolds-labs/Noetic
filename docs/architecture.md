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

## Encoder Unificado PyFolds (v1.0)

Implementação adicionada em `noetic_pawp/pyfolds_encoder.py` com pipeline completo:

1. `RIVEEncoder`: crops concêntricos + features geométricas + solve de Legendre com GCV para gerar `cn ∈ R^72`.
2. `RadialExtractor`: 8 raios em 4 eixos opostos com saída `x[B,4,2D]`.
3. `TemporalBuffer`: buffer online com `cn_dot` e `cn_ddot` em custo O(1).
4. `DendriticFuser`: partição uniforme `18+18+18+18` com replicação de `c0,c1,c2` por dendrito.
5. `MPJRDLayer`: integração dendrítica, soma somática, spikes, homeostase e campo de surpresa.
6. `IntentionCtrl`: estados `CURIOSITY/FOCUS/VIGILANCE/CONSOLIDATION` derivados de surpresa agregada.
7. `GeoTokenizer`: `tau_geo = [cn | cn_dot | cn_ipa] ∈ R^216`.
8. `UnifiedPyFoldsEncoder`: interface `step(image)` retornando saídas completas (`cn`, `tau_geo`, `state`, `R`, `V`, etc.).
