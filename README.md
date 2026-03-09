# Noetic PAWP Prototype

Protótipo inicial do **PAWP (Phonetic-Assisted WordPiece)** para o projeto Noetic.

## Objetivo

Implementar uma base testável para o fluxo:

- **Texto**: `Unicode -> WordPiece -> PAWP -> embeddings -> modelo`
- **Áudio (futuro)**: `áudio -> representação fonética -> PAWP -> embeddings -> modelo`

Nesta versão (`v0.3`):

- WordPiece permanece como base textual.
- PAWP adiciona pista fonética (G2P heurístico) e alinhamento subword↔IPA.
- Há comparação automática entre baseline WordPiece e PAWP.

## Estrutura

```text
noetic_pawp/
  __init__.py
  config.py
  g2p.py
  align.py
  tokenizer.py
experiments/
  compare_wordpiece_vs_pawp.py
docs/
  architecture.md
tests/
  test_pawp_pipeline.py
```

## Execução rápida

```bash
PYTHONPATH=. python experiments/compare_wordpiece_vs_pawp.py
```

## Testes

```bash
PYTHONPATH=. python -m pytest -q
```

## Próximos passos

1. Substituir G2P heurístico por G2P robusto por idioma.
2. Melhorar alinhamento para dígrafos e nasalização.
3. Adicionar integração com treino (PyTorch) para comparação de métricas.
4. Conectar fluxo de áudio com encoder acústico.
