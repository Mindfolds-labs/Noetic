# Arquitetura do Projeto (PAWP v0.4)

## Camadas

1. **Input adapter**: texto/áudio/OCR para Unicode ou unidades fonéticas.
2. **Tokenizer layer**: WordPiece base + enriquecimento PAWP.
3. **Representation layer**: fusão de embeddings (`PAWPFusion`).
4. **Core model**: encoder PyTorch baseline (`PAWPEncoderModel`).
5. **Noetic**: etapa futura (fora do escopo atual).

## Módulos implementados

- `config.py`: dataclasses de configuração do tokenizer/modelo.
- `unicode_rules.py`: normalização e pré-tokenização.
- `phonetics.py`: G2P heurístico (PT) para IPA units.
- `roots.py`: heurísticas de raiz/sufixo.
- `tokenizer.py`: `PAWPTokenizer`, alinhamento subword↔IPA, encode e comparação baseline.
- `fusion.py`: camada de fusão de embeddings (PyTorch).
- `model.py`: encoder Transformer pequeno para baseline (PyTorch).

## Observações

- O projeto está preparado para PyTorch, mas mantém import isolado para funcionar sem `torch` no fluxo de tokenização.
- Scripts de treino (`train_mnist.py`) exigem instalação explícita de `torch`/`torchvision`.


## Experimento de treino real

- Script: `scripts/train_pyfolds_digits.py`
- Dataset: `sklearn.datasets.load_digits`
- Compara: PyFolds+head linear vs baseline MLP
- Saída: `docs/pyfolds_digits_training.md` e `docs/pyfolds_digits_training.json`
