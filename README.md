# PAWP Prototype (v0.4)

Repositório de referência para **PAWP (Phonetic-Assisted WordPiece)** com arquitetura modular e pronta para evolução em PyTorch.

## Arquitetura até o Noetic

```text
entrada (texto/áudio/OCR)
  -> normalização/adaptação
  -> WordPiece base
  -> PAWP (ipa + alinhamento + raiz + idioma)
  -> fusion (embeddings)
  -> encoder/modelo (PyTorch)
  -> Noetic (integração futura)
```

## Estrutura

```text
.
├── docs/
│   ├── architecture.md
│   └── adr/
│       └── 0001-pawp-v1-scope.md
├── scripts/
│   ├── demo_encode.py
│   ├── run_pawp_demo.py
│   ├── train_mnist.py
│   ├── train_pyfolds_digits.py
│   └── train_text_baseline.py
├── src/pawp/
│   ├── __init__.py
│   ├── config.py
│   ├── fusion.py
│   ├── model.py
│   ├── phonetics.py
│   ├── roots.py
│   ├── tokenizer.py
│   └── unicode_rules.py
└── tests/
    ├── test_model_interface.py
    └── test_tokenizer.py
```

## Setup

```bash
python -m pip install -e .
python -m pip install pytest
```

> Para treino com MNIST e módulos `fusion/model`: instale também `torch` e `torchvision`.

## Uso rápido

```bash
python scripts/demo_encode.py
python scripts/train_pyfolds_digits.py
pytest
```


## Resultado de treino real

Após rodar `python scripts/train_pyfolds_digits.py`, os resultados ficam em:

- `docs/pyfolds_digits_training.md`
- `docs/pyfolds_digits_training.json`


## Núcleo Noético (PyFolds)

Foi adicionado um núcleo noético com estado recorrente para integração com `cn` (72D):

- `NoeticPyFoldsCore`: dinâmica de membrana + spikes com surrogate gradient + traço de surpresa.
- `NoeticMMRNBridge`: projeta saída noética para um espaço latente tipo PRS.

Arquivos:
- `noetic_pawp/noetic_model.py`
- `tests/test_noetic_model.py`



## Avaliação e gates

```bash
python scripts/eval/benchmark_baseline_vs_new.py
python scripts/eval/evaluation_gates.py --sprint 1 --tokenizer-ok --ipa-ok --concept-ok
python scripts/eval/benchmark_rive_pge_experimental.py --enable-experimental-rive-pge
```
