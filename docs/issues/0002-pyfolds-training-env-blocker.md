# Issue 0002 — Bloqueio de treino PyFolds no ambiente atual

## Contexto
Solicitado treino real com PyFolds por **>=20 épocas** e geração de relatório de acurácia.

## Evidências técnicas
Tentativa de instalar dependências (`torch`, `torchvision`, `scikit-learn`, `pyfolds`) falhou por bloqueio de acesso ao índice de pacotes (erro de proxy/403), impedindo import das bibliotecas.

Sem essas libs, o pipeline `scripts/train_pyfolds_digits.py` não consegue executar o ciclo:
1. carregar dataset (`sklearn.datasets.load_digits`),
2. construir rede PyFolds (`pyfolds.network.NetworkBuilder`),
3. otimizar cabeça supervisionada em `torch`.

## Impacto matemático/experimental
Sem execução real:
- não há curva de convergência (loss vs epoch),
- não há métrica final de generalização (test_acc),
- não há avaliação de estabilidade do reward schedule no MPJRD.

## Correção objetiva aplicada
- Script de treino atualizado para aceitar `--epochs` com default **20**.
- Geração de saída parametrizada (`--out-prefix`) para facilitar rastreabilidade experimental.
- Imports pesados protegidos em runtime com erro explícito e acionável.
- Comentários técnicos adicionados nos pontos críticos do mapeamento de entrada e reward bipolar.

## Próximo passo recomendado
Rodar em ambiente com rede habilitada para instalação de dependências e executar:

```bash
python -m pip install torch torchvision scikit-learn pyfolds
python scripts/train_pyfolds_digits.py --epochs 20 --out-prefix pyfolds_digits_training_20ep
```

Isso produzirá:
- `docs/pyfolds_digits_training_20ep.json`
- `docs/pyfolds_digits_training_20ep.md`
