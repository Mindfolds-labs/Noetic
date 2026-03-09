# Prompts prontos para o Codex (Noetic + PyFolds)

## Prompt 1 — Implementar integração Noetic ↔ LEIBREG

```text
Você está no repositório Noetic. Implemente a integração completa entre Noetic e LEIBREG (PyFolds), incluindo ponte multimodal, testes e pipeline de treino.

CONTEXTO
- Noetic já possui blocos maduros (PAWP/tokenizer, RIVE, memória associativa).
- O LEIBREG no PyFolds oferece WordSpace, REGCore e Imagination.
- Objetivo: projetar texto/imagem/memória para espaço 4D comum, fundir modalidades e validar com treino.

ENTREGÁVEIS
1) Criar `noetic_pawp/leibreg_bridge.py`
   - `NoeticLeibregConfig` com dimensões, configs dos submódulos, fusion_mode, LR e weight_decay.
   - `NoeticLeibregBridge(nn.Module)` com:
     - extração de embedding texto/imagem/memória;
     - projeção para WordSpace 4D;
     - fusão (`mean`, `attention`, `gate`);
     - passagem por REGCore + Imagination;
     - `forward(..., return_intermediate=False)` retornando `fused_point`, `reg_output`, `imagination` e confiança quando existir;
     - `train_step(batch)` com loss de consistência multimodal.

2) Criar `noetic_pawp/training/train_multimodal.py`
   - `MultimodalDataset` para JSON `{text, image_path, concept}`;
   - `MultimodalTrainer` com DataLoader, collate, `train_epoch`, `validate`, `train`;
   - checkpoint periódico e final.

3) Criar testes de integração em `noetic_pawp/tests/test_leibreg_integration.py`
   - init da bridge;
   - forward texto-only;
   - forward imagem-only;
   - forward multimodal;
   - `train_step`;
   - pipeline de treino (1 época com dataset temporário).

4) Criar scripts:
   - `scripts/train_noetic_leibreg.py` (CLI: --data --epochs --batch_size --lr --save --device)
   - `scripts/test_noetic_leibreg.py` (smoke test completo)

CRITÉRIOS DE ACEITE
- `pytest noetic_pawp/tests/test_leibreg_integration.py -v` passa.
- `python scripts/test_noetic_leibreg.py` passa.
- Fluxo texto-only, imagem-only e multimodal funcional.
- Checkpoints salvos.
- Código tipado, com docstrings e imports corretos.

IMPORTANTE
- Verifique a estrutura real do repositório antes de importar módulos (ajuste paths de import se necessário).
- Não quebre APIs já existentes.
- Se algum módulo externo do PyFolds não existir no ambiente, implemente fallback controlado e explique no relatório final.
```

---

## Prompt 2 — Implementar lado PyFolds + teste real de treinamento

```text
Você está no repositório PyFolds. Implemente os componentes LEIBREG necessários para receber embeddings do Noetic e valide com um teste de treinamento executável.

OBJETIVO
Garantir que o PyFolds exponha uma API estável para:
- projeção multimodal em WordSpace 4D;
- raciocínio por proximidade (REGCore);
- imaginação conceitual (Imagination);
- treino mínimo reproduzível com métricas.

ENTREGÁVEIS
1) `pyfolds/leibreg/wordspace.py`
   - `WordSpaceConfig` e `WordSpace`
   - métodos: `project_text`, `project_image`, `project_memory`
   - entrada [batch, dim_in] → saída [batch, 4]

2) `pyfolds/leibreg/reg_core.py`
   - `REGCoreConfig` e `REGCore`
   - forward preservando dimensão 4D

3) `pyfolds/leibreg/imagination.py`
   - `ImaginationConfig` e `Imagination`
   - saída dict com `hypothesis` e opcional `confidence`

4) Treino mínimo de validação
   - script `scripts/train_leibreg_smoke.py`
   - gerar dados sintéticos de texto/imagem/memória já embeddados;
   - treinar por poucas épocas;
   - imprimir loss por época e salvar checkpoint.

5) Testes
   - `tests/test_leibreg_modules.py` cobrindo shapes dos três módulos;
   - `tests/test_leibreg_training_smoke.py` executando 1 treino curto (CPU) e validando:
     - loss finita;
     - checkpoint criado;
     - saída 4D consistente.

CRITÉRIOS DE ACEITE
- `pytest tests/test_leibreg_modules.py -v` passa.
- `pytest tests/test_leibreg_training_smoke.py -v` passa.
- `python scripts/train_leibreg_smoke.py --epochs 3 --device cpu` executa sem erro.

QUALIDADE
- Tipagem e docstrings.
- Sem dependências desnecessárias.
- Código determinístico (seed fixa) no smoke training.
- Relatório final com comandos executados e resultados.
```

---

## Dica de uso

1. Copie o Prompt 1 e rode no Codex dentro do repositório **Noetic**.
2. Copie o Prompt 2 e rode no Codex dentro do repositório **PyFolds**.
3. Compare os contratos de API (`WordSpaceConfig`, `REGCoreConfig`, `ImaginationConfig`) para manter compatibilidade entre os projetos.
