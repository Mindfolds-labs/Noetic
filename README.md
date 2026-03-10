# Noetic/PAWP: Arquitetura Experimental para Representação Linguística Multimodal

> **Status do projeto:** protótipo de pesquisa aplicado em Engenharia de Software, com trilhas experimentais controladas por *feature flags* e validação incremental por testes e *gates*.

## Resumo

Este repositório implementa uma base de referência para **PAWP (Phonetic-Assisted WordPiece)** e para um **núcleo noético recorrente (PyFolds/MMRN)**, com objetivo de investigar tokenização fonética assistida, fusão multimodal e integração com memória associativa.

A solução foi organizada para favorecer:
- **modularidade arquitetural** (componentes desacoplados);
- **rastreabilidade técnica** (ADR, documentação e *issues* técnicas);
- **validação contínua** (testes unitários, integração e *benchmarks*);
- **evolução segura** (flags experimentais e scripts de avaliação).

## 1. Escopo (visão de Engenharia de Software)

O sistema cobre, em estágio de prototipação:

1. **Pré-processamento multimodal** (texto/áudio/OCR);
2. **Tokenização base + enriquecimento fonético** (PAWP);
3. **Fusão de representações** para entrada de modelos;
4. **Núcleo noético recorrente** para dinâmica latente;
5. **Camadas de integração** com memória/ponte semântica (LeibReg);
6. **Exportação e avaliação** por scripts reprodutíveis.

> Fora de escopo atual: garantias de produção (SLA, observabilidade distribuída completa, hardening de segurança e versionamento formal de API pública).

## 2. Arquitetura de referência

```text
entrada (texto/áudio/OCR)
  -> normalização e adaptação
  -> tokenização WordPiece base
  -> PAWP (IPA + alinhamento + raiz + idioma)
  -> fusão de embeddings
  -> encoder/modelo (PyTorch)
  -> núcleo noético (PyFolds/MMRN)
  -> memória associativa / ponte semântica
  -> avaliação e export
```

### 2.1 Princípios arquiteturais

- **Separação de responsabilidades:** processamento, modelagem, integração e avaliação em módulos distintos.
- **Baixo acoplamento experimental:** trilhas de pesquisa ativadas por *feature flags*.
- **Testabilidade:** organização com suíte unitária e de integração.
- **Documentação viva:** ADRs e relatórios técnicos em `docs/`.

## 3. Estrutura do repositório

```text
.
├── core/                         # Núcleos de memória, atenção e wordspace
├── data/                         # Conceitos-semente e artefatos de dados
├── docs/                         # Arquitetura, ADRs, benchmarks e issues técnicas
├── experiments/                  # Experimentos comparativos e scripts exploratórios
├── noetic_pawp/                  # Implementações noéticas/PAWP e integrações
├── noetic_pyfolds_bridge/        # Ponte de integração Noetic <-> PyFolds
├── pyfolds/                      # Componentes de base LeibReg/PyFolds
├── scripts/                      # Treino, demonstração, avaliação e saúde do sistema
├── src/pawp/                     # Implementação principal PAWP (núcleo tokenização/fusão)
└── tests/                        # Testes unitários e de integração
```

## 4. Requisitos

- Python **3.10+** (recomendado);
- `pip` atualizado;
- Dependências do projeto via `pyproject.toml`.

Para rotinas de treino com visão computacional e módulos neurais:
- `torch`
- `torchvision`

## 5. Instalação

```bash
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest
```

## 6. Execução rápida

### 6.1 Demonstração de codificação

```bash
python scripts/demo_encode.py
```

### 6.2 Treino de referência (PyFolds digits)

```bash
python scripts/train_pyfolds_digits.py
```

Saídas esperadas:
- `docs/pyfolds_digits_training.md`
- `docs/pyfolds_digits_training.json`

### 6.3 Testes

```bash
pytest
```

## 7. Avaliação, qualidade e gates

O projeto adota avaliação incremental por *benchmarks* e *quality gates*:

```bash
python scripts/eval/benchmark_baseline_vs_new.py
python scripts/eval/evaluation_gates.py --sprint 1 --tokenizer-ok --ipa-ok --concept-ok
python scripts/eval/benchmark_rive_pge_experimental.py --enable-experimental-rive-pge
```

Recomenda-se executar os scripts de benchmark com o mesmo ambiente/dependências para comparabilidade dos resultados.

## 8. Núcleo noético (PyFolds/MMRN)

Componentes centrais adicionados:

- **`NoeticPyFoldsCore`**: dinâmica recorrente tipo membrana/spike com surrogate gradient e traço de surpresa;
- **`NoeticMMRNBridge`**: projeção da saída noética para espaço latente compatível com trilhas semânticas.

Arquivos de referência:
- `noetic_pawp/noetic_model.py`
- `tests/test_noetic_model.py`

## 9. Governança técnica

- **ADRs:** decisões arquiteturais em `docs/adr/`.
- **Issue logs técnicos:** investigações e validações em `docs/issues/`.
- **Rastreabilidade de evolução:** histórico em `CHANGELOG.md`.

## 10. Contribuição

Contribuições devem priorizar:

1. mudança mínima e coesa por PR;
2. atualização de documentação técnica quando houver impacto arquitetural;
3. inclusão/ajuste de testes para comportamento novo;
4. uso de *feature flags* para funcionalidades experimentais que alterem comportamento padrão.

## 11. Roadmap resumido

- consolidação de API noética;
- ampliação de cenários de benchmark multimodal;
- maturidade de exportação e inferência em ambientes móveis/web;
- formalização de critérios de aceitação para hardening de produção.

---

Se você estiver usando este repositório em contexto acadêmico/industrial, recomenda-se citar explicitamente o caráter **experimental** dos módulos noéticos e dos caminhos habilitados por flags.
