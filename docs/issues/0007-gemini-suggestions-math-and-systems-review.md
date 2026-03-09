# Issue 0007 — Revisão técnico-científica das sugestões (Gemini) no Noetic

## Escopo da revisão
Esta issue avalia **onde encaixar** as 7 sugestões no código atual, priorizando correções com risco controlado e justificativa matemática.

## Resultado executivo
- **Implementado agora (objetivo e seguro):** melhoria do alinhamento IPA em `noetic_pawp/align.py` de heurística proporcional para programação dinâmica monotônica (estilo DTW/edit-distance).
- **Recomendado para próxima fase:**
  1. Legendre GPU (alto impacto, baixo risco).
  2. Paralelização parcial de dendritos (alto impacto, risco moderado por mudança de dinâmica).
- **Postergado com justificativa:** uncertainty weighting, LSH, continuidade Bézier e DEQ, por demandarem alterações arquiteturais mais amplas/validação experimental.

## Mapeamento proposta → código atual

### 1) Alinhamento dinâmico (DTW-like) — **ALTA / já aplicado**
- **Antes:** alocação proporcional em `align_subwords_to_ipa`, sem modelar inserções/deleções fonéticas.
- **Agora:** grade dinâmica com custo de substituição + inserção/deleção, retrotraço e projeção para spans por subword.
- **Impacto matemático:** melhora correspondência monotônica entre sequência de grafemas e fonemas em casos não lineares (letras mudas, alofonia comum).

### 2) Legendre na GPU — **ALTA / pendente**
- O `RIVEEncoder._legendre_project` usa `np.linalg.lstsq` por canal dentro do forward.
- Isso não participa de autograd e força CPU, tornando o pipeline menos eficiente.
- **Plano seguro:** pré-computar base/discretização e projetor como buffer em torch; forward via matmul/einsum.

### 3) Paralelização de dendritos — **ALTA / pendente**
- Em `MPJRDLayer`, há lista de neurônios com loop Python e concatenação.
- **Risco:** refatorar para tensores 3D altera estatística interna (BatchNorm/ordem/escala de ativação).
- **Plano seguro:** primeiro vetorizar apenas integração por neurônio (sem trocar topologia global), medir equivalência numérica e só então expandir.

### 4) Uncertainty weighting da loss — **MÉDIA / pendente**
- Pode estabilizar multi-termo de loss, mas exige protocolo de treinamento e monitoramento da identificabilidade de `log_vars`.
- Recomenda-se ativar atrás de feature flag com ablação.

### 5) LSH na memória associativa — **MÉDIA / pendente**
- Troca de hash exato para vizinhança aproximada aumenta recall semântico, porém com maior complexidade operacional e tuning de colisão.

### 6) Continuidade C1/C2 em Bézier — **BAIXA / pendente**
- Válido para suavidade geométrica, mas depende de representação das curvas e definição de segmentos no treino atual.

### 7) DEQ no REGCore — **BAIXA / pesquisa**
- Aumenta capacidade, porém introduz desafios de convergência, memória implícita e tuning de fixed-point solver.

## Validação de estabilidade e domínio (mudança aplicada)
- Custo de edição usa valores positivos e caminho monotônico, evitando inversões de índice.
- Spans finais são forçados a cobrir o fim da sequência IPA no último token, mantendo contrato de cobertura total.
- Tokens silenciosos podem produzir span vazio, preservando consistência com inserção/deleção.

## Referências científicas úteis
- Müller, M. (2007). *Dynamic Time Warping*. In: Information Retrieval for Music and Motion.
- Kendall, Gal, Cipolla (2018). *Multi-Task Learning Using Uncertainty to Weigh Losses*.
- Bai, Kolter, Koltun (2019). *Deep Equilibrium Models*.

## Próximas ações recomendadas
1. Benchmark de alinhamento com conjunto controlado de pares grafema→IPA (pt/en) e métrica de boundary F1.
2. Refatoração Legendre para torch (com teste de equivalência numérica vs versão NumPy).
3. Vetorização incremental da MPJRD com teste de regressão de saída/loss por seed fixa.
