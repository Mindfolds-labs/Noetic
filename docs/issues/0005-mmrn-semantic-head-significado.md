# 0005 — Validação matemática da camada semântica (significado/significante) no MMRN

## Contexto

A proposta adiciona ao MMRN o aprendizado de:

1. **significante** (forma/símbolo), já coberto por OCR geométrico;
2. **significado** (conceito, atributos, relações e contexto), via cabeças semânticas no espaço PRS.

## Formulação adotada

Dado o embedding noético `h ∈ R^d` (saída do PRS), definimos projeções lineares:

- `z_concept = Wc h + bc`
- `z_attr = Wa h + ba`
- `z_rel = Wr h + br`
- `z_ctx = Wx h + bx`

com perdas:

- `L_concept = CE(z_concept, y_concept)`
- `L_ctx = CE(z_ctx, y_ctx)`
- `L_attr = BCEWithLogits(z_attr, y_attr_multi_hot)`
- `L_rel = BCEWithLogits(z_rel, y_rel_multi_hot)`

e perda total:

`L_total = Σ λ_i L_i`

incluindo os termos geométricos já existentes (`L_cls, L_proj, L_contour, L_bezier, L_topo`).

## Coerência matemática

- **Domínio/codomínio**: logits não limitados em `R`; CE/BCE aplicadas diretamente sem sigmoid prévio (estável numericamente).
- **Convergência prática**: combinação ponderada por `λ` permite controlar competição entre tarefas.
- **Estabilidade**: inicialização e funções suaves (GELU + LayerNorm) ajudam no treinamento multiobjetivo.

## Riscos e recomendações

1. **Escala de gradientes**: BCE multi-rótulo pode dominar quando o número de atributos/relações é alto.
   - Mitigação: calibrar `lambda_attr`, `lambda_rel` e considerar `pos_weight` quando houver forte desbalanceamento.
2. **Colisão semântica**: conceitos abstratos (ex.: “amor”) exigem contexto textual/relacional rico, não apenas visão.
   - Mitigação: curriculum por concretude (objetos concretos → abstratos).
3. **Ambiguidade contextual**: mesma superfície lexical com múltiplos sentidos.
   - Mitigação: usar supervisão explícita de contexto (`context_id`) e negativos contrastivos.

## Referências

- Caruana, R. (1997). *Multitask Learning*.
- Kendall et al. (2018). *Multi-Task Learning Using Uncertainty to Weigh Losses*.
- Zhang & Yang (2021). *A Survey on Multi-Task Learning*.
- Goodfellow et al. (2016). *Deep Learning* (cap. de otimização e funções de perda).
