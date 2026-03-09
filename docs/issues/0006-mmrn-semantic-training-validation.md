# 0006 — Revisão científica: estabilidade e treinamento da camada semântica no MMRN

## Escopo da revisão

Esta revisão cobre:
- coerência matemática da perda multiobjetivo (OCR geométrico + semântica);
- estabilidade numérica e domínio das funções;
- impacto de parâmetros críticos (`lambda_*`) no aprendizado e convergência.

## Achados técnicos

1. **Combinação multiobjetivo válida**
   - `CE` para classes exclusivas (`concept_id`, `context_id`) e `BCEWithLogits` para multi-rótulo (`attributes`, `relations`) é a formulação correta.
2. **Risco de erro silencioso de anotação**
   - Sem validação de shape, alvos semânticos incorretos poderiam produzir gradientes inválidos ou erros tardios.
   - Correção aplicada: validação explícita dos tensores-alvo por tarefa.
3. **Parâmetros de perda potencialmente inválidos**
   - Pesos negativos em `lambda_*` distorcem o problema de otimização (incentivo a aumentar certas perdas).
   - Correção aplicada: validação para `lambda_* >= 0`.
4. **Normalização do termo topológico**
   - Mantida normalização por elementos espaciais para controlar escala entre resoluções.

## Recomendações para convergência

- **Ajuste de pesos por incerteza** (Kendall et al., 2018) para reduzir tuning manual dos `lambda_*`.
- **Rebalanceamento em multi-rótulo** com `pos_weight` quando atributos/relações forem esparsos.
- **Curriculum semântico**: conceitos concretos -> abstratos para reduzir entropia inicial.

## Status de validação experimental

- Tentativa de treino local bloqueada por ausência de `torch` no ambiente (restrição de rede/proxy para instalação).
- A validação executável nesta sessão ficou em checagens estáticas (`py_compile`) e revisão formal da formulação.

## Referências

- Caruana, R. (1997). *Multitask Learning*.
- Kendall, A., Gal, Y., Cipolla, R. (2018). *Multi-Task Learning Using Uncertainty to Weigh Losses*.
- Zhang, Y., Yang, Q. (2021). *A Survey on Multi-Task Learning*.
