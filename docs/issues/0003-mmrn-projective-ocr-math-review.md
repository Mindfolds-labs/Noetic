# Issue 0003 — Revisão matemática/técnica do MMRN + Projective OCR

## Escopo revisado
- Módulo Projective OCR com invariância a escala/ruído/perspectiva.
- Coerência das saídas \\((\hat{y}, \hat{C}, \hat{B}, g, q)\\).
- Loss híbrida \\(L = \sum_i \lambda_i L_i\\).

## Pontos validados
1. **Estabilidade do pré-normalizador (PPN)**
   - Inicialização da cabeça de transformação em identidade reduz risco de colapso geométrico nas primeiras épocas.
   - Uso de `grid_sample` e `affine_grid` mantém pipeline totalmente diferenciável.

2. **Consistência geométrica em contorno/Bézier**
   - Contorno supervisionado por magnitude Sobel normalizada evita dependência apenas de textura.
   - Regularização por segunda diferença em pontos de controle de Bézier impõe suavidade (curvatura limitada).

3. **Domínio/convergência da loss**
   - `cross_entropy` para classificação com logits sem saturação explícita.
   - `L_proj` via MSE contra identidade controla deriva projetiva sem necessidade de GT homografia em digits.
   - clipping de gradiente em 1.0 reduz explosão de gradiente em batches com forte ruído.

## Limitações abertas (cientificamente relevantes)
1. **Transformação afim != homografia completa**
   - O PPN atual modela 6 DoF; texto em perspectiva extrema demanda 8 DoF.
2. **Topo-loss aproximada**
   - Preservação de massa de contorno é proxy fraco para número de componentes/loops.
3. **Supervisão de Bézier indireta**
   - Sem anotação de curvas de referência, `L_bezier` atua como regularizador, não como ajuste supervisionado completo.

## Sugestões objetivas de evolução
- Trocar PPN por camada de homografia (DLT normalizado + restrição de condição numérica).
- Adicionar perda topológica por Euler characteristic aproximada ou persistent homology (bibliotecas TDA).
- Criar dataset sintético com GT de curvas para supervisão direta de `B_hat`.

## Referências de suporte
- Jaderberg et al., *Spatial Transformer Networks*, NeurIPS 2015.
- Simo-Serra et al., *Sketch Simplification by Deep Networks*, TOG 2016 (representações de traço/curva).
- Hu et al., *Topology-Preserving Deep Image Segmentation*, NeurIPS 2019 (perdas topológicas).
