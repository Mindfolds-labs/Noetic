# Issue 0004 — Validação matemática do pipeline RIVE+MPJRD

## Correções objetivas aplicadas
1. **Consistência dimensional em `L_pds`**
   - Ajuste para comparar tensores `(B,)` vs `(B,)` no MSE (evitando broadcasting incorreto silencioso).
2. **Estabilidade da BerHu**
   - Clamp de `c >= 1e-6` para evitar divisão por quase zero quando erro é muito pequeno.
3. **Domínio do `n_eff`**
   - Regressão log-log com `clip([0.5, 5.0])` para limitar extrapolação não-física.
4. **Fluxo de gradiente**
   - `RIVEEncoder` executa em `no_grad` para respeitar premissa geométrica sem parâmetros e reduzir custo.

## Justificativa matemática
- BerHu segue prática de depth estimation robusta (Laina et al., 2016) com transição L1/L2 para outliers.
- Métricas de validação seguem padrão de Eigen et al. (AbsRel, RMSE, δ-threshold).
- Regularização de gradiente melhora preservação de descontinuidades de profundidade.

## Limitações científicas atuais
- Dataset demo usa pseudo-depth em `digits`; útil para verificação de fluxo, não para benchmark SOTA.
- `RIVEEncoder` tem loops Python/NumPy por imagem; adequado para protótipo, não ideal para alta escala.
- PDS-consistency atual usa proxy radial; recomenda-se substituir por estimador geométrico mais fiel quando houver GT adequado.

## Próximos passos recomendados
- Conectar loaders reais NYUv2/KITTI e avaliar com splits padronizados.
- Vetorizar partes do RIVE (ou cache offline de features) para throughput.
- Adicionar ablação (`lambda_grad`, `lambda_pds`, n_dendrites) e análise de convergência.

## Referências
- Eigen et al., *Depth Map Prediction from a Single Image using a Multi-Scale Deep Network* (NIPS 2014).
- Laina et al., *Deeper Depth Prediction with Fully Convolutional Residual Networks* (3DV 2016).
