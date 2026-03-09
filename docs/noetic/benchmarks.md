# Noetic Benchmarks

## Metodologia
Medição por `time.perf_counter()` com amostragem de passos fixos para pipeline e preprocessador.

## Métricas
- tempo total
- throughput (samples/s)
- latência média por passo (ms)

## Interpretação
Compare throughput de datasets entre modos cache/prefetch e latência do RIVE por batch size.

## Modos de execução
```bash
python -m noetic_pawp.benchmarks.run --output-json benchmark.json
```
