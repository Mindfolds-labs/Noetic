# 0009 — G2P backend priority and deterministic cache keys

## Context
The IPA side-channel is consumed by downstream alignment and multimodal fusion. Non-deterministic backend selection can produce distributional drift in phonetic units for identical `(text, lang)` inputs, affecting reproducibility.

## Technical notes
- We introduced an explicit backend priority list in `PAWPConfig` to make routing policy observable and configurable.
- Cache keys now include deterministic backend policy encoding (`backend_key`), preventing collisions across different routing configurations.
- Missing optional dependencies are handled with `ModuleNotFoundError`, avoiding broad exception swallowing that could hide semantic errors.

## Mathematical/system impact
Although G2P here is symbolic (not gradient-based), deterministic IPA mapping reduces variance in:
1. subword↔phoneme alignment spans,
2. multimodal feature construction,
3. any learned projections conditioned on IPA sequences.

This is equivalent to controlling exogenous noise in a preprocessing operator so that training and evaluation remain on the same induced input measure.
