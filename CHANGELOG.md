# Changelog

## [0.5.0] — 2026-03-09

### Added
- `noetic_pawp/interfaces.py`: contratos estáveis `CognitiveOutput` e `NoeticCore`
- `scripts/check_health.py`: health check pré-release automatizado
- Cache LRU em `PAWPTokenizer.wordpiece_tokenize` e `infer_root_segments`
- `PAWPTokenizer.clear_caches()` para reset explícito de estado
- `tests/unit/test_interfaces.py` e `tests/unit/test_tokenizer_cache.py`

### Changed
- `scripts/eval/sprint_gates.py` renomeado para `scripts/eval/evaluation_gates.py`
- `noetic_pawp/tests/` movido para `tests/integration_noetic/`

### Notes
- Integração com PyFolds permanece exclusivamente em `noetic_pawp/leibreg_bridge.py` e `noetic_pyfolds_bridge/`
- Próxima fase: refatoração do PyFolds (remoção de código duplicado)
