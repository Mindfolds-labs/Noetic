# Noetic ↔ PyFolds Cognitive Architecture

```text
+---------------------------+
| Input Text                |
+-------------+-------------+
              |
              v
+---------------------------+
| PAWP Tokenizer            |
| - word pieces             |
| - IPA / phonetic units    |
+-------------+-------------+
              |
              v
+---------------------------+
| Cognitive Embeddings      |
| - concept_embeddings      |
| - phonetic_features       |
| - language_id             |
+-------------+-------------+
              |
              v
+---------------------------+
| Noetic Cognitive Core     |
| (symbolic fusion)         |
+-------------+-------------+
              |
              v
+---------------------------+
| noetic_pyfolds_bridge     |
| NoeticPyFoldsBridge       |
| - encode_text             |
| - convert_to_neural_input |
| - run_pyfolds             |
| - decode_neural_state     |
+-------------+-------------+
              |
              v
+---------------------------+
| PyFolds Neural Engine     |
| - spikes                  |
| - membrane potentials     |
| - dendritic_states        |
+-------------+-------------+
              |
              v
+---------------------------+
| Surprise Signal           |
| prediction_error =        |
| ||predicted - actual||    |
+-------------+-------------+
              |
              v
+---------------------------+
| Feedback to Cognition     |
| - adapt attention         |
| - update embeddings       |
+---------------------------+
```

## Shared Data Contract

- **SymbolicState**
  - `concept_embeddings`
  - `phonetic_features`
  - `language_id`
- **NeuralState**
  - `spikes`
  - `membrane_potential`
  - `dendritic_states`
  - `surprise_trace`
