from noetic_pyfolds_bridge import NeuralState, NoeticPyFoldsBridge, SymbolicState


def test_bridge_pipeline_generates_valid_states() -> None:
    bridge = NoeticPyFoldsBridge(embedding_dim=16)

    symbolic_state = bridge.encode_text("Noetic and PyFolds integration", language="en")
    assert isinstance(symbolic_state, SymbolicState)
    assert len(symbolic_state.concept_embeddings) == 16
    assert len(symbolic_state.phonetic_features) == 16

    neural_input = bridge.convert_to_neural_input(symbolic_state)
    assert len(neural_input) == 16

    neural_state = bridge.run_pyfolds(neural_input)
    assert isinstance(neural_state, NeuralState)
    assert len(neural_state.spikes) == 16
    assert all(spike in (0.0, 1.0) for spike in neural_state.spikes)
    assert len(neural_state.membrane_potential) == 16
    assert len(neural_state.dendritic_states) == 16
    assert len(neural_state.surprise_trace) == 16

    predicted_state = [0.0] * 16
    decoded = bridge.decode_neural_state(predicted_state, neural_state)
    assert decoded["prediction_error"] >= 0.0
    assert decoded["surprise_signal"] >= 0.0
    assert len(decoded["adapted_attention_weights"]) == 16
    assert len(decoded["updated_cognitive_embeddings"]) == 16
