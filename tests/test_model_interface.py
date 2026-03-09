import importlib


def test_model_module_has_clear_torch_dependency_message() -> None:
    spec = importlib.util.find_spec("torch")
    if spec is None:
        try:
            importlib.import_module("pawp.model")
            assert False, "pawp.model should fail without torch"
        except ImportError as exc:
            assert "requires PyTorch" in str(exc)
    else:
        mod = importlib.import_module("pawp.model")
        assert hasattr(mod, "PAWPEncoderModel")
