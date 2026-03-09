import importlib.util
from pathlib import Path


def _load_train_mnist_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_mnist.py"
    spec = importlib.util.spec_from_file_location("train_mnist_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_dataset_fallback(monkeypatch):
    tv = __import__("torchvision")
    module = _load_train_mnist_module()

    def _raise(*args, **kwargs):
        raise RuntimeError("download blocked")

    monkeypatch.setattr(module.datasets, "MNIST", _raise)
    ds, source = module.build_dataset()
    assert source == "fake_mnist"
    assert isinstance(ds, tv.datasets.FakeData)
