import pytest

torch = pytest.importorskip("torch")

from noetic_pawp.noetic_system import NoeticSystem


def test_noetic_system_build_and_perceive() -> None:
    system = NoeticSystem.build()
    out = system.perception.encode("cognitive multimodal")
    assert "points" in out and "ipa_ids" in out and "fused" in out
    assert isinstance(out["fused"], list)


def test_noetic_system_memory_bootstrap() -> None:
    system = NoeticSystem.build()
    concepts = system.ontology.load_bootstrap()
    system.memory.remember_concepts(concepts[:3])
    assert len(system.memory.concepts.all()) == 3


def test_noetic_system_action_controller() -> None:
    system = NoeticSystem.build()
    cn = torch.randn(2, 72)
    out = system.action.act(cn)
    assert "prs" in out
    assert out["prs"].shape[0] == 2
