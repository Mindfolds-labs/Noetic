import pytest

torch = pytest.importorskip("torch")

from core.wordspace.semantic_anchor_system import SemanticAnchorSystem
from core.wordspace.semantic_drift_monitor import SemanticDriftMonitor
from core.wordspace.semantic_losses import SemanticStabilityLoss
from core.wordspace.wordspace_point import WordSpacePoint


def test_wordspace_point_keeps_identity_context_separate() -> None:
    point = WordSpacePoint(
        text_vec=(1.0,),
        ipa_vec=(1.0,),
        context_vec=(0.2, 0.3),
        assoc_vec=(0.5, 0.7),
        core_identity=(1.0, 2.0),
        context_state=(0.5, -0.5),
    )
    assert point.core_identity == (1.0, 2.0)
    assert point.context_state == (0.5, -0.5)
    assert point.concept_representation == (1.5, 1.5)


def test_anchor_learning_rate_scaling() -> None:
    anchors = SemanticAnchorSystem("data/concepts/seed_concepts.json")
    cid = "concept.greeting.hello"
    core = torch.tensor([1.0, 1.0])
    delta = torch.tensor([1.0, 1.0])
    updated = anchors.apply_identity_update(cid, core, delta, base_lr=1.0, anchor_lr_scale=0.1)
    assert torch.allclose(updated, torch.tensor([1.1, 1.1]))


def test_semantic_stability_loss_and_drift_monitor() -> None:
    init = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    cur = torch.tensor([[0.98, 0.02], [0.02, 0.98]])

    loss = SemanticStabilityLoss()(cur, init)
    assert float(loss.item()) >= 0.0

    metrics = SemanticDriftMonitor(knn_k=1).measure(cur, init)
    assert metrics.semantic_drift_index >= 0.0
    assert 0.0 <= metrics.knn_stability <= 1.0
