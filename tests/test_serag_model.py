import pytest
import torch
from torch_geometric.data import Data

from serad_kg.model import Serag, balanced_bce_with_logits, pairwise_ranking_loss


def test_balanced_bce_is_invariant_to_majority_class_duplication() -> None:
    reference = balanced_bce_with_logits(
        torch.tensor([2.0, -1.0]),
        torch.tensor([1.0, 0.0]),
    )
    duplicated = balanced_bce_with_logits(
        torch.tensor(([2.0] * 100) + [-1.0]),
        torch.tensor(([1.0] * 100) + [0.0]),
    )

    assert duplicated == pytest.approx(reference)


def test_pairwise_ranking_loss_rewards_correct_ordering() -> None:
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    correctly_ranked = pairwise_ranking_loss(
        torch.tensor([2.0, 1.5, -1.0, -2.0]),
        labels,
        margin=0.5,
        randomize_pairs=False,
    )
    incorrectly_ranked = pairwise_ranking_loss(
        torch.tensor([-1.0, -2.0, 2.0, 1.5]),
        labels,
        margin=0.5,
        randomize_pairs=False,
    )

    assert correctly_ranked < incorrectly_ranked


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_auxiliary_losses_train_both_branches_at_alpha_extremes(alpha: float) -> None:
    torch.manual_seed(42)
    model = Serag(
        torch.randn(4, 6),
        torch.randn(2, 6),
        compressed_dim=3,
        plausibility_weight=alpha,
        auxiliary_loss_weight=0.25,
    )
    graph = Data(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        edge_attr=torch.tensor([0, 1, 0]),
        num_nodes=4,
    )
    triples = torch.tensor([[0, 0, 1], [1, 1, 2], [2, 0, 0], [3, 1, 1]])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    mask = torch.ones(4, dtype=torch.bool)

    loss, _ = model(graph, triples, labels, mask)
    loss.backward()

    assert model.local_scorer[0].weight.grad is not None
    assert model.local_scorer[0].weight.grad.abs().sum() > 0
    assert model.global_bias.grad is not None
    assert model.global_log_scale.grad is not None
    assert model.global_log_scale.grad.abs() > 0


def test_global_score_is_not_constrained_to_be_non_positive() -> None:
    model = Serag(torch.zeros(3, 4), torch.zeros(1, 4), compressed_dim=2)
    graph = Data(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_attr=torch.tensor([0, 0]),
        num_nodes=3,
    )
    triples = torch.tensor([[0, 0, 1], [1, 0, 2]])
    model.eval()
    with torch.no_grad():
        model.global_bias.fill_(100.0)
        _, global_score, _ = model.scores(graph, triples)

    assert torch.all(global_score > 0)


def test_train_statistics_put_both_branches_on_comparable_scales() -> None:
    torch.manual_seed(7)
    model = Serag(torch.randn(5, 8), torch.randn(2, 8), compressed_dim=4)
    graph = Data(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
        edge_attr=torch.tensor([0, 1, 0, 1]),
        num_nodes=5,
    )
    triples = torch.tensor(
        [[0, 0, 1], [1, 1, 2], [2, 0, 3], [3, 1, 4], [4, 0, 0]]
    )
    mask = torch.ones(5, dtype=torch.bool)
    model.eval()

    model.update_score_normalization(graph, triples, mask)
    local, global_score, _ = model.scores(graph, triples)

    assert local.std(unbiased=False).item() == pytest.approx(1.0, rel=1e-4)
    assert global_score.std(unbiased=False).item() == pytest.approx(1.0, rel=1e-4)


def test_score_normalization_disables_dropout_and_restores_training_mode() -> None:
    torch.manual_seed(7)
    model = Serag(torch.randn(5, 8), torch.randn(2, 8), compressed_dim=4)
    graph = Data(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
        edge_attr=torch.tensor([0, 1, 0, 1]),
        num_nodes=5,
    )
    triples = torch.tensor(
        [[0, 0, 1], [1, 1, 2], [2, 0, 3], [3, 1, 4], [4, 0, 0]]
    )
    mask = torch.ones(5, dtype=torch.bool)
    model.train()

    model.update_score_normalization(graph, triples, mask)
    first_statistics = (
        model.local_score_mean.clone(),
        model.local_score_std.clone(),
        model.global_score_mean.clone(),
        model.global_score_std.clone(),
    )
    model.update_score_normalization(graph, triples, mask)
    second_statistics = (
        model.local_score_mean,
        model.local_score_std,
        model.global_score_mean,
        model.global_score_std,
    )

    assert model.training
    for first, second in zip(first_statistics, second_statistics, strict=True):
        assert torch.equal(first, second)


def test_dynamic_score_normalization_updates_during_training_forward() -> None:
    torch.manual_seed(7)
    model = Serag(
        torch.randn(5, 8),
        torch.randn(2, 8),
        compressed_dim=4,
        score_normalization="dynamic",
    )
    graph = Data(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
        edge_attr=torch.tensor([0, 1, 0, 1]),
        num_nodes=5,
    )
    triples = torch.tensor(
        [[0, 0, 1], [1, 1, 2], [2, 0, 3], [3, 1, 4], [4, 0, 0]]
    )
    labels = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
    mask = torch.ones(5, dtype=torch.bool)
    model.train()

    model(graph, triples, labels, mask)

    assert not torch.equal(model.local_score_mean, torch.tensor(0.0))
    assert not torch.equal(model.global_score_std, torch.tensor(1.0))
