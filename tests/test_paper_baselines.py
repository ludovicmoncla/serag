import numpy as np
import torch
from torch_geometric.data import Data

from serad_kg.baselines.cca import CCA
from serad_kg.baselines.cca_cli import build_parser as build_cca_parser
from serad_kg.baselines.hvsse import HVSSE, build_heterogeneous_views
from serad_kg.baselines.hvsse_cli import build_parser as build_hvsse_parser


def test_cca_scores_and_contrastive_training_are_finite() -> None:
    triples = torch.tensor(
        [[0, 0, 1], [0, 1, 2], [3, 0, 1], [4, 1, 2], [5, 0, 0]], dtype=torch.long
    )
    model = CCA(
        torch.randn(6, 12),
        torch.randn(2, 12),
        triples[:3],
        hidden_dim=8,
        score_batch_size=2,
    )

    loss, components = model.training_loss(triples[:3], torch.ones(3))
    loss.backward()
    model.eval()
    text, structure, combined = model.scores(Data(), triples)

    assert torch.isfinite(loss)
    assert set(components) == {"text", "structure", "contrastive", "disagreement"}
    assert text.shape == structure.shape == combined.shape == (5,)
    assert torch.isfinite(combined).all()


def test_hvsse_views_use_only_training_examples_as_context() -> None:
    triples = np.asarray(
        [
            [0, 0, 1],
            [0, 1, 2],
            [3, 0, 1],
            [0, 0, 4],
            [5, 1, 1],
        ],
        dtype=np.int64,
    )
    train_mask = np.asarray([True, True, True, False, False])

    edge_index, _, neighbors, head_degree, tail_degree = build_heterogeneous_views(
        triples, train_mask, neighbors_per_view=2, seed=7
    )

    assert set(edge_index[0].tolist()).issubset({0, 1, 2})
    valid_neighbors = neighbors[neighbors.ge(0)]
    assert set(valid_neighbors.tolist()).issubset({0, 1, 2})
    assert head_degree.shape == tail_degree.shape == (5,)


def test_hvsse_scores_and_unsupervised_training_are_finite() -> None:
    triples_numpy = np.asarray(
        [
            [0, 0, 1],
            [0, 1, 2],
            [3, 0, 1],
            [0, 0, 4],
            [5, 1, 1],
        ],
        dtype=np.int64,
    )
    train_mask = np.asarray([True, True, True, False, False])
    triples = torch.as_tensor(triples_numpy)
    edge_index, edge_type, neighbors, head_degree, tail_degree = build_heterogeneous_views(
        triples_numpy, train_mask, neighbors_per_view=2, seed=7
    )
    model = HVSSE(
        num_entities=6,
        num_relations=2,
        all_triples=triples,
        reference_indices=torch.tensor([0, 1, 2]),
        edge_index=edge_index,
        edge_type=edge_type,
        view_neighbors=neighbors,
        head_degree_buckets=head_degree,
        tail_degree_buckets=tail_degree,
        embedding_dim=8,
        num_heads=2,
        dropout=0.0,
    )
    corrupted = torch.tensor([[5, 0, 1], [0, 1, 4], [2, 0, 1]])

    loss, components = model.training_loss(corrupted)
    loss.backward()
    model.eval()
    translation, view_score, combined = model.scores(Data(), triples)

    assert torch.isfinite(loss)
    assert set(components) == {"translation", "contrastive"}
    assert translation.shape == view_score.shape == combined.shape == (5,)
    assert torch.isfinite(combined).all()


def test_new_baseline_clis_accept_shared_prepared_data_arguments() -> None:
    cca = build_cca_parser().parse_args(
        ["--data-dir", "data", "--prepared-data-dir", "prepared"]
    )
    hvsse = build_hvsse_parser().parse_args(
        ["--data-dir", "data", "--prepared-data-dir", "prepared"]
    )

    assert str(cca.prepared_data_dir) == "prepared"
    assert str(hvsse.prepared_data_dir) == "prepared"
