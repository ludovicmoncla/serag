import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from serad_kg import pipeline
from serad_kg.config import TrainingConfig
from serad_kg.data import PreparedSnapshot, Vocabulary
from serad_kg.pipeline import (
    REPEATED_METRICS,
    aggregate_repeated_metrics,
    evaluate_snapshot,
    mask_training_edges,
)


class FixedScoreModel(nn.Module):
    def __init__(self, scores: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("fixed_scores", scores)

    def forward(self, graph, triples, labels, mask):
        return torch.tensor(0.0), self.fixed_scores

    def score_frame(self, graph, triples, labels, descriptions):
        return pd.DataFrame(
            {
                "triple": descriptions,
                "label": labels.numpy(),
                "local_score": self.fixed_scores.numpy(),
                "global_score": self.fixed_scores.numpy(),
                "score": self.fixed_scores.numpy(),
            }
        )


def test_edge_masking_keeps_anomalies_and_holds_out_positive_targets() -> None:
    triples = torch.tensor(
        [[0, 0, 1], [1, 0, 2], [2, 0, 3], [3, 0, 0], [0, 0, 2], [1, 0, 3]]
    )
    prepared = PreparedSnapshot(
        graph=Data(
            edge_index=triples[:4, [0, 2]].T,
            edge_attr=triples[:4, 1],
            edge_example_index=torch.arange(4),
            num_nodes=4,
        ),
        triples=triples,
        triples_numpy=triples.numpy(),
        labels=torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
        train_mask=torch.ones(6, dtype=torch.bool),
        validation_mask=torch.zeros(6, dtype=torch.bool),
        test_mask=torch.zeros(6, dtype=torch.bool),
    )
    torch.manual_seed(7)

    masked_graph, target_mask, masked_count = mask_training_edges(prepared, 0.5)

    assert masked_count == 2
    assert masked_graph.edge_index.shape[1] == 2
    assert target_mask[prepared.labels.eq(0)].all()
    assert int((target_mask & prepared.labels.eq(1)).sum()) == 2


def test_evaluation_selects_threshold_on_validation(tmp_path: Path) -> None:
    triples = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0]])
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    prepared = PreparedSnapshot(
        graph=Data(edge_index=torch.tensor([[0], [1]]), edge_attr=torch.tensor([0]), num_nodes=2),
        triples=triples,
        triples_numpy=triples.numpy(),
        labels=labels,
        train_mask=torch.tensor([True, True, False, False, False, False]),
        validation_mask=torch.tensor([False, False, True, True, False, False]),
        test_mask=torch.tensor([False, False, False, False, True, True]),
    )
    model = FixedScoreModel(torch.tensor([0.1, 0.9, 0.2, 0.8, 0.4, 0.6]))
    entities = Vocabulary({"A": 0, "B": 1}, {0: "A", 1: "B"})
    relations = Vocabulary({"R": 0}, {0: "R"})

    metrics = evaluate_snapshot(model, prepared, entities, relations, tmp_path, [1.0], [0.8])

    assert metrics["threshold_validation"] == pytest.approx(0.2)
    assert metrics["auc_validation"] == pytest.approx(1.0)
    assert metrics["auprc_validation"] == pytest.approx(1.0)
    assert metrics["auc_test"] == pytest.approx(1.0)
    assert metrics["auprc_test"] == pytest.approx(1.0)
    assert metrics["precision_validation"] == pytest.approx(1.0)
    assert metrics["recall_validation"] == pytest.approx(1.0)
    assert metrics["f1_validation"] == pytest.approx(1.0)
    assert metrics["precision_test"] == pytest.approx(0.0)
    assert metrics["recall_test"] == pytest.approx(0.0)
    assert metrics["f1_test"] == pytest.approx(0.0)
    assert (tmp_path / "roc_validation.png").is_file()
    assert (tmp_path / "pr_validation.png").is_file()
    assert (tmp_path / "pr_test.png").is_file()
    assert (tmp_path / "model.pt").is_file()


def test_evaluation_can_skip_model_checkpoint(tmp_path: Path) -> None:
    triples = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0]])
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    prepared = PreparedSnapshot(
        graph=Data(edge_index=torch.tensor([[0], [1]]), edge_attr=torch.tensor([0]), num_nodes=2),
        triples=triples,
        triples_numpy=triples.numpy(),
        labels=labels,
        train_mask=torch.tensor([False, False, False, False]),
        validation_mask=torch.tensor([True, True, False, False]),
        test_mask=torch.tensor([False, False, True, True]),
    )
    model = FixedScoreModel(torch.tensor([0.1, 0.9, 0.2, 0.8]))
    entities = Vocabulary({"A": 0, "B": 1}, {0: "A", 1: "B"})
    relations = Vocabulary({"R": 0}, {0: "R"})

    evaluate_snapshot(
        model,
        prepared,
        entities,
        relations,
        tmp_path,
        [1.0],
        [0.8],
        save_model=False,
    )

    assert (tmp_path / "scores.csv").is_file()
    assert not (tmp_path / "model.pt").exists()


def test_repeated_metrics_include_mean_std_and_confidence_interval() -> None:
    rows = []
    for seed, auc_test in [(41, 0.7), (42, 0.8), (43, 0.9)]:
        row = {"seed": seed, "snapshot": 0}
        row.update({metric: auc_test for metric in REPEATED_METRICS})
        rows.append(row)

    aggregate = aggregate_repeated_metrics(pd.DataFrame(rows), ["snapshot"]).iloc[0]

    assert aggregate["n_seeds"] == 3
    assert aggregate["auc_test_mean"] == pytest.approx(0.8)
    assert aggregate["auc_test_std"] == pytest.approx(0.1)
    assert aggregate["auc_test_ci95_low"] < 0.8 < aggregate["auc_test_ci95_high"]
    assert aggregate["elapsed_seconds_mean"] == pytest.approx(0.8)


def test_repeated_run_writes_per_seed_and_aggregate_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(config: TrainingConfig) -> pd.DataFrame:
        value = config.random_seed / 100
        return pd.DataFrame(
            [
                {
                    "snapshot": 0,
                    "status": "ok",
                    **{metric: value for metric in REPEATED_METRICS},
                }
            ]
        )

    monkeypatch.setattr(pipeline, "run", fake_run)
    config = TrainingConfig(data_dir=tmp_path, output_dir=tmp_path / "repeated")

    results = pipeline.run_repeated(config, [41, 42, 43])

    assert results["by_seed"]["seed"].tolist() == [41, 42, 43]
    assert results["overall"].iloc[0]["auc_test_mean"] == pytest.approx(0.42)
    assert (config.output_dir / "summary_by_seed.csv").is_file()
    assert (config.output_dir / "summary_aggregate.csv").is_file()
    assert (config.output_dir / "overall_aggregate.csv").is_file()


def test_repeated_run_resume_reuses_matching_completed_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called_seeds = []

    def result_for(seed: int) -> pd.DataFrame:
        value = seed / 100
        return pd.DataFrame(
            [
                {
                    "snapshot": "pooled",
                    "status": "ok",
                    **{metric: value for metric in REPEATED_METRICS},
                }
            ]
        )

    def fake_run(config: TrainingConfig) -> pd.DataFrame:
        called_seeds.append(config.random_seed)
        return result_for(config.random_seed)

    monkeypatch.setattr(pipeline, "run", fake_run)
    config = TrainingConfig(
        data_dir=tmp_path,
        output_dir=tmp_path / "repeated",
        device="cpu",
        save_model=False,
    )
    completed_config = replace(
        config,
        output_dir=config.output_dir / "seed_41",
        random_seed=41,
    )
    completed_config.output_dir.mkdir(parents=True)
    serialized = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(completed_config).items()
    } | {"resolved_device": "cpu"}
    (completed_config.output_dir / "config.json").write_text(
        json.dumps(serialized), encoding="utf-8"
    )
    result_for(41).to_csv(completed_config.output_dir / "summary.csv", index=False)

    results = pipeline.run_repeated(config, [41, 42], resume=True)

    assert called_seeds == [42]
    assert results["by_seed"]["seed"].tolist() == [41, 42]
    assert results["overall"].iloc[0]["auc_test_mean"] == pytest.approx(0.415)


def test_alpha_sweep_runs_every_alpha_and_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called_configs = []

    def fake_run(config: TrainingConfig) -> pd.DataFrame:
        called_configs.append(config)
        value = config.plausibility_weight + config.random_seed / 100
        return pd.DataFrame(
            [
                {
                    "snapshot": 0,
                    "status": "ok",
                    **{metric: value for metric in REPEATED_METRICS},
                }
            ]
        )

    monkeypatch.setattr(pipeline, "run", fake_run)
    config = TrainingConfig(data_dir=tmp_path, output_dir=tmp_path / "alpha_sweep")

    results = pipeline.run_alpha_sweep(config, [0.25, 0.75], [41, 42])

    assert len(results["by_alpha"]) == 4
    assert set(results["by_alpha"]["alpha"]) == {0.25, 0.75}
    assert set(results["by_alpha"]["seed"]) == {41, 42}
    assert {called.output_dir.relative_to(config.output_dir).as_posix() for called in called_configs} == {
        "alpha_0.25/seed_41",
        "alpha_0.25/seed_42",
        "alpha_0.75/seed_41",
        "alpha_0.75/seed_42",
    }
    assert (config.output_dir / "summary_by_alpha_and_seed.csv").is_file()
    assert (config.output_dir / "alpha_aggregate.csv").is_file()


@pytest.mark.parametrize("alpha", [-0.01, 1.01])
def test_training_config_rejects_alpha_outside_unit_interval(
    tmp_path: Path, alpha: float
) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        TrainingConfig(data_dir=tmp_path, plausibility_weight=alpha)
