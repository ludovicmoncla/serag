import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from serad_kg.preparation import (
    PreparationConfig,
    load_prepared_experiment,
    prepare_experiment,
)


def write_toy_dataset(directory: Path) -> None:
    directory.mkdir()
    (directory / "entity2id.txt").write_text(
        "A\t0\nB\t1\nC\t2\nD\t3\nE\t4\nF\t5\n", encoding="utf-8"
    )
    (directory / "relation2id.txt").write_text("R\t0\nS\t1\n", encoding="utf-8")
    rows = []
    for time in range(5):
        rows.extend(
            [
                (time % 6, 0, (time + 1) % 6, time),
                ((time + 1) % 6, 1, (time + 2) % 6, time),
                ((time + 2) % 6, 0, (time + 3) % 6, time),
                ((time + 3) % 6, 1, (time + 4) % 6, time),
            ]
        )
    pd.DataFrame(rows).to_csv(directory / "train.txt", sep=" ", header=False, index=False)


def write_ratio_dataset(directory: Path, snapshot_count: int = 30) -> None:
    directory.mkdir()
    event_count = snapshot_count * 10
    (directory / "entity2id.txt").write_text(
        "".join(f"E{identifier}\t{identifier}\n" for identifier in range(event_count * 2)),
        encoding="utf-8",
    )
    (directory / "relation2id.txt").write_text("R\t0\nS\t1\n", encoding="utf-8")
    rows = [
        (event * 2, event % 2, event * 2 + 1, event // 10)
        for event in range(event_count)
    ]
    pd.DataFrame(rows).to_csv(directory / "train.txt", sep=" ", header=False, index=False)


def test_pooled_preparation_is_disjoint_and_loadable(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "prepared"
    write_toy_dataset(data_dir)
    manifest = prepare_experiment(
        PreparationConfig(
            data_dir=data_dir,
            output_dir=output_dir,
            protocol="pooled",
            max_snapshots=5,
            anomalies_per_snapshot=4,
        )
    )

    triples_by_split = {
        split: set(
            map(
                tuple,
                manifest.loc[manifest["split"].eq(split), ["subject", "relation", "object"]]
                .to_numpy()
                .tolist(),
            )
        )
        for split in ("train", "validation", "test")
    }
    assert triples_by_split["train"].isdisjoint(triples_by_split["validation"])
    assert triples_by_split["train"].isdisjoint(triples_by_split["test"])
    assert triples_by_split["validation"].isdisjoint(triples_by_split["test"])
    prepared, metadata = load_prepared_experiment(
        output_dir, num_entities=6, num_relations=2, device=torch.device("cpu")
    )
    assert len(prepared.labels) == len(manifest)
    assert metadata["config"]["protocol"] == "pooled"


def test_manifest_checksum_detects_changes(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "prepared"
    write_toy_dataset(data_dir)
    prepare_experiment(
        PreparationConfig(
            data_dir=data_dir,
            output_dir=output_dir,
            protocol="pooled",
            max_snapshots=5,
            anomalies_per_snapshot=4,
        )
    )
    with (output_dir / "manifest.csv").open("a", encoding="utf-8") as stream:
        stream.write("0,0,1,1,train,0\n")

    with pytest.raises(ValueError, match="checksum"):
        load_prepared_experiment(
            output_dir, num_entities=6, num_relations=2, device=torch.device("cpu")
        )


@pytest.mark.parametrize(
    ("protocol", "protocol_options"),
    [
        ("pooled", {}),
        (
            "chronological",
            {
                "chronological_train_snapshots": 20,
                "chronological_validation_snapshots": 5,
            },
        ),
    ],
)
def test_ratio_variants_share_fixed_evaluation_and_nested_training_anomalies(
    tmp_path: Path, protocol: str, protocol_options: dict[str, int]
) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "prepared"
    write_ratio_dataset(data_dir)
    ratios = (0.01, 0.025, 0.05, 0.10)
    pool = prepare_experiment(
        PreparationConfig(
            data_dir=data_dir,
            output_dir=output_dir,
            protocol=protocol,
            max_snapshots=30,
            max_anomaly_ratio=0.20,
            evaluation_anomaly_ratio=0.10,
            train_anomaly_ratios=ratios,
            **protocol_options,
        )
    )

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["snapshot_count"] == 30
    assert metadata["validation_anomaly_ratio"] == 0.10
    assert metadata["test_anomaly_ratio"] == 0.10
    pool_test = pool.loc[pool["split"].eq("test")].reset_index(drop=True)
    test_positive_count = int(pool_test["label"].eq(1).sum())
    test_anomaly_count = int(pool_test["label"].eq(0).sum())
    assert test_anomaly_count == int(test_positive_count * 0.10 / (1.0 - 0.10))
    previous_anomalies: set[tuple[int, int, int]] = set()
    evaluation_checksums = set()
    for ratio in ratios:
        variant_dir = output_dir / f"train_ratio_{ratio:g}"
        variant = pd.read_csv(variant_dir / "manifest.csv")
        variant_test = variant.loc[variant["split"].eq("test")].reset_index(drop=True)
        pd.testing.assert_frame_equal(variant_test, pool_test)

        training = variant.loc[variant["split"].eq("train")]
        positive_count = int(training["label"].eq(1).sum())
        anomaly_count = int(training["label"].eq(0).sum())
        assert anomaly_count == int(positive_count * ratio / (1.0 - ratio))
        anomalies = set(
            map(
                tuple,
                training.loc[training["label"].eq(0), ["subject", "relation", "object"]]
                .to_numpy()
                .tolist(),
            )
        )
        assert previous_anomalies.issubset(anomalies)
        previous_anomalies = anomalies

        variant_metadata = json.loads(
            (variant_dir / "metadata.json").read_text(encoding="utf-8")
        )
        evaluation_checksums.add(variant_metadata["fixed_evaluation_sha256"])
        load_prepared_experiment(
            variant_dir,
            num_entities=600,
            num_relations=2,
            device=torch.device("cpu"),
        )
    assert len(evaluation_checksums) == 1


def test_anomaly_ratio_cannot_exceed_twenty_percent(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_toy_dataset(data_dir)
    with pytest.raises(ValueError, match="at most 0.20"):
        prepare_experiment(
            PreparationConfig(
                data_dir=data_dir,
                output_dir=tmp_path / "prepared",
                protocol="pooled",
                max_snapshots=5,
                max_anomaly_ratio=0.21,
            )
        )


def test_evaluation_ratio_requires_maximum_ratio(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_toy_dataset(data_dir)
    with pytest.raises(ValueError, match="evaluation_anomaly_ratio require"):
        prepare_experiment(
            PreparationConfig(
                data_dir=data_dir,
                output_dir=tmp_path / "prepared",
                protocol="pooled",
                max_snapshots=5,
                evaluation_anomaly_ratio=0.10,
            )
        )


def test_genai_chronological_anomalies_use_only_their_source_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "prepared"
    write_ratio_dataset(data_dir, snapshot_count=6)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    calls: list[set[int]] = []

    def fake_generate(
        count: int,
        context: list[tuple[str, str, str]],
        **_: object,
    ) -> list[tuple[str, str, str]]:
        context_entity_ids = {
            int(label.removeprefix("E"))
            for subject, _, object_ in context
            for label in (subject, object_)
        }
        calls.append(context_entity_ids)
        candidates = [(object_, relation, subject) for subject, relation, object_ in context]
        return candidates[:count]

    monkeypatch.setattr("serad_kg.preparation.generate_anomalies_genai", fake_generate)
    prepare_experiment(
        PreparationConfig(
            data_dir=data_dir,
            output_dir=output_dir,
            protocol="chronological",
            negative_sampling="genai",
            max_snapshots=6,
            chronological_train_snapshots=3,
            chronological_validation_snapshots=1,
            max_anomaly_ratio=0.20,
            train_anomaly_ratios=(0.10,),
        )
    )

    assert calls
    for snapshot_index in range(6):
        cache = pd.read_csv(
            output_dir / "generated_anomalies" / f"snapshot_{snapshot_index:03d}.csv"
        )
        lower = snapshot_index * 20
        upper = lower + 20
        assert cache["sub_id"].between(lower, upper - 1).all()
        assert cache["obj_id"].between(lower, upper - 1).all()

    def unexpected_generate(*_: object, **__: object) -> list[tuple[str, str, str]]:
        raise AssertionError("Completed snapshots should have been loaded from cache")

    monkeypatch.setattr("serad_kg.preparation.generate_anomalies_genai", unexpected_generate)
    prepare_experiment(
        PreparationConfig(
            data_dir=data_dir,
            output_dir=output_dir,
            protocol="chronological",
            negative_sampling="genai",
            max_snapshots=6,
            chronological_train_snapshots=3,
            chronological_validation_snapshots=1,
            max_anomaly_ratio=0.20,
            train_anomaly_ratios=(0.10,),
            resume_genai=True,
        )
    )
