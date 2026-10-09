from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from serad_kg.anomaly_cache import (
    anomaly_cache_path,
    load_anomaly_cache,
    reject_positive_anomalies,
    save_anomaly_cache,
)


def test_anomaly_cache_round_trip(tmp_path: Path) -> None:
    path = anomaly_cache_path(tmp_path, 2)
    anomalies = np.asarray([[0, 0, 1], [1, 1, 2], [2, 0, 0]], dtype=np.int64)
    entities = {0: "France", 1: "Germany", 2: "Italy"}
    relations = {0: "Meet", 1: "Criticize"}

    save_anomaly_cache(
        path,
        anomalies,
        snapshot_index=2,
        source_time=48,
        id_to_entity=entities,
        id_to_relation=relations,
        generator="llm",
        model="test-model",
    )

    loaded = load_anomaly_cache(
        path,
        snapshot_index=2,
        num_entities=3,
        num_relations=2,
        id_to_entity=entities,
        id_to_relation=relations,
    )
    assert np.array_equal(loaded, anomalies)
    assert pd.read_csv(path)["source_time"].unique().tolist() == [48]


def test_anomaly_cache_rejects_wrong_snapshot(tmp_path: Path) -> None:
    path = anomaly_cache_path(tmp_path, 0)
    save_anomaly_cache(
        path,
        np.asarray([[0, 0, 1]], dtype=np.int64),
        snapshot_index=0,
        source_time=0,
        id_to_entity={0: "France", 1: "Germany"},
        id_to_relation={0: "Meet"},
        generator="llm",
        model="test-model",
    )

    with pytest.raises(ValueError, match="expected snapshot index 1"):
        load_anomaly_cache(path, snapshot_index=1, num_entities=2, num_relations=1)


def test_reject_positive_anomalies_removes_positives_and_duplicates() -> None:
    positives = np.asarray([[0, 0, 1]], dtype=np.int64)
    anomalies = np.asarray([[0, 0, 1], [1, 0, 0], [1, 0, 0], [2, 0, 0], [2, 0, 1]], dtype=np.int64)

    filtered = reject_positive_anomalies(anomalies, positives)

    assert filtered.tolist() == [[1, 0, 0], [2, 0, 0], [2, 0, 1]]
