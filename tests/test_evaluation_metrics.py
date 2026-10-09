import numpy as np
import pytest

from serad_kg.evaluation import (
    anomaly_auprc,
    anomaly_classification_metrics,
    anomaly_precision_recall_threshold,
)


def test_anomaly_auprc_treats_zero_label_as_positive_class() -> None:
    labels = np.asarray([1, 1, 1, 0])
    plausibility_scores = np.asarray([0.9, 0.8, 0.7, 0.1])

    assert anomaly_auprc(labels, plausibility_scores) == pytest.approx(1.0)


def test_precision_recall_threshold_maximizes_anomaly_f1() -> None:
    labels = np.asarray([0, 0, 1, 1, 1])
    plausibility_scores = np.asarray([0.1, 0.3, 0.2, 0.8, 0.9])

    threshold = anomaly_precision_recall_threshold(labels, plausibility_scores)
    metrics = anomaly_classification_metrics(labels, plausibility_scores, threshold)

    assert threshold == pytest.approx(0.3)
    assert metrics == {"precision": 2 / 3, "recall": 1.0, "f1": 0.8}
