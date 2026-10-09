import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from serad_kg.baselines import lognet_pipeline
from serad_kg.baselines.lognet_pipeline import LogNetConfig
from serad_kg.pipeline import REPEATED_METRICS


def test_lognet_repeated_run_resume_reuses_matching_completed_seed(
    tmp_path: Path, monkeypatch
) -> None:
    prepared_data_dir = tmp_path / "prepared"
    prepared_data_dir.mkdir()
    manifest_path = prepared_data_dir / "manifest.csv"
    manifest_path.write_text("split,label\ntrain,1\n", encoding="utf-8")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    called_seeds = []

    def result_for(seed: int) -> pd.DataFrame:
        value = seed / 100
        return pd.DataFrame(
            [
                {
                    "protocol": "pooled",
                    "status": "ok",
                    **{metric: value for metric in REPEATED_METRICS},
                }
            ]
        )

    def fake_run(config: LogNetConfig) -> pd.DataFrame:
        called_seeds.append(config.seed)
        return result_for(config.seed)

    monkeypatch.setattr(lognet_pipeline, "run", fake_run)
    config = LogNetConfig(
        data_dir=tmp_path,
        prepared_data_dir=prepared_data_dir,
        output_dir=tmp_path / "repeated",
        device="cpu",
        save_model=False,
    )
    completed_config = replace(
        config,
        output_dir=config.output_dir / "seed_41",
        seed=41,
    )
    completed_config.output_dir.mkdir(parents=True)
    serialized = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(completed_config).items()
    } | {
        "resolved_device": "cpu",
        "prepared_manifest_sha256": manifest_sha256,
    }
    (completed_config.output_dir / "config.json").write_text(
        json.dumps(serialized), encoding="utf-8"
    )
    result_for(41).to_csv(completed_config.output_dir / "summary.csv", index=False)

    results = lognet_pipeline.run_repeated(config, [41, 42], resume=True)

    assert called_seeds == [42]
    assert results["by_seed"]["seed"].tolist() == [41, 42]
    assert results["aggregate"].iloc[0]["auc_test_mean"] == 0.415
