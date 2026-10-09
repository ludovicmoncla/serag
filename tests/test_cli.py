from pathlib import Path

import pytest

from serad_kg.baselines.lognet_cli import build_parser as build_lognet_parser
from serad_kg.cli import build_parser


def test_parser_uses_extended_training_budget_by_default() -> None:
    args = build_parser().parse_args(["--data-dir", "data"])

    assert args.epochs == 300
    assert args.patience == 25
    assert args.edge_mask_ratio == pytest.approx(0.0)
    assert args.scheduler_patience == 8
    assert args.scheduler_factor == pytest.approx(0.5)
    assert args.min_learning_rate == pytest.approx(1e-6)
    assert args.ranking_loss_weight == pytest.approx(0.0)
    assert args.ranking_margin == pytest.approx(0.5)
    assert args.score_normalization == "dynamic"
    assert args.save_model is True
    assert args.output_dir == Path("outputs/serag")


def test_parser_accepts_multiple_alphas() -> None:
    args = build_parser().parse_args(
        ["--data-dir", "data", "--alphas", "0", "0.5", "1"]
    )

    assert args.alphas == [0.0, 0.5, 1.0]


def test_parser_rejects_alpha_outside_unit_interval() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--data-dir", "data", "--alpha", "1.1"])


def test_training_commands_can_disable_model_saving() -> None:
    serag_args = build_parser().parse_args(["--data-dir", "data", "--no-save-model"])
    lognet_args = build_lognet_parser().parse_args(
        ["--data-dir", "data", "--prepared-data-dir", "prepared", "--no-save-model"]
    )

    assert serag_args.save_model is False
    assert lognet_args.save_model is False


def test_lognet_selects_checkpoints_on_validation_auprc_by_default() -> None:
    args = build_lognet_parser().parse_args(
        ["--data-dir", "data", "--prepared-data-dir", "prepared"]
    )

    assert args.checkpoint_metric == "auprc"


def test_training_commands_accept_resume_mode() -> None:
    serag_args = build_parser().parse_args(["--data-dir", "data", "--resume"])
    lognet_args = build_lognet_parser().parse_args(
        ["--data-dir", "data", "--prepared-data-dir", "prepared", "--resume"]
    )

    assert serag_args.resume is True
    assert lognet_args.resume is True


def test_serag_parser_accepts_dynamic_score_normalization() -> None:
    args = build_parser().parse_args(
        ["--data-dir", "data", "--score-normalization", "dynamic"]
    )

    assert args.score_normalization == "dynamic"
