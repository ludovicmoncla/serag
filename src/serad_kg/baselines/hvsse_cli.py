from __future__ import annotations

import argparse
from pathlib import Path

from .hvsse_pipeline import HVSSEConfig, run, run_repeated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the independent HVSSE reproduction on prepared data"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--prepared-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hvsse"))
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--neighbors-per-view", type=int, default=2)
    parser.add_argument("--max-degree-bucket", type=int, default=15)
    parser.add_argument("--contrastive-weight", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--save-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save model.pt after training",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = HVSSEConfig(
        data_dir=args.data_dir,
        prepared_data_dir=args.prepared_data_dir,
        output_dir=args.output_dir,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        neighbors_per_view=args.neighbors_per_view,
        max_degree_bucket=args.max_degree_bucket,
        contrastive_weight=args.contrastive_weight,
        margin=args.margin,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        seed=args.seed,
        device=args.device,
        save_model=args.save_model,
    )
    if args.seeds is None:
        run(config)
    else:
        run_repeated(config, args.seeds, resume=args.resume)


if __name__ == "__main__":
    main()
