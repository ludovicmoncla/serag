from __future__ import annotations

import argparse
from pathlib import Path

from .cca_pipeline import CCAConfig, run, run_repeated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the protocol-compatible CCA adaptation on prepared data"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--prepared-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cca"))
    parser.add_argument(
        "--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--entity-descriptions-file",
        type=Path,
        help="Optional headerless TSV containing entity_id and description",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--structural-weight", type=float, default=0.2)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--confidence-warmup-epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=8)
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
    config = CCAConfig(
        data_dir=args.data_dir,
        prepared_data_dir=args.prepared_data_dir,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model,
        entity_descriptions_file=args.entity_descriptions_file,
        hidden_dim=args.hidden_dim,
        structural_weight=args.structural_weight,
        contrastive_weight=args.contrastive_weight,
        contrastive_temperature=args.contrastive_temperature,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        score_batch_size=args.score_batch_size,
        epochs=args.epochs,
        confidence_warmup_epochs=args.confidence_warmup_epochs,
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
