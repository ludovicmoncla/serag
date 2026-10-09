from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd
import torch

from ..data import PreparedSnapshot, load_vocabulary
from ..evaluation import anomaly_auprc
from ..pipeline import (
    aggregate_repeated_metrics,
    evaluate_snapshot,
    load_completed_result,
    set_random_seed,
)
from ..preparation import load_prepared_experiment
from .hvsse import HVSSE, build_heterogeneous_views


@dataclass(frozen=True)
class HVSSEConfig:
    data_dir: Path
    prepared_data_dir: Path
    output_dir: Path = Path("outputs/hvsse")
    embedding_dim: int = 128
    num_heads: int = 4
    neighbors_per_view: int = 2
    max_degree_bucket: int = 15
    contrastive_weight: float = 0.5
    margin: float = 1.0
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 200
    patience: int = 25
    min_delta: float = 1e-5
    seed: int = 42
    device: str = "auto"
    save_model: bool = True

    def __post_init__(self) -> None:
        if self.num_heads < 1:
            raise ValueError("num_heads must be at least 1")
        if self.embedding_dim < 2 or self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if self.neighbors_per_view < 1:
            raise ValueError("neighbors_per_view must be at least 1")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.epochs < 1 or self.patience < 1:
            raise ValueError("epochs and patience must be positive")
        if self.min_delta < 0:
            raise ValueError("min_delta must be non-negative")

    def resolved_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


def _corrupt_reference_triples(
    triples: torch.Tensor, num_entities: int, generator: torch.Generator
) -> torch.Tensor:
    corrupted = triples.clone()
    replace_head = torch.rand(len(triples), generator=generator).lt(0.5)
    replacements = torch.randint(num_entities, (len(triples),), generator=generator)
    subjects = corrupted[:, 0].cpu()
    objects = corrupted[:, 2].cpu()
    replacements = torch.where(
        replace_head & replacements.eq(subjects),
        (replacements + 1) % num_entities,
        replacements,
    )
    replacements = torch.where(
        ~replace_head & replacements.eq(objects),
        (replacements + 1) % num_entities,
        replacements,
    )
    corrupted[replace_head.to(corrupted.device), 0] = replacements[replace_head].to(
        corrupted.device
    )
    corrupted[(~replace_head).to(corrupted.device), 2] = replacements[~replace_head].to(
        corrupted.device
    )
    return corrupted


def train_model(
    prepared: PreparedSnapshot,
    num_entities: int,
    num_relations: int,
    config: HVSSEConfig,
) -> tuple[HVSSE, list[float], list[float], pd.DataFrame]:
    reference_indices = torch.nonzero(prepared.train_mask, as_tuple=False).flatten()
    views = build_heterogeneous_views(
        prepared.triples_numpy,
        prepared.train_mask.detach().cpu().numpy(),
        neighbors_per_view=config.neighbors_per_view,
        seed=config.seed,
        max_degree_bucket=config.max_degree_bucket,
    )
    model = HVSSE(
        num_entities=num_entities,
        num_relations=num_relations,
        all_triples=prepared.triples,
        reference_indices=reference_indices,
        edge_index=views[0].to(prepared.triples.device),
        edge_type=views[1].to(prepared.triples.device),
        view_neighbors=views[2].to(prepared.triples.device),
        head_degree_buckets=views[3].to(prepared.triples.device),
        tail_degree_buckets=views[4].to(prepared.triples.device),
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        max_degree_bucket=config.max_degree_bucket,
        contrastive_weight=config.contrastive_weight,
        margin=config.margin,
        dropout=config.dropout,
    ).to(prepared.triples.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    train_losses: list[float] = []
    validation_losses: list[float] = []
    history: list[dict[str, float | int]] = []
    best_auprc = float("-inf")
    best_weights = None
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        corrupted = _corrupt_reference_triples(
            prepared.triples[reference_indices], num_entities, generator
        )
        model.train()
        optimizer.zero_grad()
        train_loss, components = model.training_loss(corrupted)
        train_loss.backward()
        optimizer.step()
        train_losses.append(float(train_loss.detach()))

        model.eval()
        with torch.no_grad():
            _, _, all_scores = model.scores(prepared.graph, prepared.triples)
            validation_scores = all_scores[prepared.validation_mask]
            validation_labels = prepared.labels[prepared.validation_mask]
            validation_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                validation_scores, validation_labels
            )
            validation_auprc = anomaly_auprc(
                validation_labels.cpu().numpy(), validation_scores.cpu().numpy()
            )
        validation_losses.append(float(validation_loss))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_losses[-1],
                "train_translation_loss": float(components["translation"].detach()),
                "train_contrastive_loss": float(components["contrastive"].detach()),
                "validation_loss": validation_losses[-1],
                "validation_auprc": validation_auprc,
            }
        )
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            print(
                f"  Epoch {epoch:03d}/{config.epochs:03d} — "
                f"train loss: {train_losses[-1]:.4f} — "
                f"validation AUPRC: {validation_auprc:.4f}",
                flush=True,
            )
        if validation_auprc > best_auprc + config.min_delta:
            best_auprc = validation_auprc
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    model._cached_scores = None
    return model, train_losses, validation_losses, pd.DataFrame(history)


def run(config: HVSSEConfig) -> pd.DataFrame:
    started_at = time.perf_counter()
    set_random_seed(config.seed)
    device = config.resolved_device()
    entities = load_vocabulary(config.data_dir / "entity2id.txt")
    relations = load_vocabulary(config.data_dir / "relation2id.txt")
    prepared, metadata = load_prepared_experiment(
        config.prepared_data_dir,
        num_entities=len(entities.id_to_text),
        num_relations=len(relations.id_to_text),
        device=device,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    serialized = {
        key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()
    } | {
        "resolved_device": str(device),
        "prepared_manifest_sha256": metadata["manifest_sha256"],
        "implementation": "independent-paper-reproduction",
    }
    (config.output_dir / "config.json").write_text(
        json.dumps(serialized, indent=2), encoding="utf-8"
    )
    print("HVSSE — independent paper reproduction", flush=True)
    print(f"Prepared data: {config.prepared_data_dir}", flush=True)
    print(f"Device: {device} | examples: {len(prepared.labels):,}", flush=True)
    model, train_losses, validation_losses, history = train_model(
        prepared, len(entities.id_to_text), len(relations.id_to_text), config
    )
    history.to_csv(config.output_dir / "training_history.csv", index=False)
    metrics = evaluate_snapshot(
        model,
        prepared,
        entities,
        relations,
        config.output_dir,
        train_losses,
        validation_losses,
        save_model=config.save_model,
    )
    elapsed = time.perf_counter() - started_at
    metrics["elapsed_seconds"] = elapsed
    protocol = str(metadata["config"]["protocol"])
    result = pd.DataFrame([{"protocol": protocol, "status": "ok", **metrics}])
    result.to_csv(config.output_dir / "summary.csv", index=False)
    print(
        f"Completed {protocol}: test AUROC={metrics['auc_test']:.4f}, "
        f"AUPRC={metrics['auprc_test']:.4f}, F1={metrics['f1_test']:.4f} "
        f"in {elapsed:.1f}s",
        flush=True,
    )
    return result


def run_repeated(
    config: HVSSEConfig, seeds: list[int], *, resume: bool = False
) -> dict[str, pd.DataFrame]:
    unique_seeds = list(dict.fromkeys(seeds))
    if len(unique_seeds) < 2:
        raise ValueError("At least two distinct seeds are required")
    root_output = config.output_dir
    root_output.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in unique_seeds:
        seed_config = replace(config, seed=seed, output_dir=root_output / f"seed_{seed}")
        seed_result = load_completed_result(seed_config) if resume else None
        if seed_result is None:
            seed_result = run(seed_config).copy()
        else:
            seed_result = seed_result.copy()
            print(f"Resuming: completed HVSSE seed {seed} reused", flush=True)
        seed_result.insert(0, "seed", seed)
        results.append(seed_result)
    by_seed = pd.concat(results, ignore_index=True)
    by_seed.to_csv(root_output / "summary_by_seed.csv", index=False)
    aggregate = aggregate_repeated_metrics(by_seed[by_seed["status"].eq("ok")], ["protocol"])
    aggregate.to_csv(root_output / "summary_aggregate.csv", index=False)
    return {"by_seed": by_seed, "aggregate": aggregate}
