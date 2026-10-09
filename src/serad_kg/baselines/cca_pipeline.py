from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from ..data import PreparedSnapshot, Vocabulary, load_vocabulary
from ..evaluation import anomaly_auprc
from ..pipeline import (
    aggregate_repeated_metrics,
    encode_vocabulary,
    evaluate_snapshot,
    load_completed_result,
    set_random_seed,
)
from ..preparation import load_prepared_experiment
from .cca import CCA


@dataclass(frozen=True)
class CCAConfig:
    data_dir: Path
    prepared_data_dir: Path
    output_dir: Path = Path("outputs/cca")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    entity_descriptions_file: Path | None = None
    hidden_dim: int = 128
    structural_weight: float = 0.2
    contrastive_weight: float = 1.0
    contrastive_temperature: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 256
    score_batch_size: int = 256
    epochs: int = 20
    confidence_warmup_epochs: int = 4
    patience: int = 8
    min_delta: float = 1e-5
    seed: int = 42
    device: str = "auto"
    save_model: bool = True

    def __post_init__(self) -> None:
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2 for contrastive learning")
        if self.score_batch_size < 1:
            raise ValueError("score_batch_size must be at least 1")
        if self.epochs < 1 or self.patience < 1:
            raise ValueError("epochs and patience must be positive")
        if self.confidence_warmup_epochs < 0:
            raise ValueError("confidence_warmup_epochs must be non-negative")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
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


def _description_texts(vocabulary: Vocabulary, path: Path | None) -> list[str]:
    descriptions: dict[int, str] = {}
    if path is not None:
        frame = pd.read_csv(path, sep="\t", header=None, names=["id", "description"])
        if frame["id"].duplicated().any():
            raise ValueError(f"Duplicate entity identifier in {path}")
        descriptions = {
            int(identifier): str(description)
            for identifier, description in frame.itertuples(index=False, name=None)
        }
        invalid = set(descriptions) - set(vocabulary.id_to_text)
        if invalid:
            raise ValueError(f"Entity identifier outside the vocabulary in {path}: {min(invalid)}")
    return [
        (
            f"{vocabulary.id_to_text[index]}. {descriptions[index]}"
            if index in descriptions
            else vocabulary.id_to_text[index]
        )
        for index in range(len(vocabulary.id_to_text))
    ]


def _encode_inputs(
    config: CCAConfig,
    entities: Vocabulary,
    relations: Vocabulary,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoder = SentenceTransformer(config.embedding_model, device=str(device))
    entity_texts = _description_texts(entities, config.entity_descriptions_file)
    # Recent Sentence-Transformers versions return inference tensors. Clone them
    # outside its inference context so trainable projections can save their inputs
    # for backward even though the base embeddings themselves remain frozen.
    entity_embeddings = encoder.encode(
        entity_texts, convert_to_tensor=True, device=str(device)
    ).clone()
    relation_embeddings = encode_vocabulary(encoder, relations, device).clone()
    return entity_embeddings, relation_embeddings


def _adaptive_confidence(scores: torch.Tensor) -> torch.Tensor:
    """Map the current plausibility ranking to label-free confidence weights."""
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores)
    ranks[order] = torch.linspace(0.05, 1.0, len(scores), device=scores.device)
    return ranks


def train_model(
    prepared: PreparedSnapshot,
    entity_embeddings: torch.Tensor,
    relation_embeddings: torch.Tensor,
    config: CCAConfig,
) -> tuple[CCA, list[float], list[float], pd.DataFrame]:
    reference_indices = torch.nonzero(prepared.train_mask, as_tuple=False).flatten()
    reference_triples = prepared.triples[reference_indices]
    model = CCA(
        entity_embeddings,
        relation_embeddings,
        reference_triples,
        hidden_dim=config.hidden_dim,
        contrastive_temperature=config.contrastive_temperature,
        structural_weight=config.structural_weight,
        contrastive_weight=config.contrastive_weight,
        score_batch_size=config.score_batch_size,
    ).to(prepared.triples.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    confidence = torch.ones(len(reference_indices), device=prepared.triples.device)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    train_losses: list[float] = []
    validation_losses: list[float] = []
    history: list[dict[str, float | int]] = []
    best_auprc = float("-inf")
    best_weights = None
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        permutation = torch.randperm(len(reference_indices), generator=generator)
        total_loss = 0.0
        batches = 0
        component_totals = {name: 0.0 for name in ("text", "structure", "contrastive")}
        for start in range(0, len(permutation), config.batch_size):
            positions_cpu = permutation[start : start + config.batch_size]
            if len(positions_cpu) < 2:
                continue
            positions = positions_cpu.to(prepared.triples.device)
            optimizer.zero_grad()
            loss, components = model.training_loss(
                reference_triples[positions], confidence[positions]
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            for name in component_totals:
                component_totals[name] += float(components[name].detach())
            batches += 1
        train_loss = total_loss / max(batches, 1)
        train_losses.append(train_loss)

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
            if epoch > config.confidence_warmup_epochs:
                confidence = _adaptive_confidence(all_scores[reference_indices])
        validation_losses.append(float(validation_loss))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_losses[-1],
                "validation_auprc": validation_auprc,
                **{
                    f"train_{name}": value / max(batches, 1)
                    for name, value in component_totals.items()
                },
                "confidence_min": float(confidence.min()),
                "confidence_mean": float(confidence.mean()),
            }
        )
        print(
            f"  Epoch {epoch:03d}/{config.epochs:03d} — train loss: {train_loss:.4f} — "
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


def run(config: CCAConfig) -> pd.DataFrame:
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
        "implementation": "protocol-compatible-adaptation",
    }
    (config.output_dir / "config.json").write_text(
        json.dumps(serialized, indent=2), encoding="utf-8"
    )
    print("CCA — protocol-compatible adaptation", flush=True)
    print(f"Prepared data: {config.prepared_data_dir}", flush=True)
    print(f"Device: {device} | examples: {len(prepared.labels):,}", flush=True)
    entity_embeddings, relation_embeddings = _encode_inputs(
        config, entities, relations, device
    )
    model, train_losses, validation_losses, history = train_model(
        prepared, entity_embeddings, relation_embeddings, config
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
    config: CCAConfig, seeds: list[int], *, resume: bool = False
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
            print(f"Resuming: completed CCA seed {seed} reused", flush=True)
        seed_result.insert(0, "seed", seed)
        results.append(seed_result)
    by_seed = pd.concat(results, ignore_index=True)
    by_seed.to_csv(root_output / "summary_by_seed.csv", index=False)
    aggregate = aggregate_repeated_metrics(by_seed[by_seed["status"].eq("ok")], ["protocol"])
    aggregate.to_csv(root_output / "summary_aggregate.csv", index=False)
    return {"by_seed": by_seed, "aggregate": aggregate}
