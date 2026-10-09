from __future__ import annotations

import math

import pandas as pd
import torch
from torch import nn
from torch.nn import functional
from torch_geometric.data import Data


def _mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, hidden_dim),
    )


class CCA(nn.Module):
    """Protocol-compatible adaptation of Contrastive Confidence Adaption.

    The paper's two reconstruction views and interactive contrastive objective are
    retained.  Fixed sentence embeddings replace the paper's separately pretrained
    BERT because ICEWS18 does not provide the entity descriptions required by the
    official implementation.
    """

    def __init__(
        self,
        entity_text_embeddings: torch.Tensor,
        relation_text_embeddings: torch.Tensor,
        reference_triples: torch.Tensor,
        *,
        hidden_dim: int = 128,
        contrastive_temperature: float = 0.1,
        structural_weight: float = 0.2,
        contrastive_weight: float = 1.0,
        score_batch_size: int = 256,
    ) -> None:
        super().__init__()
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")
        if contrastive_temperature <= 0:
            raise ValueError("contrastive_temperature must be positive")
        if structural_weight < 0 or contrastive_weight < 0:
            raise ValueError("score weights must be non-negative")
        if score_batch_size < 1:
            raise ValueError("score_batch_size must be at least 1")

        num_entities, text_dim = entity_text_embeddings.shape
        num_relations, relation_dim = relation_text_embeddings.shape
        if text_dim != relation_dim:
            raise ValueError("entity and relation text embeddings must have equal dimensions")
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.temperature = contrastive_temperature
        self.structural_weight = structural_weight
        self.contrastive_weight = contrastive_weight
        self.score_batch_size = score_batch_size

        self.entity_text = nn.Embedding.from_pretrained(entity_text_embeddings, freeze=True)
        self.relation_text = nn.Embedding.from_pretrained(relation_text_embeddings, freeze=True)
        self.text_entity_projection = nn.Linear(text_dim, hidden_dim)
        self.text_relation_projection = nn.Linear(text_dim, hidden_dim)
        self.text_head_decoder = _mlp(2 * hidden_dim, hidden_dim)
        self.text_tail_decoder = _mlp(2 * hidden_dim, hidden_dim)

        self.entity_structure = nn.Embedding(num_entities, hidden_dim)
        self.relation_structure = nn.Embedding(num_relations, hidden_dim)
        self.structure_head_decoder = _mlp(2 * hidden_dim, hidden_dim)
        self.structure_tail_decoder = _mlp(2 * hidden_dim, hidden_dim)
        self.text_interaction = _mlp(2 * hidden_dim, hidden_dim)
        self.structure_interaction = _mlp(2 * hidden_dim, hidden_dim)
        self.register_buffer("reference_triples", reference_triples.long())
        self._cached_scores: tuple[int, tuple[torch.Tensor, ...]] | None = None
        self.reset_parameters()

    def train(self, mode: bool = True) -> CCA:
        if mode:
            self._cached_scores = None
        return super().train(mode)

    def reset_parameters(self) -> None:
        bound = 6.0 / math.sqrt(self.hidden_dim)
        nn.init.uniform_(self.entity_structure.weight, -bound, bound)
        nn.init.uniform_(self.relation_structure.weight, -bound, bound)

    def _entity_candidates(self, *, textual: bool) -> torch.Tensor:
        if textual:
            candidates = self.text_entity_projection(self.entity_text.weight)
        else:
            candidates = self.entity_structure.weight
        return functional.normalize(candidates, dim=-1)

    def _structural_entities(self) -> torch.Tensor:
        """Aggregate the noisy, unlabeled training graph without using gold labels."""
        subjects, relations, objects = self.reference_triples.T
        entities = self.entity_structure.weight
        relation_vectors = self.relation_structure(relations)
        messages = torch.zeros_like(entities)
        counts = torch.zeros(self.num_entities, 1, device=entities.device, dtype=entities.dtype)
        messages.index_add_(0, subjects, entities[objects] - relation_vectors)
        messages.index_add_(0, objects, entities[subjects] + relation_vectors)
        ones = torch.ones(len(subjects), 1, device=entities.device, dtype=entities.dtype)
        counts.index_add_(0, subjects, ones)
        counts.index_add_(0, objects, ones)
        return entities + messages / counts.clamp_min(1.0)

    def representations(
        self, triples: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        subjects, relations, objects = triples.T
        text_entities = self.text_entity_projection(self.entity_text.weight)
        text_relations = self.text_relation_projection(self.relation_text(relations))
        text_head = self.text_head_decoder(
            torch.cat([text_relations, text_entities[objects]], dim=-1)
        )
        text_tail = self.text_tail_decoder(
            torch.cat([text_entities[subjects], text_relations], dim=-1)
        )

        structure_entities = self._structural_entities()
        structure_relations = self.relation_structure(relations)
        structure_head = self.structure_head_decoder(
            torch.cat([structure_relations, structure_entities[objects]], dim=-1)
        )
        structure_tail = self.structure_tail_decoder(
            torch.cat([structure_entities[subjects], structure_relations], dim=-1)
        )
        return text_head, text_tail, structure_head, structure_tail

    def per_triple_components(
        self, triples: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        subjects, _, objects = triples.T
        text_head, text_tail, structure_head, structure_tail = self.representations(triples)
        text_candidates = self._entity_candidates(textual=True)
        structure_candidates = self._entity_candidates(textual=False)

        text_head_logits = functional.normalize(text_head, dim=-1) @ text_candidates.T
        text_tail_logits = functional.normalize(text_tail, dim=-1) @ text_candidates.T
        structure_head_logits = (
            functional.normalize(structure_head, dim=-1) @ structure_candidates.T
        )
        structure_tail_logits = (
            functional.normalize(structure_tail, dim=-1) @ structure_candidates.T
        )
        text_loss = 0.5 * (
            functional.cross_entropy(text_head_logits, subjects, reduction="none")
            + functional.cross_entropy(text_tail_logits, objects, reduction="none")
        )
        structure_loss = 0.5 * (
            functional.cross_entropy(structure_head_logits, subjects, reduction="none")
            + functional.cross_entropy(structure_tail_logits, objects, reduction="none")
        )
        text_joint = self.text_interaction(torch.cat([text_head, text_tail], dim=-1))
        structure_joint = self.structure_interaction(
            torch.cat([structure_head, structure_tail], dim=-1)
        )
        disagreement = 1.0 - functional.cosine_similarity(
            text_joint, structure_joint, dim=-1
        )
        return text_loss, structure_loss, disagreement, text_joint, structure_joint

    def training_loss(
        self, triples: torch.Tensor, confidence: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        text_loss, structure_loss, disagreement, text_joint, structure_joint = (
            self.per_triple_components(triples)
        )
        normalized_text = functional.normalize(text_joint, dim=-1)
        normalized_structure = functional.normalize(structure_joint, dim=-1)
        logits = normalized_text @ normalized_structure.T / self.temperature
        targets = torch.arange(len(triples), device=triples.device)
        contrastive_each = 0.5 * (
            functional.cross_entropy(logits, targets, reduction="none")
            + functional.cross_entropy(logits.T, targets, reduction="none")
        )
        weights = confidence.to(text_loss.dtype).clamp_min(0.05)
        reconstruction = (
            text_loss + self.structural_weight * structure_loss
        )
        loss = (weights * reconstruction).mean() + self.contrastive_weight * (
            weights * contrastive_each
        ).mean()
        return loss, {
            "text": text_loss.mean(),
            "structure": structure_loss.mean(),
            "contrastive": contrastive_each.mean(),
            "disagreement": disagreement.mean(),
        }

    def _score_batch(self, triples: torch.Tensor) -> tuple[torch.Tensor, ...]:
        text_loss, structure_loss, disagreement, _, _ = self.per_triple_components(triples)
        text_score = -text_loss
        structure_score = -structure_loss
        combined = -(
            text_loss
            + self.structural_weight * structure_loss
            + self.contrastive_weight * disagreement
        )
        return text_score, structure_score, combined

    def scores(self, _graph: Data, triples: torch.Tensor) -> tuple[torch.Tensor, ...]:
        cache_key = triples.data_ptr()
        if not self.training and self._cached_scores is not None and self._cached_scores[0] == cache_key:
            return self._cached_scores[1]
        batches = [
            self._score_batch(triples[start : start + self.score_batch_size])
            for start in range(0, len(triples), self.score_batch_size)
        ]
        result = tuple(torch.cat(parts) for parts in zip(*batches, strict=True))
        if not self.training:
            self._cached_scores = (cache_key, tuple(score.detach() for score in result))
        return result

    def forward(
        self, graph: Data, triples: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, combined = self.scores(graph, triples)
        loss = functional.binary_cross_entropy_with_logits(combined[mask], labels[mask])
        return loss, combined

    def score_frame(
        self, graph: Data, triples: torch.Tensor, labels: torch.Tensor, descriptions: list[str]
    ) -> pd.DataFrame:
        text, structure, combined = self.scores(graph, triples)
        return pd.DataFrame(
            {
                "triple": descriptions,
                "label": labels.detach().cpu().numpy(),
                "text_score": text.detach().cpu().numpy(),
                "structure_score": structure.detach().cpu().numpy(),
                "score": combined.detach().cpu().numpy(),
            }
        )
