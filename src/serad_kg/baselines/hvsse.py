from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv


def build_heterogeneous_views(
    triples: np.ndarray,
    train_mask: np.ndarray,
    *,
    neighbors_per_view: int = 2,
    seed: int = 42,
    max_degree_bucket: int = 15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build head/tail positive and negative co-occurrence views.

    Only training examples can be source neighbors. Validation and test examples
    therefore receive training context without becoming part of that context.
    """
    if triples.ndim != 2 or triples.shape[1] != 3:
        raise ValueError("triples must have shape (n, 3)")
    if neighbors_per_view < 1:
        raise ValueError("neighbors_per_view must be at least 1")
    if len(train_mask) != len(triples) or not np.asarray(train_mask).any():
        raise ValueError("train_mask must select at least one triple")
    rng = np.random.default_rng(seed)
    train_indices = np.flatnonzero(train_mask)
    by_head: dict[int, list[int]] = defaultdict(list)
    by_tail: dict[int, list[int]] = defaultdict(list)
    for index in train_indices:
        head, _, tail = triples[index]
        by_head[int(head)].append(int(index))
        by_tail[int(tail)].append(int(index))

    views = np.full((4, len(triples), neighbors_per_view), -1, dtype=np.int64)

    def sample(values: list[int], excluded: int) -> np.ndarray:
        candidates = np.asarray([value for value in values if value != excluded], dtype=np.int64)
        if not len(candidates):
            return np.empty(0, dtype=np.int64)
        return rng.choice(candidates, size=min(neighbors_per_view, len(candidates)), replace=False)

    def negative(index: int, *, head_view: bool) -> np.ndarray:
        entity = int(triples[index, 0 if head_view else 2])
        column = 0 if head_view else 2
        selected: list[int] = []
        selected_set: set[int] = set()
        max_attempts = max(20, 10 * neighbors_per_view)
        for _ in range(max_attempts):
            candidate = int(train_indices[rng.integers(len(train_indices))])
            if candidate in selected_set or int(triples[candidate, column]) == entity:
                continue
            selected.append(candidate)
            selected_set.add(candidate)
            if len(selected) == neighbors_per_view:
                break
        return np.asarray(selected[:neighbors_per_view], dtype=np.int64)

    positive_edges: list[tuple[int, int]] = []
    edge_types: list[int] = []
    for index, (head, _, tail) in enumerate(triples):
        positive_head = sample(by_head.get(int(head), []), index)
        positive_tail = sample(by_tail.get(int(tail), []), index)
        negative_head = negative(index, head_view=True)
        negative_tail = negative(index, head_view=False)
        for view, selected in enumerate(
            (positive_head, positive_tail, negative_head, negative_tail)
        ):
            views[view, index, : len(selected)] = selected
        positive_edges.extend((int(source), index) for source in positive_head)
        edge_types.extend([0] * len(positive_head))
        positive_edges.extend((int(source), index) for source in positive_tail)
        edge_types.extend([1] * len(positive_tail))

    if positive_edges:
        edge_index = torch.as_tensor(positive_edges, dtype=torch.long).T.contiguous()
        edge_type = torch.as_tensor(edge_types, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty(0, dtype=torch.long)
    head_degree = np.asarray(
        [len(by_head.get(int(head), [])) for head in triples[:, 0]], dtype=np.int64
    )
    tail_degree = np.asarray(
        [len(by_tail.get(int(tail), [])) for tail in triples[:, 2]], dtype=np.int64
    )
    head_buckets = np.minimum(np.floor(np.log2(head_degree + 1)), max_degree_bucket).astype(
        np.int64
    )
    tail_buckets = np.minimum(np.floor(np.log2(tail_degree + 1)), max_degree_bucket).astype(
        np.int64
    )
    return (
        edge_index,
        edge_type,
        torch.as_tensor(views, dtype=torch.long),
        torch.as_tensor(head_buckets, dtype=torch.long),
        torch.as_tensor(tail_buckets, dtype=torch.long),
    )


class HVSSE(nn.Module):
    """Reproduction of HVSSE's published architectural principles.

    The model uses four entity-co-occurrence views, a BiLSTM triple encoder,
    degree-aware sparse Graph-Spatial-Transformer, contrastive learning, and a
    TransE-style confidence score. No official implementation was available for
    verification when this reproduction was written.
    """

    def __init__(
        self,
        *,
        num_entities: int,
        num_relations: int,
        all_triples: torch.Tensor,
        reference_indices: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        view_neighbors: torch.Tensor,
        head_degree_buckets: torch.Tensor,
        tail_degree_buckets: torch.Tensor,
        embedding_dim: int = 128,
        num_heads: int = 4,
        max_degree_bucket: int = 15,
        contrastive_weight: float = 0.5,
        margin: float = 1.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads must be at least 1")
        if embedding_dim < 2 or embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if margin < 0 or contrastive_weight < 0:
            raise ValueError("margin and contrastive_weight must be non-negative")
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.contrastive_weight = contrastive_weight
        self.margin = margin
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.triple_encoder = nn.LSTM(
            embedding_dim,
            embedding_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.local_projection = nn.Sequential(
            nn.Linear(3 * embedding_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.head_degree_embedding = nn.Embedding(max_degree_bucket + 1, embedding_dim)
        self.tail_degree_embedding = nn.Embedding(max_degree_bucket + 1, embedding_dim)
        self.edge_type_embedding = nn.Embedding(2, embedding_dim)
        self.spatial_transformer = TransformerConv(
            embedding_dim,
            embedding_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            edge_dim=embedding_dim,
            beta=True,
        )
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("all_triples", all_triples.long())
        self.register_buffer("reference_indices", reference_indices.long())
        self.register_buffer("view_edge_index", edge_index.long())
        self.register_buffer("view_edge_type", edge_type.long())
        self.register_buffer("view_neighbors", view_neighbors.long())
        self.register_buffer("head_degree_buckets", head_degree_buckets.long())
        self.register_buffer("tail_degree_buckets", tail_degree_buckets.long())
        self._cached_scores: tuple[int, tuple[torch.Tensor, ...]] | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def train(self, mode: bool = True) -> HVSSE:
        if mode:
            self._cached_scores = None
        return super().train(mode)

    def local_representations(self, triples: torch.Tensor) -> torch.Tensor:
        subjects, relations, objects = triples.T
        sequence = torch.stack(
            [
                self.entity_embeddings(subjects),
                self.relation_embeddings(relations),
                self.entity_embeddings(objects),
            ],
            dim=1,
        )
        encoded, _ = self.triple_encoder(sequence)
        return self.local_projection(encoded.flatten(start_dim=1))

    def spatial_representations(self, triples: torch.Tensor) -> torch.Tensor:
        local = self.local_representations(triples)
        spatial_input = (
            local
            + self.head_degree_embedding(self.head_degree_buckets)
            + self.tail_degree_embedding(self.tail_degree_buckets)
        )
        edge_attributes = self.edge_type_embedding(self.view_edge_type)
        global_context = self.spatial_transformer(
            spatial_input, self.view_edge_index, edge_attributes
        )
        return self.output_norm(local + self.dropout(global_context))

    def _view_mean(self, representations: torch.Tensor, view: int) -> tuple[torch.Tensor, torch.Tensor]:
        neighbors = self.view_neighbors[view]
        valid = neighbors.ge(0)
        gathered = representations[neighbors.clamp_min(0)]
        mean = (gathered * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        available = valid.any(dim=1)
        mean = torch.where(available.unsqueeze(-1), mean, representations)
        return mean, available

    def view_similarities(
        self, representations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        similarities = []
        availability = []
        for view in range(4):
            view_representation, available = self._view_mean(representations, view)
            similarities.append(
                functional.cosine_similarity(representations, view_representation, dim=-1)
            )
            availability.append(available)
        positive = 0.5 * (similarities[0] + similarities[1])
        negative = 0.5 * (similarities[2] + similarities[3])
        positive_available = availability[0] | availability[1]
        return positive, negative, positive_available

    def translation_distance(self, triples: torch.Tensor) -> torch.Tensor:
        subjects, relations, objects = triples.T
        return torch.linalg.vector_norm(
            self.entity_embeddings(subjects)
            + self.relation_embeddings(relations)
            - self.entity_embeddings(objects),
            ord=1,
            dim=-1,
        ) / self.embedding_dim

    def training_loss(self, corrupted_triples: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        representations = self.spatial_representations(self.all_triples)
        positive_similarity, negative_similarity, available = self.view_similarities(
            representations
        )
        positive_distance = self.translation_distance(self.all_triples[self.reference_indices])
        negative_distance = self.translation_distance(corrupted_triples)
        translation_loss = functional.relu(
            self.margin + positive_distance - negative_distance
        ).mean()
        reference_available = available[self.reference_indices]
        contrastive_each = functional.softplus(
            self.margin
            - positive_similarity[self.reference_indices]
            + negative_similarity[self.reference_indices]
        )
        contrastive_loss = (
            contrastive_each[reference_available].mean()
            if bool(reference_available.any())
            else contrastive_each.new_zeros(())
        )
        loss = translation_loss + self.contrastive_weight * contrastive_loss
        return loss, {
            "translation": translation_loss,
            "contrastive": contrastive_loss,
        }

    def scores(self, _graph: Data, triples: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if len(triples) != len(self.all_triples):
            raise ValueError("HVSSE scores must be computed on the prepared manifest order")
        cache_key = triples.data_ptr()
        if not self.training and self._cached_scores is not None and self._cached_scores[0] == cache_key:
            return self._cached_scores[1]
        representations = self.spatial_representations(triples)
        positive_similarity, negative_similarity, _ = self.view_similarities(representations)
        translation = -self.translation_distance(triples)
        view_score = positive_similarity - negative_similarity
        combined = translation + self.contrastive_weight * view_score
        result = (translation, view_score, combined)
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
        translation, view_score, combined = self.scores(graph, triples)
        return pd.DataFrame(
            {
                "triple": descriptions,
                "label": labels.detach().cpu().numpy(),
                "translation_score": translation.detach().cpu().numpy(),
                "view_score": view_score.detach().cpu().numpy(),
                "score": combined.detach().cpu().numpy(),
            }
        )
