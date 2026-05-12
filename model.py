"""
LUKE-Graph model.

Extends LUKE (Language Understanding with Knowledge-base Embeddings) with a
Gated Relational Graph Convolutional Network (RGCN) applied on top of the
entity representations produced by the transformer encoder.

Architecture overview
---------------------
1. LUKE transformer encoder (entity-aware self-attention) produces hidden
   states for both word tokens and entity tokens.
2. The entity hidden states are fed into a two-layer RGCN with three edge
   types (placeholder, co-sentence, co-reference).
3. A question-aware gating mechanism fuses query context into each GCN layer,
   allowing the graph reasoning to be conditioned on the specific question.
4. A linear scorer selects the candidate entity whose representation, when
   concatenated with the [PLACEHOLDER] embedding, scores highest.

Reference
---------
Shima Foolad, Kourosh Kiani.
"LUKE-Graph: A Transformer-based Approach with Gated Relational Graph
Attention for Cloze-style Reading Comprehension."
Neurocomputing (2024).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

from luke.model import LukeEntityAwareAttentionModel


# --------------------------------------------------------------------------- #
# Gated Relational GCN                                                          #
# --------------------------------------------------------------------------- #

class GatedRGCN(nn.Module):
    """
    Two-layer Relational Graph Convolutional Network with question-aware gating.

    After each RGCN message-passing step, a gating module attends over the
    query token representations to produce a question-aware node update.  The
    gate controls how much of the question context is blended into each node's
    representation, enabling selective reasoning conditioned on the query.

    Args:
        hidden_size: Dimensionality of node features (= LUKE hidden size).
        num_relations: Number of edge relation types (default 3).
    """

    def __init__(self, hidden_size: int, num_relations: int = 3):
        super().__init__()
        self.rgcn1 = RGCNConv(hidden_size, hidden_size, num_relations)
        self.rgcn2 = RGCNConv(hidden_size, hidden_size, num_relations)

        # Query-attention weight projection  (2 * hidden → hidden)
        self.w_gate = nn.Linear(hidden_size * 2, hidden_size)
        # Blend gate  (2 * hidden → hidden)
        self.q_gate = nn.Linear(hidden_size * 2, hidden_size)

    # ---------------------------------------------------------------------- #
    # Question-aware gating                                                    #
    # ---------------------------------------------------------------------- #

    def _question_aware_gate(
        self,
        h: torch.Tensor,
        query_embs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Blend query context into each entity node representation.

        For each node i, soft-attention over all query tokens produces a
        query summary vector q_i.  A sigmoid gate then interpolates between
        the current node embedding and a tanh-transformed q_i.

        Args:
            h: Node features, shape (B*N, F).
            query_embs: Padded query token embeddings, shape (B, L, F).

        Returns:
            Updated node features, shape (B*N, F).
        """
        BN, F = h.size()
        B = query_embs.size(0)
        N = BN // B

        h = h.view(B, N, F)  # (B, N, F)

        # Compute attended query summary for each node
        q_list = []
        for node_idx in range(N):
            # Expand node embedding across the query sequence
            h_i = h[:, node_idx: node_idx + 1, :].expand_as(query_embs)  # (B, L, F)
            attention_weights = torch.sigmoid(
                self.w_gate(torch.cat([h_i, query_embs], dim=-1))
            )  # (B, L, F)
            q_i = (attention_weights * query_embs).sum(dim=1)  # (B, F)
            q_list.append(q_i)

        q = torch.stack(q_list, dim=1)  # (B, N, F)

        # Gated blend: alpha controls how much query context to inject
        alpha = torch.sigmoid(self.q_gate(torch.cat([h, q], dim=-1)))  # (B, N, F)
        h = alpha * torch.tanh(q) + (1 - alpha) * h  # (B, N, F)

        return h.view(BN, F)

    # ---------------------------------------------------------------------- #
    # Forward pass                                                             #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        query_embs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Node feature matrix, shape (B*N, hidden_size).
            edge_index: Graph connectivity, shape (2, E).
            edge_type: Edge relation types, shape (E,).
            query_embs: Padded query token embeddings, shape (B, L, hidden_size).

        Returns:
            Updated node features, shape (B*N, hidden_size).
        """
        h = self.rgcn1(x, edge_index, edge_type).relu()
        h = self._question_aware_gate(h, query_embs)
        h = self.rgcn2(h, edge_index, edge_type).relu()
        h = self._question_aware_gate(h, query_embs)
        return h


# --------------------------------------------------------------------------- #
# Full LUKE-Graph model                                                          #
# --------------------------------------------------------------------------- #

class LukeGraphForEntitySpanQA(LukeEntityAwareAttentionModel):
    """
    LUKE-Graph: LUKE encoder + Gated RGCN for cloze-style QA.

    The model scores each candidate entity span in the passage by:
      1. Encoding words and entities with the LUKE transformer.
      2. Refining entity representations with a question-conditioned RGCN.
      3. Concatenating each entity's embedding with the [PLACEHOLDER] embedding
         and projecting to a scalar score.

    During training, binary cross-entropy loss is computed against the
    ground-truth answer labels.  During inference, the entity with the
    highest logit is selected as the predicted answer.

    Args:
        args: Namespace object containing model_config and other hyperparameters.
    """

    def __init__(self, args):
        super().__init__(args.model_config)
        self.args = args

        self.graph_encoder = GatedRGCN(
            hidden_size=args.model_config.hidden_size,
            num_relations=3,
        )
        self.dropout = nn.Dropout(args.model_config.hidden_dropout_prob)
        self.scorer = nn.Linear(args.model_config.hidden_size * 2, 1)

        self.apply(self.init_weights)

    def forward(
        self,
        word_ids: torch.Tensor,
        word_segment_ids: torch.Tensor,
        word_attention_mask: torch.Tensor,
        entity_ids: torch.Tensor,
        entity_position_ids: torch.Tensor,
        entity_segment_ids: torch.Tensor,
        entity_attention_mask: torch.Tensor,
        edges: torch.Tensor,
        edges_type: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        """
        Args:
            word_ids: Token IDs, shape (B, seq_len).
            word_segment_ids: Token-type IDs, shape (B, seq_len).
            word_attention_mask: Attention mask for tokens, shape (B, seq_len).
            entity_ids: Entity type IDs, shape (B, num_entities).
            entity_position_ids: Token-level position IDs for each entity span,
                shape (B, num_entities, max_mention_length).
            entity_segment_ids: Segment IDs for entities, shape (B, num_entities).
            entity_attention_mask: Attention mask for entities, shape (B, num_entities).
            edges: Padded adjacency list, shape (B, max_edges, 2).
            edges_type: Padded edge relation types, shape (B, max_edges).
            labels: Binary answer labels, shape (B, num_entities - 1). None at inference.

        Returns:
            Training: Tuple (scalar loss,).
            Inference: Logits tensor, shape (B, num_entities - 1).
        """
        # ---- LUKE encoder ------------------------------------------------- #
        encoder_outputs = super().forward(
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask,
        )

        word_hidden_states = encoder_outputs[0]   # (B, seq_len, F)
        entity_hidden_states = encoder_outputs[1]  # (B, N, F)

        # ---- Extract query embeddings (tokens before first [SEP]) ---------- #
        # [SEP] token id is 2; there are exactly three per example.
        sep_positions = (word_ids == 2).nonzero(as_tuple=False)
        assert sep_positions.shape[0] == 3 * word_hidden_states.shape[0], (
            "Expected exactly 3 [SEP] tokens per example."
        )
        B, N, F = entity_hidden_states.shape

        # Index of the first [SEP] in each example (end of query)
        first_sep_indices = sep_positions[::3, 1]  # shape (B,)
        query_embs = [
            word_hidden_states[i, : first_sep_indices[i], :]
            for i in range(B)
        ]
        query_embs = torch.nn.utils.rnn.pad_sequence(
            query_embs, batch_first=True, padding_value=0.0
        )  # (B, max_query_len, F)

        # ---- Build batched edge_index for PyG ----------------------------- #
        # Node indices are offset by i*N so each batch item has its own subgraph.
        edge_index_list = []
        for i in range(B):
            he = edges[i].T.clone()  # (2, num_edges_i)
            he[0] += i * N
            he[1] += i * N
            edge_index_list.append(he)
        edge_index = torch.cat(edge_index_list, dim=1)  # (2, total_edges)

        # ---- Gated RGCN --------------------------------------------------- #
        entity_flat = entity_hidden_states.view(B * N, F)
        entity_flat = self.graph_encoder(
            entity_flat,
            edge_index,
            edges_type.view(-1),
            query_embs,
        )
        entity_hidden_states = entity_flat.view(B, N, F)

        # ---- Scoring ------------------------------------------------------ #
        # Index 0 = [PLACEHOLDER] token; 1: = candidate entity tokens
        placeholder_emb = entity_hidden_states[:, :1, :]   # (B, 1, F)
        doc_entity_emb = entity_hidden_states[:, 1:, :]    # (B, N-1, F)
        doc_entity_mask = entity_attention_mask[:, 1:]     # (B, N-1)

        feature_vector = torch.cat(
            [placeholder_emb.expand_as(doc_entity_emb), doc_entity_emb], dim=2
        )  # (B, N-1, 2F)
        feature_vector = self.dropout(feature_vector)
        logits = self.scorer(feature_vector)  # (B, N-1, 1)

        if labels is None:
            # Mask out padding positions with a large negative value
            return logits.squeeze(-1) + ((doc_entity_mask - 1) * 10_000).type_as(logits)

        loss = F.binary_cross_entropy_with_logits(
            logits.view(-1),
            labels.view(-1).type_as(logits),
            reduction="none",
        )
        loss = loss.masked_select(doc_entity_mask.view(-1).bool()).sum()
        loss = loss / doc_entity_mask.sum().type_as(loss)
        return (loss,)
