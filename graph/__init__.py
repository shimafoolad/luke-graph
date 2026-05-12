"""
Graph construction for LUKE-Graph.

Builds a typed graph over the candidate entities in a passage window,
with three edge relation types:

  - Type 0 (placeholder): bidirectional edges between every entity node
    and the [PLACEHOLDER] node (index 0 in the entity sequence).
  - Type 1 (co-sentence): edges between distinct entities that appear in
    the same sentence of the passage.
  - Type 2 (co-reference): edges between entities across different
    sentences that share the same surface text (same unique entity).

The resulting edge lists are stored in InputFeatures.edges and
InputFeatures.edges_type and are later converted to tensors in the
DataLoader collate function.
"""

from __future__ import annotations

from typing import List, Tuple

# Relation type constants
RELATION_PLACEHOLDER = 0   # entity ↔ placeholder node
RELATION_CO_SENTENCE = 1   # same sentence, different entity occurrence
RELATION_CO_REFERENCE = 2  # same entity text, different sentence


def build_entity_graph(
    entities: list,
    context_text: str,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Build a typed relational graph over candidate entities in one passage window.

    The entity node indices here correspond to their position in the ``entities``
    list (1-based, because index 0 is reserved for the [PLACEHOLDER] token in
    the model's entity sequence).

    Args:
        entities:
            List of entity dicts for the current document span, each with
            'start', 'end', and 'text' keys (character offsets into
            ``context_text``).
        context_text:
            The raw passage string, possibly containing "@highlight" markers.

    Returns:
        edges: List of (src, dst) integer index pairs.
        edges_type: Parallel list of relation-type integers (0, 1, or 2).
    """
    edges: List[Tuple[int, int]] = []
    edges_type: List[int] = []

    # Deduplicated entity surface forms — the index here is the "concept ID"
    # used to identify co-referent occurrences across sentences.
    unique_entity_texts: List[str] = []
    for ent in entities:
        if ent["text"] not in unique_entity_texts:
            unique_entity_texts.append(ent["text"])

    # ------------------------------------------------------------------ #
    # Segment the passage into sentences and @highlight clauses,           #
    # then map each entity occurrence to (node_id, concept_id) pairs.      #
    # node_id is 1-based (0 is reserved for [PLACEHOLDER]).               #
    # ------------------------------------------------------------------ #
    doc_parts = context_text.split("@highlight")
    nodes_by_segment = _collect_nodes_by_segment(
        entities, unique_entity_texts, doc_parts
    )

    # ------------------------------------------------------------------ #
    # Add edges                                                             #
    # ------------------------------------------------------------------ #
    for seg_nodes in nodes_by_segment:
        for node_id, concept_id in seg_nodes:

            # Type 0: bidirectional placeholder edges
            edges.append((0, node_id))
            edges_type.append(RELATION_PLACEHOLDER)
            edges.append((node_id, 0))
            edges_type.append(RELATION_PLACEHOLDER)

            # Type 1: co-sentence edges (within the same segment)
            for other_node_id, _ in seg_nodes:
                if node_id != other_node_id:
                    edges.append((node_id, other_node_id))
                    edges_type.append(RELATION_CO_SENTENCE)

            # Type 2: co-reference edges (same concept, different segment)
            for other_seg_nodes in nodes_by_segment:
                if other_seg_nodes is seg_nodes:
                    continue
                for other_node_id, other_concept_id in other_seg_nodes:
                    if concept_id == other_concept_id:
                        edges.append((node_id, other_node_id))
                        edges_type.append(RELATION_CO_REFERENCE)

    return edges, edges_type


def _collect_nodes_by_segment(
    entities: list,
    unique_entity_texts: List[str],
    doc_parts: List[str],
) -> List[List[Tuple[int, int]]]:
    """
    Walk through each sentence / @highlight segment of the passage and
    collect which entity occurrences (node_id, concept_id) fall within it.

    Character offsets are tracked relative to the original context string
    so that they can be compared against entity['start'] / entity['end'].

    The first doc_part (doc_parts[0]) is split on ". " to produce individual
    sentences.  Subsequent parts are @highlight clauses and are treated as
    single segments.

    Returns a list of segments, where each segment is a list of
    (node_id, concept_id) tuples.  node_id is 1-based.
    """
    nodes_by_segment: List[List[Tuple[int, int]]] = []
    node_counter = 1  # 0 is reserved for [PLACEHOLDER]
    char_cursor = 0   # running character offset in the original context

    # ---- Regular sentences (split on ". ") ----------------------------- #
    body_text = doc_parts[0]
    sentences = body_text.split(". ")
    for sent in sentences:
        if not sent:
            continue
        sent_end = char_cursor + len(sent)
        seg_nodes: List[Tuple[int, int]] = []
        for ent in entities:
            if char_cursor <= ent["end"] <= sent_end + 1:
                concept_id = unique_entity_texts.index(ent["text"])
                seg_nodes.append((node_counter, concept_id))
                node_counter += 1
        nodes_by_segment.append(seg_nodes)
        # Advance past the sentence and its ". " separator (2 chars)
        char_cursor = sent_end + 2

    # ---- @highlight clauses -------------------------------------------- #
    # After processing the body, advance past the "@highlight" marker itself.
    # "@highlight" is 10 characters.
    if body_text.endswith(". "):
        char_cursor += 10
    else:
        char_cursor += 8  # 2 chars already consumed by the loop above

    for hl_part in doc_parts[1:]:
        hl_end = char_cursor + len(hl_part)
        seg_nodes = []
        for ent in entities:
            if char_cursor <= ent["end"] < hl_end:
                concept_id = unique_entity_texts.index(ent["text"])
                seg_nodes.append((node_counter, concept_id))
                node_counter += 1
        nodes_by_segment.append(seg_nodes)
        char_cursor = hl_end + 10  # skip the next "@highlight" marker

    return nodes_by_segment
