# LUKE-Graph: A Transformer-based Approach with Gated Relational Graph Attention for Cloze-style Reading Comprehension

[![Paper](https://img.shields.io/badge/Paper-Neurocomputing-blue)](https://www.sciencedirect.com/science/article/abs/pii/S0925231223009098)
[![Based on LUKE](https://img.shields.io/badge/Based%20on-LUKE-orange)](https://github.com/studio-ousia/luke)

> **LUKE-Graph: A Transformer-based Approach with Gated Relational Graph Attention for Cloze-style Reading Comprehension**  
> Shima Foolad, Kourosh Kiani  
> Department of Electrical & Computer Engineering, Semnan University, Semnan, Iran  
> *Published in Neurocomputing*

---

## Overview

LUKE-Graph is a hybrid reading comprehension model that combines the strengths of **transformer-based entity-aware representations** (via [LUKE](https://github.com/studio-ousia/luke)) and **graph-based relation-aware reasoning** (via Gated Relational Graph Attention Networks). It is designed for cloze-style machine reading comprehension tasks, particularly those requiring commonsense and multi-hop reasoning.

Incorporating prior knowledge into pre-training models has shown promise for cloze-style reading comprehension. However, existing approaches that rely on external knowledge graphs (KGs) struggle with identifying the most relevant ambiguous entities and extracting optimal subgraphs. LUKE-Graph addresses these challenges by **constructing a heterogeneous graph directly from entity relationships within the document**, without relying on any external KG.

---

## Framework

![The framework of the LUKE-Graph method](figures/luke_graph_framework.JPG)

<p align="center">
  <em>Fig. 1: The framework of the LUKE-Graph method.</em>
</p>

The LUKE-Graph method comprises two primary components:

### 1. Transformer-based Module (Entity-aware Representations)
- Uses the pre-trained **LUKE** language model, which treats words and entities in a document as separate input tokens.
- Delivers contextualized representations via an **entity-aware self-attention mechanism**.

### 2. Graph-based Module (Relation-aware Representations)
- Constructs a **heterogeneous graph** connecting entities within a sentence and across different sentences — based on intuitive relationships in the document, without external KGs.
- Incorporates LUKE's contextualized entity representations into a **two-layer Relational Graph Attention Network (RGAT)** to resolve entity relationships before answering questions.
- Augments each RGAT layer with a **question-based gating mechanism** — called **Gated-RGAT** — which controls question information during graph convolution to emulate the human reasoning process of selecting the most suitable entity candidate.
- A linear classifier scores each candidate entity, and the highest-scoring candidate is selected as the final answer.

---

## Results

### ReCoRD Dataset (Commonsense Reasoning)

| Model | Dev F1 | Dev EM | Test F1 | Test EM |
|---|---|---|---|---|
| Human | 91.64 | 91.28 | 91.69 | 91.31 |
| BERT-Base | - | - | 56.1 | 54.0 |
| BERT-Large | 72.2 | 70.2 | 72.0 | 71.3 |
| Graph-BERT | - | - | 63.0 | 60.8 |
| SKG-BERT | 71.6 | 70.9 | 72.8 | 72.2 |
| KT-NET | 73.6 | 71.6 | 74.8 | 73.0 |
| XLNet-Verifier | 82.1 | 80.6 | 82.7 | 81.5 |
| KELM | 75.6 | 75.1 | 76.7 | 76.2 |
| RoBERTa | 89.5 | 89.0 | 90.6 | 90.0 |
| T5-Large | - | - | 86.8 | 85.9 |
| T5-11B | 93.8 | 93.2 | 94.1 | 93.4 |
| PaLM 540B | **94.0** | **94.6** | 94.2 | 93.3 |
| DeBERTa-1.5B (Ensemble) | 91.4 | 91.0 | **94.5** | **94.1** |
| LUKE | 90.96 | 90.4 | 91.2 | 90.6 |
| **LUKE-Graph (ours)** | **91.36** | **90.95** | **91.5** | **91.2** |

> All models are based on a single model except for DeBERTa.

### WikiHop Dataset (Multi-hop Reasoning)

| Model | Dev Acc | Test Acc |
|---|---|---|
| Human | - | 74.1 |
| Entity-GCN | 64.8 | 67.6 |
| BAG | 66.5 | 69.0 |
| CFC | 66.4 | 70.6 |
| HDEGraph | 68.1 | 70.9 |
| Path-GCN | 70.8 | 72.5 |
| Longformer-base | 75.0 | - |
| Longformer-large | 77.6 | 81.9 |
| ETC-large | **79.8** | 82.3 |
| RealFormer-large | 79.21 | **84.4** |
| LUKE | 73.2 | 77.1 |
| **LUKE-Graph (ours)** | **77.8** | **81.0** |

> All models are based on a single model for fair comparison.

LUKE-Graph **surpasses the LUKE state-of-the-art baseline** on both the ReCoRD and WikiHop datasets.

---

## Based On

This repository builds upon the official LUKE implementation:

> **[studio-ousia/luke](https://github.com/studio-ousia/luke)**  
> Specifically, changes were applied based on the [`examples/legacy/entity_span_qa`](https://github.com/studio-ousia/luke/blob/master/examples/legacy/entity_span_qa) component.

---

## Citation

If you use this work, please cite:

```bibtex
@article{foolad2024lukegraph,
  title     = {LUKE-Graph: A Transformer-based Approach with Gated Relational Graph Attention for Cloze-style Reading Comprehension},
  author    = {Foolad, Shima and Kiani, Kourosh},
  journal   = {Neurocomputing},
  year      = {2023},
  publisher = {Elsevier}
}
```

