"""
Official evaluation script for ReCoRD v1.0.

Computes Exact Match (EM) and F1 scores for cloze-style reading comprehension.
Predictions are compared against one or more gold answers per question; the
maximum score over all gold answers is used (following the SQuAD convention).

Can be run as a standalone script:

    python -m luke_graph.evaluation.record_eval \\
        data/record/dev.json predictions.json

Original script adopted from the SQuAD evaluation script and the official
ReCoRD evaluation release.
"""

from __future__ import annotations, print_function

import argparse
import json
import re
import string
import sys
from collections import Counter
from typing import Dict, List, Tuple


# --------------------------------------------------------------------------- #
# Text normalisation                                                            #
# --------------------------------------------------------------------------- #

def normalize_answer(text: str) -> str:
    """Lower-case, strip punctuation, articles, and extra whitespace."""

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def collapse_whitespace(s: str) -> str:
        return " ".join(s.split())

    def strip_punctuation(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    return collapse_whitespace(remove_articles(strip_punctuation(text.lower())))


# --------------------------------------------------------------------------- #
# Per-prediction metrics                                                        #
# --------------------------------------------------------------------------- #

def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between a prediction and a single reference string."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Return True if normalised prediction equals normalised ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]):
    """Apply *metric_fn* to each reference and return the maximum score."""
    return max(metric_fn(prediction, gt) for gt in ground_truths)


# --------------------------------------------------------------------------- #
# Dataset-level evaluation                                                      #
# --------------------------------------------------------------------------- #

def evaluate(
    dataset: list,
    predictions: Dict[str, str],
) -> Tuple[Dict[str, float], List[str]]:
    """
    Compute EM and F1 across the full ReCoRD dataset.

    Args:
        dataset: The ``"data"`` list from the ReCoRD JSON file.  Each entry
            contains a passage dict with a ``"qas"`` list.
        predictions: Mapping from question ID (str) to predicted answer text.

    Returns:
        metrics: Dict with ``"exact_match"`` and ``"f1"`` (both in [0, 100]).
        correct_ids: List of question IDs answered exactly correctly.
    """
    total = 0
    exact_match_sum = 0.0
    f1_sum = 0.0
    correct_ids: List[str] = []

    for passage in dataset:
        for qa in passage["qas"]:
            total += 1
            qid = qa["id"]

            if qid not in predictions:
                print(
                    f"Unanswered question {qid} will receive score 0.",
                    file=sys.stderr,
                )
                continue

            gold_answers = [ans["text"] for ans in qa["answers"]]
            prediction = predictions[qid]

            em = metric_max_over_ground_truths(exact_match_score, prediction, gold_answers)
            if int(em) == 1:
                correct_ids.append(qid)
            exact_match_sum += em
            f1_sum += metric_max_over_ground_truths(f1_score, prediction, gold_answers)

    metrics = {
        "exact_match": 100.0 * exact_match_sum / total,
        "f1": 100.0 * f1_sum / total,
    }
    print(f"* Exact Match: {metrics['exact_match']:.2f}")
    print(f"* F1:          {metrics['f1']:.2f}")
    return metrics, correct_ids


# --------------------------------------------------------------------------- #
# CLI entry point                                                               #
# --------------------------------------------------------------------------- #

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Official evaluation script for ReCoRD v1.0."
    )
    parser.add_argument("data_file", help="Dataset file in JSON format.")
    parser.add_argument("pred_file", help="Model prediction file in JSON format.")
    parser.add_argument(
        "--output-correct-ids",
        action="store_true",
        help="Write correctly answered question IDs to correct_ids.json.",
    )
    return parser


if __name__ == "__main__":
    EXPECTED_VERSION = "1.0"
    args = _build_arg_parser().parse_args()

    with open(args.data_file) as fh:
        dataset_json = json.load(fh)
        version = dataset_json.get("version", "unknown")
        if version != EXPECTED_VERSION:
            print(
                f"Warning: evaluation expects v{EXPECTED_VERSION}, "
                f"but dataset reports v{version}.",
                file=sys.stderr,
            )
        dataset = dataset_json["data"]

    with open(args.pred_file) as fh:
        predictions = json.load(fh)

    metrics, correct_ids = evaluate(dataset, predictions)

    if args.output_correct_ids:
        out_path = "correct_ids.json"
        print(f"Writing {len(correct_ids)} correctly answered IDs to {out_path}.")
        with open(out_path, "w") as fh:
            json.dump(correct_ids, fh)
