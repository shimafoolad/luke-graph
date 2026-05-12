"""
Training and evaluation entrypoint for LUKE-Graph on the ReCoRD dataset.

Usage
-----
    python -m luke_graph.main entity-span-qa run \\
        --data-dir data/record \\
        --output-dir outputs/record \\
        --num-train-epochs 2

See ``run --help`` for the full list of options.
"""

from __future__ import annotations

import json
import logging
import os
from argparse import Namespace
from collections import defaultdict

import numpy as np
import click
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import WEIGHTS_NAME

from luke.utils.entity_vocab import MASK_TOKEN

from ..utils import set_seed
from ..utils.trainer import Trainer, trainer_args
from .model import LukeGraphForEntitySpanQA
from .evaluation import evaluate as evaluate_on_record
from .data import (
    HIGHLIGHT_TOKEN,
    PLACEHOLDER_TOKEN,
    ENTITY_MARKER_TOKEN,
    RecordProcessor,
    convert_examples_to_features,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# CLI group                                                                     #
# --------------------------------------------------------------------------- #

@click.group(name="entity-span-qa")
def cli():
    """LUKE-Graph commands for the ReCoRD cloze-style QA task."""
    pass


# --------------------------------------------------------------------------- #
# run command                                                                   #
# --------------------------------------------------------------------------- #

@cli.command()
@click.option("--checkpoint-file", type=click.Path(exists=True), default=None,
              help="Path to a saved model checkpoint for evaluation.")
@click.option("--data-dir", default="data/record", type=click.Path(exists=True),
              show_default=True, help="Directory containing train.json and dev.json.")
@click.option("--doc-stride", default=128, show_default=True,
              help="Stride for the sliding document window.")
@click.option("--do-eval/--no-eval", default=True,
              help="Whether to run evaluation after training.")
@click.option("--do-train/--no-train", default=True,
              help="Whether to run training.")
@click.option("--eval-batch-size", default=32, show_default=True)
@click.option("--max-query-length", default=90, show_default=True,
              help="Maximum number of query tokens.")
@click.option("--max-seq-length", default=512, show_default=True,
              help="Maximum total input sequence length after tokenisation.")
@click.option("--num-train-epochs", default=2.0, show_default=True)
@click.option("--seed", default=4, show_default=True)
@click.option("--train-batch-size", default=1, show_default=True)
@trainer_args
@click.pass_obj
def run(common_args, **task_args):
    """Train and/or evaluate LUKE-Graph on ReCoRD."""
    task_args.update(common_args)
    args = Namespace(**task_args)

    set_seed(args.seed)
    args.experiment.log_parameters(
        {p.name: getattr(args, p.name) for p in run.params}
    )

    _extend_embeddings(args)

    results = {}

    if args.do_train:
        results = _train(args, results)

    if args.do_train and args.local_rank in (0, -1):
        _save_best_checkpoint(args)

    if args.local_rank not in (0, -1):
        return {}

    # Free GPU memory before evaluation
    torch.cuda.empty_cache()

    if args.do_eval:
        results = _eval(args, results)

    logger.info("Results:\n%s", json.dumps(results, indent=2, sort_keys=True))
    args.experiment.log_metrics(results)
    _write_results(args, results)

    return results


# --------------------------------------------------------------------------- #
# Embedding extension                                                           #
# --------------------------------------------------------------------------- #

def _extend_embeddings(args: Namespace) -> None:
    """
    Add three new word embeddings ([HIGHLIGHT], [PLACEHOLDER], [ENTITY]) and
    resize the entity embedding table to contain only two entries
    ([UNK] and [MASK]).

    The new word embeddings are initialised by copying the embeddings of
    the '@', '#', and '*' characters respectively, following the original
    LUKE codebase convention.
    """
    args.model_config.vocab_size += 3
    word_emb = args.model_weights["embeddings.word_embeddings.weight"]

    highlight_emb = word_emb[
        args.tokenizer.convert_tokens_to_ids(["@"])[0]
    ].unsqueeze(0)
    placeholder_emb = word_emb[
        args.tokenizer.convert_tokens_to_ids(["#"])[0]
    ].unsqueeze(0)
    marker_emb = word_emb[
        args.tokenizer.convert_tokens_to_ids(["*"])[0]
    ].unsqueeze(0)

    args.model_weights["embeddings.word_embeddings.weight"] = torch.cat(
        [word_emb, highlight_emb, placeholder_emb, marker_emb]
    )
    args.tokenizer.add_special_tokens(
        {"additional_special_tokens": [HIGHLIGHT_TOKEN, PLACEHOLDER_TOKEN, ENTITY_MARKER_TOKEN]}
    )

    # Shrink entity vocabulary to [UNK] + [MASK]
    args.model_config.entity_vocab_size = 2
    entity_emb = args.model_weights["entity_embeddings.entity_embeddings.weight"]
    mask_emb = entity_emb[args.entity_vocab[MASK_TOKEN]].unsqueeze(0)
    args.model_weights["entity_embeddings.entity_embeddings.weight"] = torch.cat(
        [entity_emb[:1], mask_emb]
    )


# --------------------------------------------------------------------------- #
# Training loop                                                                 #
# --------------------------------------------------------------------------- #

def _train(args: Namespace, results: dict) -> dict:
    """Initialise the model, run the training loop, and update results."""
    model = LukeGraphForEntitySpanQA(args)
    model.load_state_dict(args.model_weights, strict=False)
    model.to(args.device)

    train_dataloader, _, _, _ = load_examples(args, fold="train")

    steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    num_train_steps = int(steps_per_epoch * args.num_train_epochs)

    # Track the best dev checkpoint in mutable containers (closure-friendly)
    best_dev_score = [-1.0]
    best_weights = [None]

    def step_callback(model: LukeGraphForEntitySpanQA, global_step: int) -> None:
        """Evaluate on dev at the end of each epoch and keep the best weights."""
        if global_step % steps_per_epoch != 0:
            return
        if args.local_rank not in (0, -1):
            return

        epoch = int(global_step / steps_per_epoch - 1)
        dev_results = _evaluate(args, model, fold="dev")
        epoch_metrics = {f"dev_{k}_epoch{epoch}": v for k, v in dev_results.items()}
        args.experiment.log_metrics(epoch_metrics, epoch=epoch)
        results.update(epoch_metrics)
        tqdm.write(f"[epoch {epoch}] dev: {dev_results}")

        if dev_results["exact_match"] > best_dev_score[0]:
            state = (
                model.module.state_dict()
                if hasattr(model, "module")
                else model.state_dict()
            )
            best_weights[0] = {k: v.cpu().clone() for k, v in state.items()}
            best_dev_score[0] = dev_results["exact_match"]
            results["best_epoch"] = epoch

        model.train()

    trainer = Trainer(
        args,
        model=model,
        dataloader=train_dataloader,
        num_train_steps=num_train_steps,
        step_callback=step_callback,
    )
    trainer.train()

    # Stash best weights so _save_best_checkpoint can persist them
    args._best_weights = best_weights[0]
    return results


def _save_best_checkpoint(args: Namespace) -> None:
    logger.info("Saving best checkpoint to %s", args.output_dir)
    torch.save(
        args._best_weights,
        os.path.join(args.output_dir, WEIGHTS_NAME),
    )


# --------------------------------------------------------------------------- #
# Evaluation                                                                    #
# --------------------------------------------------------------------------- #

def _eval(args: Namespace, results: dict) -> dict:
    """Load a checkpoint and evaluate on the dev set."""
    model = LukeGraphForEntitySpanQA(args)

    checkpoint_path = args.checkpoint_file or os.path.join(args.output_dir, WEIGHTS_NAME)
    logger.info("Loading checkpoint from %s", checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.to(args.device)

    output_file = os.path.join(args.output_dir, "predictions.json")
    dev_metrics = _evaluate(args, model, fold="dev", output_file=output_file)
    results.update({f"dev_{k}": v for k, v in dev_metrics.items()})
    return results


def _evaluate(
    args: Namespace,
    model: LukeGraphForEntitySpanQA,
    fold: str = "dev",
    output_file: str | None = None,
) -> dict:
    """
    Run inference on *fold* and compute EM/F1 against gold answers.

    Args:
        args: Training/evaluation configuration.
        model: The LUKE-Graph model (already on the correct device).
        fold: ``"train"`` or ``"dev"``.
        output_file: If given, write the predictions JSON to this path.

    Returns:
        Dict with ``"exact_match"`` and ``"f1"`` keys.
    """
    dataloader, examples, features, processor = load_examples(args, fold)

    # Accumulate per-document predictions; each entry is (logit, entity_dict)
    doc_predictions: dict = defaultdict(list)

    model.eval()
    for batch in tqdm(dataloader, desc=f"Evaluating ({fold})"):
        inputs = {k: v.to(args.device) for k, v in batch.items() if k != "feature_indices"}
        with torch.no_grad():
            logits = model(**inputs)

        for i, feature_idx in enumerate(batch["feature_indices"]):
            feature = features[feature_idx.item()]
            max_logit, max_index = logits[i].detach().max(dim=0)
            qas_id = examples[feature.example_index].qas_id
            entity = feature.entities[max_index.item()]
            doc_predictions[qas_id].append((max_logit, entity))

    # Select the entity with the highest logit per document
    predictions = {
        qas_id: sorted(candidates, key=lambda o: o[0])[-1][1]["text"]
        for qas_id, candidates in doc_predictions.items()
    }

    if output_file:
        with open(output_file, "w") as fh:
            json.dump(predictions, fh, indent=2)
        logger.info("Predictions written to %s", output_file)

    with open(os.path.join(args.data_dir, processor.dev_file)) as fh:
        dev_data = json.load(fh)["data"]

    return evaluate_on_record(dev_data, predictions)[0]


def _write_results(args: Namespace, results: dict) -> None:
    path = os.path.join(args.output_dir, "results.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    logger.info("Results written to %s", path)


# --------------------------------------------------------------------------- #
# DataLoader construction                                                       #
# --------------------------------------------------------------------------- #

def load_examples(args: Namespace, fold: str):
    """
    Build a DataLoader for *fold* (``"train"`` or ``"dev"``).

    Returns:
        Tuple of (dataloader, examples, features, processor).
    """
    if args.local_rank not in (-1, 0) and fold == "train":
        torch.distributed.barrier()

    processor = RecordProcessor()
    if fold == "train":
        examples = processor.get_train_examples(args.data_dir)
    else:
        examples = processor.get_dev_examples(args.data_dir)

    bert_model_name = args.model_config.bert_model_name
    is_roberta = "roberta" in bert_model_name
    segment_b_id = 0 if is_roberta else 1
    add_extra_sep_token = is_roberta

    logger.info("Converting %s examples to features...", fold)
    features = convert_examples_to_features(
        examples=examples,
        tokenizer=args.tokenizer,
        max_seq_length=args.max_seq_length,
        max_mention_length=args.max_mention_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        segment_b_id=segment_b_id,
        add_extra_sep_token=add_extra_sep_token,
    )

    if args.local_rank == 0 and fold == "train":
        torch.distributed.barrier()

    dataloader = DataLoader(
        list(enumerate(features)),
        sampler=_build_sampler(args, features, fold),
        batch_size=(
            args.train_batch_size if fold == "train" else args.eval_batch_size
        ),
        collate_fn=_build_collate_fn(args, fold, segment_b_id),
    )

    return dataloader, examples, features, processor


def _build_sampler(args: Namespace, features: list, fold: str):
    if fold != "train":
        return None  # sequential by default
    return (
        DistributedSampler(features)
        if args.local_rank != -1
        else RandomSampler(features)
    )


def _build_collate_fn(args: Namespace, fold: str, segment_b_id: int):
    """Return a collate function that pads and batches InputFeatures."""

    def collate_fn(batch):
        def pad_sequence(target, padding_value):
            """Pad a named attribute or a list of tensors."""
            if isinstance(target, str):
                tensors = [
                    torch.tensor(getattr(item, target), dtype=torch.long)
                    for _, item in batch
                ]
            else:
                tensors = [torch.tensor(t, dtype=torch.long) for t in target]
            return torch.nn.utils.rnn.pad_sequence(
                tensors, batch_first=True, padding_value=padding_value
            )

        # Build entity-level sequences
        entity_ids, entity_segment_ids, entity_attention_mask, entity_position_ids = (
            [], [], [], []
        )
        edges_list, edges_type_list = [], []

        for _, item in batch:
            num_entities = len(item.entity_position_ids) + 1  # +1 for [PLACEHOLDER]
            entity_ids.append([1] * num_entities)
            entity_segment_ids.append([0] + [segment_b_id] * (num_entities - 1))
            entity_attention_mask.append([1] * num_entities)
            entity_position_ids.append(
                item.placeholder_position_ids + item.entity_position_ids
            )

            # Ensure at least two entity slots (required by the model)
            if num_entities == 1:
                entity_ids[-1].append(0)
                entity_segment_ids[-1].append(0)
                entity_attention_mask[-1].append(0)
                entity_position_ids[-1].append([-1] * args.max_mention_length)

            edges_list.append(
                torch.tensor(np.array(item.edges), dtype=torch.long)
            )
            edges_type_list.append(
                torch.tensor(item.edges_type, dtype=torch.long)
            )

        result = dict(
            word_ids=pad_sequence("word_ids", args.tokenizer.pad_token_id),
            word_attention_mask=pad_sequence("word_attention_mask", 0),
            word_segment_ids=pad_sequence("word_segment_ids", 0),
            entity_ids=pad_sequence(entity_ids, 0),
            entity_segment_ids=pad_sequence(entity_segment_ids, 0),
            entity_attention_mask=pad_sequence(entity_attention_mask, 0),
            entity_position_ids=pad_sequence(entity_position_ids, -1),
            edges=pad_sequence(edges_list, 0),
            edges_type=pad_sequence(edges_type_list, 0),
        )

        if fold == "train":
            result["labels"] = pad_sequence("labels", 0)
        else:
            result["feature_indices"] = torch.tensor(
                [idx for idx, _ in batch], dtype=torch.long
            )

        return result

    return collate_fn
