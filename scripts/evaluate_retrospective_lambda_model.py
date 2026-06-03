import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import torch  # noqa: E402
import train_retrospective_lambda_model as lambda_training  # noqa: E402


LEAKAGE_CONTROL_ASSUMPTION = (
    "Query outcomes must be hidden from retrieval/reranking/feature construction/model selection "
    "and reserved for post-retrieval predictive loss/evaluation/analysis."
)


def deterministic_split_indices(
    count: int,
    train_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if count < 2:
        raise ValueError("at least two examples are required for retrospective evaluation")
    if train_fraction <= 0.0 or train_fraction >= 1.0:
        raise ValueError("train_fraction must be in (0, 1)")

    indices = list(range(count))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_count = max(1, min(count - 1, int(round(count * train_fraction))))
    return sorted(indices[:train_count]), sorted(indices[train_count:])


def mean_nll(losses: list[float]) -> float:
    if not losses:
        return math.nan
    return float(sum(losses) / len(losses))


def evaluate_examples(
    examples: list[dict[str, Any]],
    train_fraction: float,
    seed: int,
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("examples must not be empty")

    train_indices, eval_indices = deterministic_split_indices(
        len(examples),
        train_fraction=train_fraction,
        seed=seed,
    )
    train_examples = [examples[index] for index in train_indices]
    eval_examples = [examples[index] for index in eval_indices]

    torch.manual_seed(seed)
    training_summary = lambda_training.train_model(
        train_examples,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
    )
    model = training_summary["model"]

    learned_losses = [
        float(
            lambda_training.learned_lambda_loss_for_example(
                model,
                example,
            )
            .detach()
            .item()
        )
        for example in eval_examples
    ]
    weak_losses = [
        float(lambda_training.weak_only_loss_for_example(example).detach().item())
        for example in eval_examples
    ]
    rule_losses = [
        float(lambda_training.rule_lambda_loss_for_example(example).detach().item())
        for example in eval_examples
    ]
    learned = mean_nll(learned_losses)
    rule = mean_nll(rule_losses)

    return {
        "example_count": len(examples),
        "train_count": len(train_examples),
        "eval_count": len(eval_examples),
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "seed": seed,
        "train_fraction": train_fraction,
        "evaluation_target": "retrospective_predictive_negative_log_likelihood",
        "outcome_usage": "held_out_query_outcomes_for_post_retrieval_predictive_evaluation_and_analysis",
        "metrics": {
            "weak_only_mean_nll": mean_nll(weak_losses),
            "rule_lambda_mean_nll": rule,
            "learned_lambda_mean_nll": learned,
            "learned_minus_rule_mean_nll": learned - rule,
        },
        "leakage_control_assumption": LEAKAGE_CONTROL_ASSUMPTION,
    }


def _load_examples_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    examples = []
    if args.examples_jsonl is not None:
        examples.extend(lambda_training.load_examples(args.examples_jsonl))
    if args.pipeline_results_jsonl is not None:
        examples.extend(
            lambda_training.load_examples_from_pipeline_results(args.pipeline_results_jsonl)
        )
    if not examples:
        raise ValueError("at least one example source is required")
    return examples


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-jsonl", type=Path, default=None)
    parser.add_argument("--pipeline-results-jsonl", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=16)
    args = parser.parse_args(argv)

    examples = _load_examples_from_args(args)
    report = evaluate_examples(
        examples,
        train_fraction=args.train_fraction,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
