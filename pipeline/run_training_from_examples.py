"""Train and evaluate the two-head lambda model directly from an examples JSONL.

The canonical orchestrator (``run_oncology_retrospective_lambda_training.py``)
enters from ``pipeline_results.jsonl`` (~2.5 GB) and rebuilds the training
examples in-process. After the endpoint-unit fix, the examples can be
regenerated once (chunked, on the machine that holds the pipeline results) and
everything downstream — training, evaluation, baselines, forward validation —
only needs the small examples file. This driver replays the orchestrator's
post-example flow faithfully:

* primary mode: deterministic 80/20 split, ``train_model_with_curves``,
  ``full_evaluation``, and the same CSV/JSON outputs ``run()`` writes;
* ``--train-end-date`` mode: past-only training for one temporal cutoff,
  exporting ``model_<cutoff>.pt`` plus the past-only examples subset — the
  inputs ``run_forward_validation.py --fold-model-dir`` expects.

Hyperparameters default to the manuscript configuration
(two_head_deepsets, epochs 100, lr 0.01, hidden 16, seed 20260603).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

_PIPELINE_DIR = Path(__file__).resolve().parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))


def _load_orchestrator() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_run_oncology_retrospective_lambda_training",
        _PIPELINE_DIR / "run_oncology_retrospective_lambda_training.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load the retrospective training orchestrator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_examples(path: Path) -> list[dict[str, Any]]:
    examples = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(json.loads(line))
    if not examples:
        raise SystemExit(f"No examples found in {path}")
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--failures-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint-key", default="ORR")
    parser.add_argument("--model-type", default="two_head_deepsets")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--simulation-iterations", type=int, default=0)
    parser.add_argument("--listwise-eta", type=float, default=0.0)
    parser.add_argument("--listwise-temperature", type=float, default=1.0)
    parser.add_argument(
        "--train-end-date",
        default=None,
        help=(
            "If set, train on past-only examples (primary completion date on or "
            "before this cutoff) and export model_<cutoff>.pt for "
            "run_forward_validation.py --fold-model-dir. Requires examples with "
            "attached true-date metadata."
        ),
    )
    args = parser.parse_args()

    orch = _load_orchestrator()
    lambda_training = orch.lambda_training
    lambda_evaluation = orch.lambda_evaluation

    examples = read_examples(args.examples_jsonl)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.train_end_date:
        train_indices, eval_indices, metadata = orch.date_based_temporal_split_indices(
            examples,
            train_end_date=args.train_end_date,
        )
        model_output = output_dir / f"model_{args.train_end_date}.pt"
        training_summary = orch.train_model_with_curves(
            examples,
            train_indices=train_indices,
            eval_indices=eval_indices,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            model_type=args.model_type,
            seed=args.seed,
            model_output=model_output,
            listwise_eta=args.listwise_eta,
            listwise_temperature=args.listwise_temperature,
        )
        past_only_path = output_dir / f"past_only_{args.train_end_date}.jsonl"
        orch.write_examples_jsonl(
            past_only_path, [examples[index] for index in train_indices]
        )
        summary = {
            "mode": "date_based_fold_training",
            "train_end_date": args.train_end_date,
            "train_count": len(train_indices),
            "eval_count": len(eval_indices),
            "split_metadata": metadata,
            "seed": args.seed,
            "epochs": args.epochs,
            "final_loss": training_summary["final_loss"],
            "final_train_nll": training_summary["train_predictive_nll_history"][-1],
            "final_eval_nll": training_summary["eval_predictive_nll_history"][-1],
            "model_output": str(model_output),
            "examples_source": str(args.examples_jsonl),
        }
        (output_dir / f"fold_training_summary_{args.train_end_date}.json").write_text(
            json.dumps(orch.json_safe(summary), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(orch.json_safe(summary), indent=2))
        return

    orch.write_examples_jsonl(output_dir / "lambda_training_examples.jsonl", examples)
    if args.failures_jsonl and args.failures_jsonl.exists():
        failures = read_examples(args.failures_jsonl)
        orch.write_csv(output_dir / "excluded_pseudo_queries.csv", failures)

    orch.write_csv(
        output_dir / "lambda_component_features.csv", orch.component_rows(examples)
    )

    model_output = output_dir / "lambda_model.pt"
    train_indices, eval_indices = lambda_evaluation.deterministic_split_indices(
        len(examples),
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    training_summary = orch.train_model_with_curves(
        examples,
        train_indices=train_indices,
        eval_indices=eval_indices,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        model_type=args.model_type,
        seed=args.seed,
        model_output=model_output,
        listwise_eta=args.listwise_eta,
        listwise_temperature=args.listwise_temperature,
    )
    serializable_training = lambda_training.serializable_training_summary(training_summary)
    (output_dir / "lambda_training_summary.json").write_text(
        json.dumps(orch.json_safe(serializable_training), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    evaluation_report = orch.full_evaluation(
        training_summary["model"],
        examples,
        train_indices=train_indices,
        eval_indices=eval_indices,
        bootstrap_iterations=args.bootstrap_iterations,
        simulation_iterations=args.simulation_iterations,
        seed=args.seed,
    )
    (output_dir / "lambda_evaluation.json").write_text(
        json.dumps(orch.json_safe(evaluation_report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    orch.write_csv(output_dir / "lambda_prediction_rows.csv", evaluation_report["prediction_rows"])
    orch.write_csv(output_dir / "lambda_nll_rows.csv", evaluation_report["nll_rows"])
    orch.write_csv(output_dir / "lambda_rate_metrics.csv", orch.rate_metric_rows(evaluation_report))
    orch.write_csv(output_dir / "lambda_calibration_bins.csv", evaluation_report["calibration_bins"])
    orch.write_csv(output_dir / "lambda_bootstrap_ci.csv", evaluation_report["bootstrap_ci"])
    orch.write_csv(
        output_dir / "lambda_disease_stratified_metrics.csv",
        evaluation_report["disease_stratified_metrics"],
    )

    headline = {
        "examples": len(examples),
        "train_count": len(train_indices),
        "eval_count": len(eval_indices),
        "seed": args.seed,
        "final_loss": training_summary["final_loss"],
        "final_train_nll": training_summary["train_predictive_nll_history"][-1],
        "final_eval_nll": training_summary["eval_predictive_nll_history"][-1],
        "model_output": str(model_output),
    }
    print(json.dumps(orch.json_safe(headline), indent=2))


if __name__ == "__main__":
    main()
