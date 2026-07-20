from __future__ import annotations

import argparse
import concurrent.futures
import csv
import html
import json
import math
import random
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "pipeline") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "pipeline"))
if str(REPO_ROOT / "pipeline") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "pipeline"))

import oncology_trial_similarity_pipeline as pipeline  # noqa: E402
import train_retrospective_lambda_model as lambda_training  # noqa: E402
import evaluate_retrospective_lambda_model as lambda_evaluation  # noqa: E402
import temporal_validation  # noqa: E402
import torch  # noqa: E402


DEFAULT_DB_ROOT = Path("/Users/wang/PHD/clinic.gov/Oncology_All_Trials/Oncology_All_Trials")


def configure_pdf_text_loading(include_pdf_text: bool) -> None:
    if include_pdf_text:
        return
    pipeline.read_pdf_excerpt = lambda path, max_chars=12000: ""


def iter_trial_jsons(db_root: Path) -> list[Path]:
    return sorted(db_root.glob("NCT*/NCT*_data.json"))


def endpoint_keys_from_json(json_path: Path) -> set[str]:
    raw = pipeline.read_json(json_path)
    extracted = pipeline.extract_trial_record_like(raw, json_path.parent.name, json_path)
    summary = pipeline.make_rule_based_summary(extracted)
    observations = pipeline.query_endpoint_observations(summary)
    eligible = set()
    for endpoint_key, observation in observations.items():
        if (
            observation.get("treatment_count") is not None
            and observation.get("treatment_denominator") is not None
        ):
            eligible.add(endpoint_key)
    return eligible


def eligible_queries(
    db_root: Path,
    endpoint_key: str,
    max_queries: int | None,
) -> tuple[list[Path], dict[str, Any]]:
    endpoint_key = endpoint_key.upper()
    rows = []
    counts = Counter()
    for json_path in iter_trial_jsons(db_root):
        try:
            keys = endpoint_keys_from_json(json_path)
        except Exception:
            counts["parse_failed"] += 1
            continue
        if keys:
            counts["has_any_supported_endpoint"] += 1
            for key in keys:
                counts[f"endpoint_{key}"] += 1
        if endpoint_key in keys:
            rows.append(json_path)
    selected = rows[:max_queries] if max_queries is not None else rows
    metadata = {
        "db_root": str(db_root),
        "endpoint_key": endpoint_key,
        "trial_json_count": len(iter_trial_jsons(db_root)),
        "eligible_query_count": len(rows),
        "selected_query_count": len(selected),
        "endpoint_counts": dict(sorted(counts.items())),
        "selection_note": (
            "Eligible pseudo-queries have a primary endpoint with treatment-arm "
            "count/denominator extractable by the current pipeline."
        ),
    }
    return selected, metadata


def ensure_hashing_index(db_root: Path, index_dir: Path) -> None:
    summaries_path = index_dir / "trial_summaries.jsonl"
    embeddings_path = index_dir / "trial_embeddings.npz"
    if summaries_path.exists() and embeddings_path.exists():
        return
    pipeline.build_index(
        db_root=db_root,
        output_dir=index_dir,
        embedding_backend="hashing",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_cached_result(result_path: Path) -> dict[str, Any] | None:
    if not result_path.exists():
        return None
    try:
        return json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        result_path.unlink()
        return None


def _generate_single_pipeline_result(payload: tuple[str, str, str, int, int, bool]) -> tuple[str, dict[str, Any]]:
    query_path_raw, index_dir_raw, result_dir_raw, top_k, rerank_top_n, include_pdf_text = payload
    configure_pdf_text_loading(include_pdf_text)
    query_path = Path(query_path_raw)
    index_dir = Path(index_dir_raw)
    result_dir = Path(result_dir_raw)
    nct_id = query_path.parent.name
    result_path = result_dir / f"{nct_id}.json"
    cached = _load_cached_result(result_path)
    if cached is not None:
        return nct_id, cached

    result = pipeline.search(
        query_json=query_path,
        index_dir=index_dir,
        top_k=top_k,
        rerank_top_n=rerank_top_n,
        embedding_backend="hashing",
        retrieval_backend="clinicalbert",
        hide_query_outcomes_for_retrieval=True,
    )
    tmp_path = result_path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(result_path)
    return nct_id, result


def generate_pipeline_results(
    query_paths: list[Path],
    index_dir: Path,
    output_dir: Path,
    top_k: int,
    rerank_top_n: int,
    include_pdf_text: bool,
    workers: int,
) -> list[dict[str, Any]]:
    result_dir = output_dir / "pseudo_query_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    payloads = [
        (
            str(query_path),
            str(index_dir),
            str(result_dir),
            top_k,
            rerank_top_n,
            include_pdf_text,
        )
        for query_path in query_paths
    ]
    results_by_nct = {}
    workers = max(1, workers)
    if workers == 1:
        for idx, payload in enumerate(payloads, start=1):
            nct_id, result = _generate_single_pipeline_result(payload)
            results_by_nct[nct_id] = result
            if idx % 10 == 0 or idx == len(payloads):
                print(f"Generated/loaded {idx}/{len(payloads)} pseudo-query results", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_generate_single_pipeline_result, payload) for payload in payloads]
            for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                nct_id, result = future.result()
                results_by_nct[nct_id] = result
                if idx % 10 == 0 or idx == len(payloads):
                    print(f"Generated/loaded {idx}/{len(payloads)} pseudo-query results", flush=True)
    return [results_by_nct[query_path.parent.name] for query_path in query_paths]


def validate_cached_pipeline_results(result_dir: Path) -> dict[str, int]:
    checked = 0
    removed = 0
    for result_path in result_dir.glob("*.json"):
        checked += 1
        if _load_cached_result(result_path) is None:
            removed += 1
    for tmp_path in result_dir.glob("*.tmp"):
        tmp_path.unlink()
        removed += 1
    return {"checked_json_files": checked, "removed_invalid_or_tmp_files": removed}


def write_examples_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    write_jsonl(path, examples)


def _list_first(value: Any, default: str = "Unknown") -> str:
    if isinstance(value, list) and value:
        return str(value[0] or default)
    if isinstance(value, str) and value:
        return value
    return default


def query_metadata_from_pipeline_result(result: dict[str, Any]) -> dict[str, Any]:
    query_summary = result.get("query_summary") or {}
    heldout = result.get("heldout_query_outcomes") or {}
    source = {**query_summary, **{key: value for key, value in heldout.items() if key == "nct_id"}}
    cancer_type = query_summary.get("cancer_type")
    if not isinstance(cancer_type, dict):
        cancer_type = {}
    nct_id = source.get("nct_id") or query_summary.get("nct_id") or heldout.get("nct_id") or ""
    return {
        "nct_id": nct_id,
        "brief_title": query_summary.get("brief_title", ""),
        "phase": query_summary.get("phase", ""),
        "status": query_summary.get("status", ""),
        "primary_site": _list_first(cancer_type.get("primary_site")),
        "histology": _list_first(cancer_type.get("histology")),
        "molecular_marker": _list_first(cancer_type.get("molecular_marker")),
        "line_of_therapy": str(cancer_type.get("line_of_therapy", "Unknown")),
        "primary_completion_date": query_summary.get("primary_completion_date"),
        "primary_completion_date_precision": query_summary.get("primary_completion_date_precision"),
        "completion_date": query_summary.get("completion_date"),
        "completion_date_precision": query_summary.get("completion_date_precision"),
        "results_first_posted_date": query_summary.get("results_first_posted_date"),
        "results_first_posted_date_precision": query_summary.get("results_first_posted_date_precision"),
        "start_date": query_summary.get("start_date"),
        "start_date_precision": query_summary.get("start_date_precision"),
        "temporal_sort_date": query_summary.get("temporal_sort_date"),
        "temporal_sort_source": query_summary.get("temporal_sort_source"),
    }


def build_examples_from_results(
    results: list[dict[str, Any]],
    endpoint_key: str,
    lambda0: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    examples = []
    failures = []
    for result in results:
        query_id = (
            result.get("heldout_query_outcomes", {}).get("nct_id")
            or result.get("query_summary", {}).get("nct_id")
            or "UNKNOWN"
        )
        try:
            example = lambda_training.build_training_example_from_pipeline_result(
                result,
                endpoint_key=endpoint_key,
                lambda0=lambda0,
                require_leakage_safe=True,
            )
        except Exception as exc:
            failures.append({"query_nct_id": query_id, "reason": str(exc)})
            continue
        if not example.get("components"):
            failures.append({"query_nct_id": query_id, "reason": "no mixture components"})
            continue
        example["query_nct_id"] = query_id
        example["query_metadata"] = query_metadata_from_pipeline_result(result)
        examples.append(example)
    return examples, failures


def component_rows(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    feature_names = lambda_training.LAMBDA_FEATURE_NAMES
    for example in examples:
        query = example["query"]
        for rank, (features, component, lambda_rule) in enumerate(
            zip(example["features"], example["components"], example["lambda_rule"]),
            start=1,
        ):
            row = {
                "query_nct_id": example.get("query_nct_id", ""),
                "candidate_rank": rank,
                "query_count": query["count"],
                "query_denominator": query["denominator"],
                "query_rate": query["count"] / query["denominator"],
                "component_alpha": component["alpha"],
                "component_beta": component["beta"],
                "component_gate": component["gate"],
                "component_denominator": component["denominator"],
                "component_discount": component["discount"],
                "lambda_rule": lambda_rule,
            }
            row.update({name: value for name, value in zip(feature_names, features)})
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value


def beta_mean(alpha: float, beta: float) -> float:
    return alpha / (alpha + beta)


def rule_lambdas(example: dict[str, Any]) -> tuple[float, list[float]]:
    lambda0 = float(example["lambda_0"])
    values = [float(value) for value in example.get("lambda_rule", [])]
    total = sum(values)
    if total <= 0.0:
        return 1.0, [0.0 for _ in values]
    return lambda0, [(1.0 - lambda0) * value / total for value in values]


def model_lambdas(model: torch.nn.Module, example: dict[str, Any]) -> tuple[float, list[float]]:
    tensors = lambda_training._validated_example_tensors(example)
    features = tensors["features"]
    scores = lambda_training._model_scores(model, features).detach()
    gates = tensors["gate"]
    lambda0 = float(tensors["lambda0"])
    positive_gate_mask = gates > 0.0
    if not positive_gate_mask.any().item():
        return 1.0, [0.0 for _ in range(len(gates))]
    log_raw = scores + gates.clamp_min(1e-12).log()
    masked_log_raw = torch.where(
        positive_gate_mask,
        log_raw,
        torch.full_like(log_raw, -torch.inf),
    )
    lambdas = (1.0 - lambda0) * torch.softmax(masked_log_raw, dim=0)
    return lambda0, [float(value) for value in lambdas]


def mixture_mean_from_lambdas(example: dict[str, Any], lambda0: float, lambdas: list[float]) -> float:
    mean = lambda0 * 0.5
    for lambda_i, component in zip(lambdas, example["components"]):
        mean += lambda_i * beta_mean(float(component["alpha"]), float(component["beta"]))
    return mean


def observed_rate(example: dict[str, Any]) -> float:
    return float(example["query"]["count"]) / float(example["query"]["denominator"])


def parse_year(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%B %d, %Y", "%B %Y", "%Y"):
        try:
            return datetime.strptime(text, fmt).year
        except ValueError:
            continue
    for token in text.replace("/", "-").split("-"):
        if len(token) == 4 and token.isdigit():
            year = int(token)
            if 1900 <= year <= 2100:
                return year
    return None


def nct_numeric_id(nct_id: str) -> int | None:
    digits = "".join(char for char in str(nct_id) if char.isdigit())
    return int(digits) if digits else None


def parse_temporal_sort_date(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            parsed = datetime.strptime(text, fmt)
            return float(parsed.year) + (parsed.timetuple().tm_yday - 1) / 366.0
        except ValueError:
            continue
    year = parse_year(text)
    return float(year) if year is not None else None


def temporal_key_for_example(example: dict[str, Any]) -> tuple[float, str]:
    metadata = example.get("query_metadata") or {}
    temporal_sort_date = parse_temporal_sort_date(metadata.get("temporal_sort_date"))
    if temporal_sort_date is not None:
        return temporal_sort_date, str(metadata.get("temporal_sort_source") or "temporal_sort_date")
    for field in ("primary_completion_date", "completion_date", "start_date"):
        year = parse_year(metadata.get(field))
        if year is not None:
            return float(year), field
    nct_value = nct_numeric_id(example.get("query_nct_id") or metadata.get("nct_id", ""))
    if nct_value is not None:
        return float(nct_value), "nct_id_numeric_proxy"
    return math.inf, "missing"


def disease_group_for_example(example: dict[str, Any]) -> str:
    metadata = example.get("query_metadata") or {}
    primary_site = str(metadata.get("primary_site") or "Unknown")
    histology = str(metadata.get("histology") or "Unknown")
    if primary_site != "Unknown":
        return primary_site
    return histology


def weak_mean(example: dict[str, Any]) -> float:
    return 0.5


def rule_mean(example: dict[str, Any]) -> float:
    lambda0, lambdas = rule_lambdas(example)
    return mixture_mean_from_lambdas(example, lambda0, lambdas)


def learned_mean(model: torch.nn.Module, example: dict[str, Any]) -> float:
    lambda0, lambdas = model_lambdas(model, example)
    return mixture_mean_from_lambdas(example, lambda0, lambdas)


def mean_loss(losses: list[float]) -> float:
    return float(sum(losses) / len(losses)) if losses else math.nan


def pure_learned_nll(model: torch.nn.Module, examples: list[dict[str, Any]]) -> float:
    return mean_loss(
        [
            float(lambda_training.learned_lambda_loss_for_example(model, example).detach().item())
            for example in examples
        ]
    )


def train_model_with_curves(
    examples: list[dict[str, Any]],
    train_indices: list[int],
    eval_indices: list[int],
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
    model_type: str,
    seed: int,
    model_output: Path | None = None,
    listwise_eta: float = 0.0,
    listwise_temperature: float = 1.0,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("examples must not be empty")
    if epochs <= 0:
        raise ValueError("epochs must be greater than 0")
    if listwise_eta < 0.0:
        raise ValueError("listwise_eta must be non-negative")
    if listwise_temperature <= 0.0:
        raise ValueError("listwise_temperature must be greater than 0")
    torch.manual_seed(seed)
    train_examples = [examples[index] for index in train_indices]
    eval_examples = [examples[index] for index in eval_indices]
    first_tensors = lambda_training._validated_example_tensors(examples[0])
    input_dim = int(first_tensors["features"].shape[1])
    model = lambda_training.create_lambda_scorer(
        model_type=model_type,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    objective_history = []
    train_nll_history = []
    eval_nll_history = []

    for _ in range(epochs):
        optimizer.zero_grad()
        losses = [
            lambda_training.predictive_loss_for_example(
                model,
                example,
                listwise_eta=listwise_eta,
                listwise_temperature=listwise_temperature,
            )
            for example in train_examples
        ]
        objective = torch.stack(losses).mean()
        objective.backward()
        optimizer.step()
        objective_history.append(float(objective.detach().item()))
        train_nll_history.append(pure_learned_nll(model, train_examples))
        eval_nll_history.append(pure_learned_nll(model, eval_examples))

    if model_output is not None:
        lambda_training.save_model_artifact(
            model_output,
            model,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            lambda0=float(first_tensors["lambda0"]),
            model_type=model_type,
        )
    summary = {
        "epochs": epochs,
        "final_loss": objective_history[-1],
        "loss_history": objective_history,
        "train_predictive_nll_history": train_nll_history,
        "eval_predictive_nll_history": eval_nll_history,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "model_type": model_type,
        "listwise_eta": float(listwise_eta),
        "listwise_temperature": float(listwise_temperature),
        "model": model,
    }
    if model_output is not None:
        summary["model_output"] = str(model_output)
    return summary


def prediction_rows(
    model: torch.nn.Module,
    examples: list[dict[str, Any]],
    indices: list[int],
    split_name: str,
) -> list[dict[str, Any]]:
    rows = []
    for index in indices:
        example = examples[index]
        metadata = example.get("query_metadata") or {}
        temporal_key, temporal_source = temporal_key_for_example(example)
        weak_pred = weak_mean(example)
        rule_pred = rule_mean(example)
        learned_pred = learned_mean(model, example)
        obs = observed_rate(example)
        rows.append(
            {
                "split": split_name,
                "example_index": index,
                "query_nct_id": example.get("query_nct_id", ""),
                "observed_rate": obs,
                "weak_predicted_rate": weak_pred,
                "rule_predicted_rate": rule_pred,
                "learned_predicted_rate": learned_pred,
                "learned_error": learned_pred - obs,
                "rule_error": rule_pred - obs,
                "weak_error": weak_pred - obs,
                "query_count": example["query"]["count"],
                "query_denominator": example["query"]["denominator"],
                "component_count": len(example["components"]),
                "primary_site": metadata.get("primary_site", "Unknown"),
                "histology": metadata.get("histology", "Unknown"),
                "disease_group": disease_group_for_example(example),
                "temporal_key": temporal_key,
                "temporal_source": temporal_source,
            }
        )
    return rows


def regression_metrics(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, float]:
    observed = [float(row["observed_rate"]) for row in rows]
    predicted = [float(row[prediction_key]) for row in rows]
    if not rows:
        return {
            "mae": math.nan,
            "rmse": math.nan,
            "correlation": math.nan,
            "mean_predicted": math.nan,
            "mean_observed": math.nan,
        }
    errors = [prediction - actual for prediction, actual in zip(predicted, observed)]
    mae = mean_loss([abs(error) for error in errors])
    rmse = math.sqrt(mean_loss([error * error for error in errors]))
    mean_pred = statistics.mean(predicted)
    mean_obs = statistics.mean(observed)
    if len(rows) > 1:
        pred_sd = statistics.stdev(predicted)
        obs_sd = statistics.stdev(observed)
        if pred_sd > 0.0 and obs_sd > 0.0:
            cov = sum((p - mean_pred) * (o - mean_obs) for p, o in zip(predicted, observed)) / (len(rows) - 1)
            corr = cov / (pred_sd * obs_sd)
        else:
            corr = math.nan
    else:
        corr = math.nan
    return {
        "mae": mae,
        "rmse": rmse,
        "correlation": corr,
        "mean_predicted": mean_pred,
        "mean_observed": mean_obs,
    }


def calibration_rows(rows: list[dict[str, Any]], bins: int = 10) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda row: float(row["learned_predicted_rate"]))
    if not sorted_rows:
        return []
    bin_count = min(bins, len(sorted_rows))
    output = []
    for bin_index in range(bin_count):
        start = round(bin_index * len(sorted_rows) / bin_count)
        end = round((bin_index + 1) * len(sorted_rows) / bin_count)
        chunk = sorted_rows[start:end]
        if not chunk:
            continue
        output.append(
            {
                "bin": bin_index + 1,
                "count": len(chunk),
                "min_predicted_rate": min(float(row["learned_predicted_rate"]) for row in chunk),
                "max_predicted_rate": max(float(row["learned_predicted_rate"]) for row in chunk),
                "mean_predicted_rate": statistics.mean(float(row["learned_predicted_rate"]) for row in chunk),
                "mean_observed_rate": statistics.mean(float(row["observed_rate"]) for row in chunk),
            }
        )
    return output


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    return lower_value + (upper_value - lower_value) * (position - lower)


def bootstrap_ci(
    rows: list[dict[str, Any]],
    nll_rows: list[dict[str, Any]],
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    count = len(rows)
    if count == 0:
        return []

    def metrics_for_indices(indices: list[int]) -> dict[str, float]:
        sampled_rows = [rows[index] for index in indices]
        sampled_nll = [nll_rows[index] for index in indices]
        weak_nll = mean_loss([float(row["weak_nll"]) for row in sampled_nll])
        rule_nll = mean_loss([float(row["rule_nll"]) for row in sampled_nll])
        learned_nll = mean_loss([float(row["learned_nll"]) for row in sampled_nll])
        learned = regression_metrics(sampled_rows, "learned_predicted_rate")
        rule = regression_metrics(sampled_rows, "rule_predicted_rate")
        return {
            "weak_mean_nll": weak_nll,
            "rule_mean_nll": rule_nll,
            "learned_mean_nll": learned_nll,
            "learned_minus_rule_nll": learned_nll - rule_nll,
            "learned_mae": learned["mae"],
            "learned_rmse": learned["rmse"],
            "learned_correlation": learned["correlation"],
            "rule_mae": rule["mae"],
            "rule_rmse": rule["rmse"],
        }

    point = metrics_for_indices(list(range(count)))
    draws: dict[str, list[float]] = {key: [] for key in point}
    for _ in range(iterations):
        indices = [rng.randrange(count) for _ in range(count)]
        values = metrics_for_indices(indices)
        for key, value in values.items():
            if math.isfinite(value):
                draws[key].append(value)
    return [
        {
            "metric": key,
            "point_estimate": value,
            "ci_lower_2_5": percentile(draws[key], 0.025),
            "ci_upper_97_5": percentile(draws[key], 0.975),
            "bootstrap_iterations": iterations,
        }
        for key, value in point.items()
    ]


def nll_metrics_from_rows(nll_rows: list[dict[str, Any]]) -> dict[str, float]:
    weak = mean_loss([float(row["weak_nll"]) for row in nll_rows])
    rule = mean_loss([float(row["rule_nll"]) for row in nll_rows])
    learned = mean_loss([float(row["learned_nll"]) for row in nll_rows])
    return {
        "weak_only_mean_nll": weak,
        "rule_lambda_mean_nll": rule,
        "learned_lambda_mean_nll": learned,
        "learned_minus_rule_mean_nll": learned - rule,
    }


def temporal_split_indices(
    examples: list[dict[str, Any]],
    train_fraction: float,
) -> tuple[list[int], list[int], dict[str, Any]]:
    return temporal_validation.fraction_split_indices(examples, train_fraction=train_fraction)


def date_based_temporal_split_indices(
    examples: list[dict[str, Any]],
    train_end_date: str,
    eval_start_date: str | None = None,
) -> tuple[list[int], list[int], dict[str, Any]]:
    return temporal_validation.date_based_split_indices(
        examples,
        train_end_date=train_end_date,
        eval_start_date=eval_start_date,
    )


def rolling_origin_temporal_splits(
    examples: list[dict[str, Any]],
    min_train_count: int,
    eval_window_size: int,
) -> list[dict[str, Any]]:
    return temporal_validation.rolling_origin_splits(
        examples,
        min_train_count=min_train_count,
        eval_window_size=eval_window_size,
    )


def stratified_metric_rows(
    prediction_rows_: list[dict[str, Any]],
    nll_rows_: list[dict[str, Any]],
    group_key: str,
    split_name: str = "eval",
    min_count: int = 1,
) -> list[dict[str, Any]]:
    nll_by_index = {int(row["example_index"]): row for row in nll_rows_}
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in prediction_rows_:
        if row.get("split") != split_name:
            continue
        groups.setdefault(str(row.get(group_key) or "Unknown"), []).append(row)
    output = []
    for group, rows in sorted(groups.items()):
        if len(rows) < min_count:
            continue
        group_nll = [nll_by_index[int(row["example_index"])] for row in rows if int(row["example_index"]) in nll_by_index]
        learned = regression_metrics(rows, "learned_predicted_rate")
        rule = regression_metrics(rows, "rule_predicted_rate")
        nll = nll_metrics_from_rows(group_nll)
        output.append(
            {
                "split": split_name,
                "group_key": group_key,
                "group": group,
                "count": len(rows),
                "learned_mean_nll": nll["learned_lambda_mean_nll"],
                "rule_mean_nll": nll["rule_lambda_mean_nll"],
                "learned_minus_rule_nll": nll["learned_minus_rule_mean_nll"],
                "learned_mae": learned["mae"],
                "learned_rmse": learned["rmse"],
                "learned_correlation": learned["correlation"],
                "rule_mae": rule["mae"],
                "rule_rmse": rule["rmse"],
            }
        )
    return output


def simulation_operating_characteristics(
    prediction_rows_: list[dict[str, Any]],
    iterations: int,
    seed: int,
    null_margin: float = 0.0,
) -> list[dict[str, Any]]:
    eval_rows = [row for row in prediction_rows_ if row.get("split") == "eval"]
    if not eval_rows or iterations <= 0:
        return []
    rng = random.Random(seed)
    scenarios = [
        ("null_weak_rate", lambda row: 0.5),
        ("observed_rate_truth", lambda row: float(row["observed_rate"])),
        ("learned_predicted_truth", lambda row: float(row["learned_predicted_rate"])),
    ]
    output = []
    for scenario, truth_fn in scenarios:
        decisions = []
        squared_errors = []
        for _ in range(iterations):
            successes = 0
            total = 0
            true_total = 0.0
            for row in eval_rows:
                n = int(float(row["query_denominator"]))
                p_true = max(0.0, min(1.0, truth_fn(row)))
                simulated_count = sum(1 for _ in range(n) if rng.random() < p_true)
                successes += simulated_count
                total += n
                true_total += p_true * n
            simulated_rate = successes / total if total else math.nan
            true_rate = true_total / total if total else math.nan
            decisions.append(1.0 if simulated_rate > 0.5 + null_margin else 0.0)
            squared_errors.append((simulated_rate - true_rate) ** 2)
        decision_rate = mean_loss(decisions)
        output.append(
            {
                "scenario": scenario,
                "iterations": iterations,
                "eval_example_count": len(eval_rows),
                "null_margin": null_margin,
                "decision_threshold_rate": 0.5 + null_margin,
                "type_i_error": decision_rate if scenario == "null_weak_rate" else math.nan,
                "power": math.nan if scenario == "null_weak_rate" else decision_rate,
                "mse": mean_loss(squared_errors),
                "note": (
                    "Lightweight binomial simulation over eval pseudo-queries. Decision is simulated pooled "
                    "ORR > threshold; use as a diagnostic, not a regulatory operating-characteristics study."
                ),
            }
        )
    return output


def nll_rows_for_examples(
    model: torch.nn.Module,
    examples: list[dict[str, Any]],
    indices: list[int],
    split_name: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for index in indices:
        example = examples[index]
        row = {
            "example_index": index,
            "query_nct_id": example.get("query_nct_id", ""),
            "weak_nll": float(lambda_training.weak_only_loss_for_example(example).detach().item()),
            "rule_nll": float(lambda_training.rule_lambda_loss_for_example(example).detach().item()),
            "learned_nll": float(lambda_training.learned_lambda_loss_for_example(model, example).detach().item()),
        }
        if split_name is not None:
            row["split"] = split_name
        rows.append(row)
    return rows


def full_evaluation(
    model: lambda_training.LambdaScorer,
    examples: list[dict[str, Any]],
    train_indices: list[int],
    eval_indices: list[int],
    bootstrap_iterations: int,
    simulation_iterations: int,
    seed: int,
) -> dict[str, Any]:
    train_rows = prediction_rows(model, examples, train_indices, "train")
    eval_rows = prediction_rows(model, examples, eval_indices, "eval")
    train_nll_rows = nll_rows_for_examples(model, examples, train_indices)
    eval_nll_rows = nll_rows_for_examples(model, examples, eval_indices)
    metrics = nll_metrics_from_rows(eval_nll_rows)
    all_prediction_rows = train_rows + eval_rows
    all_nll_rows = train_nll_rows + eval_nll_rows
    return {
        "example_count": len(examples),
        "train_count": len(train_indices),
        "eval_count": len(eval_indices),
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "seed": seed,
        "train_fraction": len(train_indices) / len(examples) if examples else math.nan,
        "evaluation_target": "retrospective_predictive_negative_log_likelihood",
        "outcome_usage": "held_out_query_outcomes_for_post_retrieval_predictive_evaluation_and_analysis",
        "metrics": metrics,
        "rate_prediction_metrics": {
            "train": {
                "weak": regression_metrics(train_rows, "weak_predicted_rate"),
                "rule": regression_metrics(train_rows, "rule_predicted_rate"),
                "learned": regression_metrics(train_rows, "learned_predicted_rate"),
            },
            "eval": {
                "weak": regression_metrics(eval_rows, "weak_predicted_rate"),
                "rule": regression_metrics(eval_rows, "rule_predicted_rate"),
                "learned": regression_metrics(eval_rows, "learned_predicted_rate"),
            },
        },
        "bootstrap_ci": bootstrap_ci(
            eval_rows,
            eval_nll_rows,
            iterations=bootstrap_iterations,
            seed=seed + 17,
        ),
        "calibration_bins": calibration_rows(eval_rows),
        "disease_stratified_metrics": stratified_metric_rows(
            all_prediction_rows,
            all_nll_rows,
            group_key="disease_group",
        ),
        "simulation_operating_characteristics": simulation_operating_characteristics(
            all_prediction_rows,
            iterations=simulation_iterations,
            seed=seed + 31,
        ),
        "prediction_rows": all_prediction_rows,
        "nll_rows": all_nll_rows,
        "leakage_control_assumption": lambda_evaluation.LEAKAGE_CONTROL_ASSUMPTION,
    }


def temporal_split_evaluation(
    examples: list[dict[str, Any]],
    train_fraction: float,
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
    model_type: str,
    seed: int,
    listwise_eta: float = 0.0,
    listwise_temperature: float = 1.0,
    split_mode: str = "fraction",
    train_end_date: str | None = None,
    eval_start_date: str | None = None,
    rolling_min_train_count: int = 10,
    rolling_eval_window_size: int = 10,
) -> dict[str, Any]:
    if split_mode == "date_based":
        if not train_end_date:
            raise ValueError("train_end_date is required for date_based temporal split")
        train_indices, eval_indices, metadata = date_based_temporal_split_indices(
            examples,
            train_end_date=train_end_date,
            eval_start_date=eval_start_date,
        )
    elif split_mode == "rolling_origin":
        splits = rolling_origin_temporal_splits(
            examples,
            min_train_count=rolling_min_train_count,
            eval_window_size=rolling_eval_window_size,
        )
        if not splits:
            raise ValueError("rolling_origin temporal split produced no eval windows")
        all_prediction_rows = []
        all_nll_rows = []
        split_summaries = []
        for split in splits:
            report = _temporal_fit_report(
                examples,
                train_indices=list(split["train_indices"]),
                eval_indices=list(split["eval_indices"]),
                metadata={**split, "split_mode": "rolling_origin"},
                epochs=epochs,
                learning_rate=learning_rate,
                hidden_dim=hidden_dim,
                model_type=model_type,
                seed=seed + int(split["split_id"]),
                listwise_eta=listwise_eta,
                listwise_temperature=listwise_temperature,
            )
            all_prediction_rows.extend(report["prediction_rows"])
            all_nll_rows.extend(report["nll_rows"])
            split_summaries.append({key: value for key, value in report.items() if key not in {"prediction_rows", "nll_rows"}})
        eval_nll_rows = [row for row in all_nll_rows if str(row.get("split", "")).startswith("temporal_eval")]
        return {
            "split_mode": "rolling_origin",
            "model_type": model_type,
            "listwise_eta": float(listwise_eta),
            "listwise_temperature": float(listwise_temperature),
            "epochs": epochs,
            "rolling_min_train_count": rolling_min_train_count,
            "rolling_eval_window_size": rolling_eval_window_size,
            "rolling_split_count": len(splits),
            "rolling_splits": split_summaries,
            "metrics": nll_metrics_from_rows(eval_nll_rows),
            "prediction_rows": all_prediction_rows,
            "nll_rows": all_nll_rows,
        }
    elif split_mode == "precomputed":
        raise ValueError("precomputed split mode is internal and requires explicit indices")
    else:
        train_indices, eval_indices, metadata = temporal_split_indices(
            examples,
            train_fraction=train_fraction,
        )
    return _temporal_fit_report(
        examples,
        train_indices=train_indices,
        eval_indices=eval_indices,
        metadata=metadata,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        model_type=model_type,
        seed=seed,
        listwise_eta=listwise_eta,
        listwise_temperature=listwise_temperature,
    )


def _temporal_fit_report(
    examples: list[dict[str, Any]],
    train_indices: list[int],
    eval_indices: list[int],
    metadata: dict[str, Any],
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
    model_type: str,
    seed: int,
    listwise_eta: float,
    listwise_temperature: float,
) -> dict[str, Any]:
    training_summary = train_model_with_curves(
        examples,
        train_indices=train_indices,
        eval_indices=eval_indices,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        model_type=model_type,
        seed=seed,
        model_output=None,
        listwise_eta=listwise_eta,
        listwise_temperature=listwise_temperature,
    )
    model = training_summary["model"]
    split_suffix = f"_{metadata['split_id']}" if metadata.get("split_id") is not None else ""
    train_split_name = f"temporal_train{split_suffix}"
    eval_split_name = f"temporal_eval{split_suffix}"
    train_rows = prediction_rows(model, examples, train_indices, train_split_name)
    eval_rows = prediction_rows(model, examples, eval_indices, eval_split_name)
    train_nll_rows = nll_rows_for_examples(model, examples, train_indices, split_name=train_split_name)
    eval_nll_rows = nll_rows_for_examples(model, examples, eval_indices, split_name=eval_split_name)
    return {
        **metadata,
        "model_type": model_type,
        "listwise_eta": float(listwise_eta),
        "listwise_temperature": float(listwise_temperature),
        "epochs": epochs,
        "final_training_loss": training_summary["final_loss"],
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "metrics": nll_metrics_from_rows(eval_nll_rows),
        "rate_prediction_metrics": {
            "train": {
                "rule": regression_metrics(train_rows, "rule_predicted_rate"),
                "learned": regression_metrics(train_rows, "learned_predicted_rate"),
            },
            "eval": {
                "rule": regression_metrics(eval_rows, "rule_predicted_rate"),
                "learned": regression_metrics(eval_rows, "learned_predicted_rate"),
            },
        },
        "prediction_rows": train_rows + eval_rows,
        "nll_rows": train_nll_rows + eval_nll_rows,
    }


def _svg_text(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _svg_document(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        f"{body}\n"
        "</svg>\n"
    )


def _write_line_svg(path: Path, title: str, xlabel: str, ylabel: str, values: list[float]) -> None:
    width, height = 760, 460
    left, right, top, bottom = 80, 30, 55, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    if not values:
        values = [0.0]
    ymin = min(values)
    ymax = max(values)
    if math.isclose(ymin, ymax):
        ymin -= 0.5
        ymax += 0.5
    x_denominator = max(1, len(values) - 1)

    points = []
    for idx, value in enumerate(values):
        x = left + plot_w * idx / x_denominator
        y = top + plot_h * (1.0 - (value - ymin) / (ymax - ymin))
        points.append(f"{x:.2f},{y:.2f}")
    body = [
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="20" font-family="Arial">{_svg_text(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<polyline points="{" ".join(points)}" fill="none" stroke="#3182bd" stroke-width="3"/>',
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-size="14" font-family="Arial">{_svg_text(xlabel)}</text>',
        f'<text transform="translate(20 {height / 2}) rotate(-90)" text-anchor="middle" font-size="14" font-family="Arial">{_svg_text(ylabel)}</text>',
        f'<text x="{left - 8}" y="{top + 5}" text-anchor="end" font-size="12" font-family="Arial">{ymax:.3f}</text>',
        f'<text x="{left - 8}" y="{top + plot_h}" text-anchor="end" font-size="12" font-family="Arial">{ymin:.3f}</text>',
    ]
    path.write_text(_svg_document(width, height, "\n".join(body)), encoding="utf-8")


def _write_multi_line_svg(
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    series: dict[str, list[float]],
) -> None:
    width, height = 760, 460
    left, right, top, bottom = 80, 145, 55, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [value for items in series.values() for value in items]
    if not values:
        values = [0.0]
    ymin = min(values)
    ymax = max(values)
    if math.isclose(ymin, ymax):
        ymin -= 0.5
        ymax += 0.5
    palette = ["#3182bd", "#31a354", "#de2d26", "#756bb1"]
    body = [
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="20" font-family="Arial">{_svg_text(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-size="14" font-family="Arial">{_svg_text(xlabel)}</text>',
        f'<text transform="translate(20 {height / 2}) rotate(-90)" text-anchor="middle" font-size="14" font-family="Arial">{_svg_text(ylabel)}</text>',
        f'<text x="{left - 8}" y="{top + 5}" text-anchor="end" font-size="12" font-family="Arial">{ymax:.3f}</text>',
        f'<text x="{left - 8}" y="{top + plot_h}" text-anchor="end" font-size="12" font-family="Arial">{ymin:.3f}</text>',
    ]
    for series_idx, (label, items) in enumerate(series.items()):
        if not items:
            continue
        x_denominator = max(1, len(items) - 1)
        points = []
        for idx, value in enumerate(items):
            x = left + plot_w * idx / x_denominator
            y = top + plot_h * (1.0 - (value - ymin) / (ymax - ymin))
            points.append(f"{x:.2f},{y:.2f}")
        color = palette[series_idx % len(palette)]
        body.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        legend_y = top + 20 + series_idx * 22
        legend_x = left + plot_w + 25
        body.extend(
            [
                f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x + 32}" y="{legend_y + 4}" font-size="13" font-family="Arial">{_svg_text(label)}</text>',
            ]
        )
    path.write_text(_svg_document(width, height, "\n".join(body)), encoding="utf-8")


def _write_bar_svg(path: Path, title: str, ylabel: str, labels: list[str], values: list[float]) -> None:
    width, height = 760, 460
    left, right, top, bottom = 80, 40, 55, 95
    plot_w = width - left - right
    plot_h = height - top - bottom
    ymax = max(values) if values else 1.0
    if ymax <= 0:
        ymax = 1.0
    colors = ["#b8b8b8", "#6baed6", "#31a354", "#756bb1"]
    body = [
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="20" font-family="Arial">{_svg_text(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<text transform="translate(20 {height / 2}) rotate(-90)" text-anchor="middle" font-size="14" font-family="Arial">{_svg_text(ylabel)}</text>',
        f'<text x="{left - 8}" y="{top + 5}" text-anchor="end" font-size="12" font-family="Arial">{ymax:.3f}</text>',
    ]
    bar_slot = plot_w / max(1, len(values))
    bar_w = bar_slot * 0.62
    for idx, (label, value) in enumerate(zip(labels, values)):
        bar_h = plot_h * max(0.0, value) / ymax
        x = left + idx * bar_slot + (bar_slot - bar_w) / 2
        y = top + plot_h - bar_h
        body.extend(
            [
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="{colors[idx % len(colors)]}"/>',
                f'<text x="{x + bar_w / 2:.2f}" y="{y - 6:.2f}" text-anchor="middle" font-size="12" font-family="Arial">{value:.3f}</text>',
                f'<text x="{x + bar_w / 2:.2f}" y="{top + plot_h + 24}" text-anchor="middle" font-size="12" font-family="Arial">{_svg_text(label)}</text>',
            ]
        )
    path.write_text(_svg_document(width, height, "\n".join(body)), encoding="utf-8")


def _write_scatter_svg(
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    points: list[tuple[float, float]],
) -> None:
    width, height = 760, 520
    left, right, top, bottom = 80, 40, 55, 75
    plot_w = width - left - right
    plot_h = height - top - bottom
    body = [
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="20" font-family="Arial">{_svg_text(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top}" stroke="#999" stroke-dasharray="5,5"/>',
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-size="14" font-family="Arial">{_svg_text(xlabel)}</text>',
        f'<text transform="translate(20 {height / 2}) rotate(-90)" text-anchor="middle" font-size="14" font-family="Arial">{_svg_text(ylabel)}</text>',
    ]
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = left + plot_w * tick
        y = top + plot_h * (1.0 - tick)
        body.extend(
            [
                f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 5}" stroke="#333"/>',
                f'<text x="{x:.2f}" y="{top + plot_h + 20}" text-anchor="middle" font-size="11" font-family="Arial">{tick:.2f}</text>',
                f'<line x1="{left - 5}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#333"/>',
                f'<text x="{left - 9}" y="{y + 4:.2f}" text-anchor="end" font-size="11" font-family="Arial">{tick:.2f}</text>',
            ]
        )
    for x_value, y_value in points:
        x = left + plot_w * max(0.0, min(1.0, x_value))
        y = top + plot_h * (1.0 - max(0.0, min(1.0, y_value)))
        body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" fill="#3182bd" opacity="0.72"/>')
    path.write_text(_svg_document(width, height, "\n".join(body)), encoding="utf-8")


def _write_calibration_svg(
    path: Path,
    title: str,
    rows: list[dict[str, Any]],
) -> None:
    points = [
        (float(row["mean_predicted_rate"]), float(row["mean_observed_rate"]))
        for row in rows
        if row.get("count", 0)
    ]
    _write_scatter_svg(
        path,
        title,
        "Mean predicted ORR in bin",
        "Mean observed ORR in bin",
        points,
    )


def _histogram(values: list[float], bins: int) -> tuple[list[int], list[float]]:
    if not values:
        return [0], [0.0, 1.0]
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        low -= 0.5
        high += 0.5
    bins = max(1, bins)
    width = (high - low) / bins
    counts = [0 for _ in range(bins)]
    for value in values:
        idx = min(bins - 1, max(0, int((value - low) / width)))
        counts[idx] += 1
    edges = [low + width * idx for idx in range(bins + 1)]
    return counts, edges


def _write_hist_svg(path: Path, title: str, xlabel: str, values: list[float], bins: int, color: str) -> None:
    width, height = 760, 460
    left, right, top, bottom = 80, 40, 55, 85
    plot_w = width - left - right
    plot_h = height - top - bottom
    counts, edges = _histogram(values, bins)
    ymax = max(counts) or 1
    body = [
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="20" font-family="Arial">{_svg_text(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-size="14" font-family="Arial">{_svg_text(xlabel)}</text>',
        f'<text x="{left - 8}" y="{top + 5}" text-anchor="end" font-size="12" font-family="Arial">{ymax}</text>',
        f'<text x="{left}" y="{top + plot_h + 24}" text-anchor="middle" font-size="12" font-family="Arial">{edges[0]:.3f}</text>',
        f'<text x="{left + plot_w}" y="{top + plot_h + 24}" text-anchor="middle" font-size="12" font-family="Arial">{edges[-1]:.3f}</text>',
    ]
    bar_slot = plot_w / len(counts)
    for idx, count in enumerate(counts):
        bar_h = plot_h * count / ymax
        x = left + idx * bar_slot + 1
        y = top + plot_h - bar_h
        body.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_slot - 2:.2f}" height="{bar_h:.2f}" fill="{color}"/>'
        )
    path.write_text(_svg_document(width, height, "\n".join(body)), encoding="utf-8")


def save_plots(
    output_dir: Path,
    training_summary: dict[str, Any],
    evaluation_report: dict[str, Any],
    examples: list[dict[str, Any]],
    components: list[dict[str, Any]],
) -> list[Path]:
    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    loss_path = plot_dir / "training_objective_curve.svg"
    _write_line_svg(
        loss_path,
        "Retrospective lambda training objective",
        "Epoch",
        "Mean objective loss",
        [float(value) for value in training_summary["loss_history"]],
    )
    paths.append(loss_path)

    nll_curve_path = plot_dir / "train_eval_nll_curve.svg"
    _write_multi_line_svg(
        nll_curve_path,
        "Train vs eval learned predictive NLL",
        "Epoch",
        "Mean predictive NLL",
        {
            "train NLL": [float(value) for value in training_summary["train_predictive_nll_history"]],
            "eval NLL": [float(value) for value in training_summary["eval_predictive_nll_history"]],
        },
    )
    paths.append(nll_curve_path)

    metrics = evaluation_report["metrics"]
    nll_path = plot_dir / "evaluation_nll_comparison.svg"
    labels = ["Weak only", "Rule lambda", "Learned lambda"]
    values = [
        metrics["weak_only_mean_nll"],
        metrics["rule_lambda_mean_nll"],
        metrics["learned_lambda_mean_nll"],
    ]
    _write_bar_svg(
        nll_path,
        "Held-out retrospective predictive NLL",
        "Mean predictive NLL",
        labels,
        [float(value) for value in values],
    )
    paths.append(nll_path)

    eval_predictions = [
        row for row in evaluation_report.get("prediction_rows", [])
        if row.get("split") == "eval"
    ]
    scatter_path = plot_dir / "predicted_vs_observed_orr_scatter.svg"
    _write_scatter_svg(
        scatter_path,
        "Predicted vs observed ORR on eval split",
        "Learned predicted ORR",
        "Observed held-out ORR",
        [
            (float(row["learned_predicted_rate"]), float(row["observed_rate"]))
            for row in eval_predictions
        ],
    )
    paths.append(scatter_path)

    calibration_path = plot_dir / "calibration_plot.svg"
    _write_calibration_svg(
        calibration_path,
        "Calibration: predicted ORR vs observed ORR",
        evaluation_report.get("calibration_bins", []),
    )
    paths.append(calibration_path)

    query_rates = [example["query"]["count"] / example["query"]["denominator"] for example in examples]
    rate_path = plot_dir / "query_response_rate_distribution.svg"
    _write_hist_svg(
        rate_path,
        "Distribution of held-out query outcomes",
        "Held-out query response rate",
        query_rates,
        min(15, max(5, len(query_rates) // 2)),
        "#756bb1",
    )
    paths.append(rate_path)

    lambda_path = plot_dir / "rule_lambda_distribution.svg"
    lambda_values = [float(row["lambda_rule"]) for row in components]
    _write_hist_svg(
        lambda_path,
        "Rule lambda distribution across mixture components",
        "Rule lambda",
        lambda_values,
        20,
        "#fd8d3c",
    )
    paths.append(lambda_path)

    return paths


def rate_metric_rows(evaluation_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for split, models in evaluation_report.get("rate_prediction_metrics", {}).items():
        for model_name, metrics in models.items():
            rows.append(
                {
                    "split": split,
                    "model": model_name,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "correlation": metrics["correlation"],
                    "mean_predicted": metrics["mean_predicted"],
                    "mean_observed": metrics["mean_observed"],
                }
            )
    return rows


def summarize_examples(examples: list[dict[str, Any]], components: list[dict[str, Any]]) -> dict[str, Any]:
    query_rates = [example["query"]["count"] / example["query"]["denominator"] for example in examples]
    component_counts = [len(example["components"]) for example in examples]
    lambda_values = [float(row["lambda_rule"]) for row in components]
    denominators = [float(row["component_denominator"]) for row in components]
    return {
        "training_example_count": len(examples),
        "component_count": len(components),
        "mean_components_per_query": statistics.mean(component_counts) if component_counts else 0.0,
        "median_components_per_query": statistics.median(component_counts) if component_counts else 0.0,
        "mean_query_response_rate": statistics.mean(query_rates) if query_rates else 0.0,
        "median_query_response_rate": statistics.median(query_rates) if query_rates else 0.0,
        "mean_rule_lambda": statistics.mean(lambda_values) if lambda_values else 0.0,
        "median_rule_lambda": statistics.median(lambda_values) if lambda_values else 0.0,
        "mean_component_denominator": statistics.mean(denominators) if denominators else 0.0,
        "median_component_denominator": statistics.median(denominators) if denominators else 0.0,
    }


def markdown_report(
    output_dir: Path,
    metadata: dict[str, Any],
    example_summary: dict[str, Any],
    training_summary: dict[str, Any],
    evaluation_report: dict[str, Any],
    failures: list[dict[str, Any]],
    figure_paths: list[Path],
) -> str:
    metrics = evaluation_report["metrics"]
    rate_rows = rate_metric_rows(evaluation_report)
    bootstrap_rows = evaluation_report.get("bootstrap_ci", [])
    calibration = evaluation_report.get("calibration_bins", [])
    temporal = evaluation_report.get("temporal_split_evaluation", {})
    disease_rows = evaluation_report.get("disease_stratified_metrics", [])
    simulation_rows = evaluation_report.get("simulation_operating_characteristics", [])
    lines = [
        "# Retrospective Lambda Training Results",
        "",
        "## Data And Cohort",
        "",
        f"- Source data: `{metadata['db_root']}`",
        f"- Text source: `{metadata.get('text_source_note', 'not recorded')}`",
        f"- Trial JSON files scanned: {metadata['trial_json_count']}",
        f"- Endpoint key: `{metadata['endpoint_key']}`",
        f"- Eligible pseudo-query count: {metadata['eligible_query_count']}",
        f"- Selected pseudo-query count: {metadata['selected_query_count']}",
        f"- Training examples with at least one mixture component: {example_summary['training_example_count']}",
        f"- Pseudo-query failures after retrieval/component construction: {len(failures)}",
        "",
        "## Training Summary",
        "",
        f"- Epochs: {training_summary['epochs']}",
        f"- Hidden dimension: {training_summary['hidden_dim']}",
        f"- Model type: `{training_summary.get('model_type', 'not recorded')}`",
        f"- Listwise allocation eta: {float(training_summary.get('listwise_eta', 0.0)):.6f}",
        f"- Listwise allocation temperature: {float(training_summary.get('listwise_temperature', 1.0)):.6f}",
        f"- Final training loss: {training_summary['final_loss']:.6f}",
        f"- Model artifact: `{training_summary.get('model_output', '')}`",
        f"- Train examples: {evaluation_report['train_count']}",
        f"- Eval examples: {evaluation_report['eval_count']}",
        "",
        "## Evaluation Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| weak_only_mean_nll | {metrics['weak_only_mean_nll']:.6f} |",
        f"| rule_lambda_mean_nll | {metrics['rule_lambda_mean_nll']:.6f} |",
        f"| learned_lambda_mean_nll | {metrics['learned_lambda_mean_nll']:.6f} |",
        f"| learned_minus_rule_mean_nll | {metrics['learned_minus_rule_mean_nll']:.6f} |",
        "",
        "Lower NLL is better. A negative `learned_minus_rule_mean_nll` means the learned lambda model beat the rule lambda baseline on the held-out split.",
        "",
        "## ORR Prediction Metrics",
        "",
        "These metrics compare the predicted posterior mean ORR with the observed held-out ORR. Lower MAE/RMSE is better; correlation closer to 1 means the ordering of high-vs-low ORR trials is better aligned.",
        "",
        "| Split | Model | MAE | RMSE | Correlation | Mean predicted | Mean observed |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rate_rows:
        def fmt(value: Any) -> str:
            return "NA" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{float(value):.6f}"
        lines.append(
            f"| {row['split']} | {row['model']} | {fmt(row['mae'])} | {fmt(row['rmse'])} | {fmt(row['correlation'])} | {fmt(row['mean_predicted'])} | {fmt(row['mean_observed'])} |"
        )
    lines.extend(
        [
            "",
            "## Bootstrap Confidence Intervals",
            "",
            "Bootstrap CIs resample the held-out eval pseudo-queries with replacement. They show how stable each metric is under plausible resampling of the eval set.",
            "",
            "| Metric | Point estimate | 2.5% | 97.5% | Iterations |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in bootstrap_rows:
        def fmt_ci(value: Any) -> str:
            return "NA" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{float(value):.6f}"
        lines.append(
            f"| {row['metric']} | {fmt_ci(row['point_estimate'])} | {fmt_ci(row['ci_lower_2_5'])} | {fmt_ci(row['ci_upper_97_5'])} | {row['bootstrap_iterations']} |"
        )
    lines.extend(
        [
            "",
            "## Calibration Bins",
            "",
            "Eval predictions are sorted by learned predicted ORR and split into bins. A well-calibrated model has mean predicted ORR close to mean observed ORR in each bin.",
            "",
            "| Bin | Count | Mean predicted ORR | Mean observed ORR |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in calibration:
        lines.append(
            f"| {row['bin']} | {row['count']} | {float(row['mean_predicted_rate']):.6f} | {float(row['mean_observed_rate']):.6f} |"
        )
    if temporal:
        temporal_metrics = temporal.get("metrics", {})
        lines.extend(
            [
                "",
                "## Temporal Split Evaluation",
                "",
                "Temporal split sorts pseudo-queries by completion/start date when available, otherwise by NCT numeric registry proxy.",
                "",
                "| Field | Value |",
                "|---|---:|",
                f"| temporal_train_count | {temporal.get('train_count')} |",
                f"| temporal_eval_count | {temporal.get('eval_count')} |",
                f"| temporal_learned_mean_nll | {float(temporal_metrics.get('learned_lambda_mean_nll', math.nan)):.6f} |",
                f"| temporal_rule_mean_nll | {float(temporal_metrics.get('rule_lambda_mean_nll', math.nan)):.6f} |",
                f"| temporal_learned_minus_rule_nll | {float(temporal_metrics.get('learned_minus_rule_mean_nll', math.nan)):.6f} |",
            ]
        )
    if disease_rows:
        lines.extend(
            [
                "",
                "## Disease-Stratified Eval Metrics",
                "",
                "| Disease group | Count | Learned NLL | Rule NLL | Learned - Rule NLL | Learned RMSE |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in disease_rows[:20]:
            lines.append(
                f"| {row['group']} | {row['count']} | {float(row['learned_mean_nll']):.6f} | {float(row['rule_mean_nll']):.6f} | {float(row['learned_minus_rule_nll']):.6f} | {float(row['learned_rmse']):.6f} |"
            )
    if simulation_rows:
        lines.extend(
            [
                "",
                "## Simulation-Based Operating Characteristics",
                "",
                "These are lightweight diagnostic simulations over held-out pseudo-queries, not a definitive regulatory operating-characteristics study.",
                "",
                "| Scenario | Type I error | Power | MSE | Iterations |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in simulation_rows:
            def fmt_sim(value: Any) -> str:
                return "NA" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{float(value):.6f}"
            lines.append(
                f"| {row['scenario']} | {fmt_sim(row.get('type_i_error'))} | {fmt_sim(row.get('power'))} | {fmt_sim(row.get('mse'))} | {row['iterations']} |"
            )
    lines.extend(
        [
        "",
        "## Cohort And Component Summary",
        "",
        "| Field | Value |",
        "|---|---:|",
        ]
    )
    for key, value in example_summary.items():
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.6f} |")
        else:
            lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Figures", ""])
    for path in figure_paths:
        lines.append(f"![{path.stem}](<{path.resolve()}>)")
    lines.extend(
        [
            "",
            "## Leakage Control",
            "",
            evaluation_report["leakage_control_assumption"],
            "",
            "Each pipeline-result row was loaded through `require_leakage_safe=True`, so rows without `retrospective_leakage_control.query_outcomes_hidden_from_retrieval=true` and a dictionary `heldout_query_outcomes` were rejected.",
            "",
            "## Caveats",
            "",
            "- This is retrospective predictive calibration, not expert validation.",
            "- The recommended optimized training configuration is `--model-type two_head_deepsets`; `--listwise-eta` is available for sensitivity experiments, but pure held-out NLL remains the evaluation metric.",
            "- Hashing retrieval was used for local full-data throughput; ClinicalBERT, Trial2Vec, or SECRET-style retrieval can be substituted if runtime allows.",
            "- Only pseudo-queries with extractable primary endpoint treatment-arm count/denominator for the selected endpoint enter training.",
        ]
    )
    if failures:
        lines.extend(["", "## First 20 Excluded Pseudo-Queries", "", "| Query | Reason |", "|---|---|"])
        for failure in failures[:20]:
            lines.append(f"| {failure['query_nct_id']} | {failure['reason']} |")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    configure_pdf_text_loading(args.include_pdf_text)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    index_dir = args.index_dir or (output_dir / "hashing_index")

    if args.pipeline_results_jsonl is not None:
        results = read_jsonl(args.pipeline_results_jsonl)
        if not results:
            raise RuntimeError("No pipeline results were loaded")
        metadata = {
            "db_root": str(args.db_root),
            "endpoint_key": args.endpoint_key.upper(),
            "trial_json_count": None,
            "eligible_query_count": len(results),
            "selected_query_count": len(results),
            "endpoint_counts": {},
            "selection_note": (
                "Pseudo-query results were loaded from an existing pipeline_results.jsonl file."
            ),
            "pipeline_results_source": str(args.pipeline_results_jsonl),
        }
        selected_rows = [
            {
                "nct_id": (
                    result.get("heldout_query_outcomes", {}).get("nct_id")
                    or result.get("query_summary", {}).get("nct_id")
                    or ""
                ),
                "json_path": "",
            }
            for result in results
        ]
    else:
        ensure_hashing_index(args.db_root, index_dir)
        query_paths, metadata = eligible_queries(args.db_root, args.endpoint_key, args.max_queries)
        if not query_paths:
            raise RuntimeError("No eligible pseudo-queries were found")
        selected_rows = [{"nct_id": path.parent.name, "json_path": str(path)} for path in query_paths]
    run_metadata = {
        **metadata,
        "include_pdf_text": args.include_pdf_text,
        "pipeline_result_workers": args.workers,
        "model_type": args.model_type,
        "listwise_eta": args.listwise_eta,
        "listwise_temperature": args.listwise_temperature,
        "temporal_split_mode": args.temporal_split_mode,
        "temporal_train_end_date": args.temporal_train_end_date,
        "temporal_eval_start_date": args.temporal_eval_start_date,
        "rolling_min_train_count": args.rolling_min_train_count,
        "rolling_eval_window_size": args.rolling_eval_window_size,
        "date_metadata_csv": str(args.date_metadata_csv) if args.date_metadata_csv else None,
        "text_source_note": (
            "JSON structured fields plus PDF excerpts"
            if args.include_pdf_text
            else "JSON structured fields only; protocol/SAP PDF text extraction disabled for throughput"
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    write_csv(output_dir / "selected_pseudo_queries.csv", selected_rows)

    pipeline_results_path = output_dir / "pipeline_results.jsonl"
    if args.pipeline_results_jsonl is None:
        results = generate_pipeline_results(
            query_paths=query_paths,
            index_dir=index_dir,
            output_dir=output_dir,
            top_k=args.top_k,
            rerank_top_n=args.rerank_top_n,
            include_pdf_text=args.include_pdf_text,
            workers=args.workers,
        )
    write_jsonl(pipeline_results_path, results)

    examples, failures = build_examples_from_results(
        results,
        endpoint_key=args.endpoint_key,
        lambda0=args.lambda0,
    )
    if not examples:
        raise RuntimeError("No training examples with mixture components were created")
    if args.date_metadata_csv is not None:
        date_metadata = temporal_validation.read_date_metadata_csv(args.date_metadata_csv)
        date_attachment_report = temporal_validation.attach_date_metadata_to_examples(
            examples,
            date_metadata,
        )
        (output_dir / "date_metadata_attachment.json").write_text(
            json.dumps(json_safe(date_attachment_report), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    examples_path = output_dir / "lambda_training_examples.jsonl"
    write_examples_jsonl(examples_path, examples)
    write_csv(output_dir / "excluded_pseudo_queries.csv", failures)

    components = component_rows(examples)
    write_csv(output_dir / "lambda_component_features.csv", components)

    model_output = output_dir / "lambda_model.pt"
    train_indices, eval_indices = lambda_evaluation.deterministic_split_indices(
        len(examples),
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    training_summary = train_model_with_curves(
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
    training_summary_path = output_dir / "lambda_training_summary.json"
    training_summary_path.write_text(
        json.dumps(json_safe(serializable_training), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    evaluation_report = full_evaluation(
        training_summary["model"],
        examples,
        train_indices=train_indices,
        eval_indices=eval_indices,
        bootstrap_iterations=args.bootstrap_iterations,
        simulation_iterations=args.simulation_iterations,
        seed=args.seed,
    )
    temporal_report = temporal_split_evaluation(
        examples,
        train_fraction=args.train_fraction,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        model_type=args.model_type,
        seed=args.seed + 101,
        listwise_eta=args.listwise_eta,
        listwise_temperature=args.listwise_temperature,
        split_mode=args.temporal_split_mode,
        train_end_date=args.temporal_train_end_date,
        eval_start_date=args.temporal_eval_start_date,
        rolling_min_train_count=args.rolling_min_train_count,
        rolling_eval_window_size=args.rolling_eval_window_size,
    )
    evaluation_report["temporal_split_evaluation"] = {
        key: value
        for key, value in temporal_report.items()
        if key not in {"prediction_rows", "nll_rows"}
    }
    evaluation_path = output_dir / "lambda_evaluation.json"
    evaluation_path.write_text(
        json.dumps(json_safe(evaluation_report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    prediction_rows_path = output_dir / "lambda_prediction_rows.csv"
    write_csv(prediction_rows_path, evaluation_report["prediction_rows"])
    nll_rows_path = output_dir / "lambda_nll_rows.csv"
    write_csv(nll_rows_path, evaluation_report["nll_rows"])
    rate_metrics_path = output_dir / "lambda_rate_metrics.csv"
    write_csv(rate_metrics_path, rate_metric_rows(evaluation_report))
    calibration_bins_path = output_dir / "lambda_calibration_bins.csv"
    write_csv(calibration_bins_path, evaluation_report["calibration_bins"])
    bootstrap_ci_path = output_dir / "lambda_bootstrap_ci.csv"
    write_csv(bootstrap_ci_path, evaluation_report["bootstrap_ci"])
    disease_metrics_path = output_dir / "lambda_disease_stratified_metrics.csv"
    write_csv(disease_metrics_path, evaluation_report["disease_stratified_metrics"])
    simulation_path = output_dir / "lambda_simulation_operating_characteristics.csv"
    write_csv(simulation_path, evaluation_report["simulation_operating_characteristics"])
    temporal_summary_path = output_dir / "lambda_temporal_split_evaluation.json"
    temporal_summary_path.write_text(
        json.dumps(
            json_safe({key: value for key, value in temporal_report.items() if key not in {"prediction_rows", "nll_rows"}}),
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporal_predictions_path = output_dir / "lambda_temporal_prediction_rows.csv"
    write_csv(temporal_predictions_path, temporal_report["prediction_rows"])
    temporal_nll_path = output_dir / "lambda_temporal_nll_rows.csv"
    write_csv(temporal_nll_path, temporal_report["nll_rows"])

    example_summary = summarize_examples(examples, components)
    summary_path = output_dir / "lambda_dataset_summary.json"
    summary_path.write_text(
        json.dumps(json_safe(example_summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    figure_paths = save_plots(
        output_dir,
        serializable_training,
        evaluation_report,
        examples,
        components,
    )
    report = markdown_report(
        output_dir,
        run_metadata,
        example_summary,
        serializable_training,
        evaluation_report,
        failures,
        figure_paths,
    )
    report_path = output_dir / "retrospective_lambda_training_results.md"
    report_path.write_text(report, encoding="utf-8")

    print(json.dumps(
        {
            "output_dir": str(output_dir),
            "index_dir": str(index_dir),
            "pipeline_results_jsonl": str(pipeline_results_path),
            "examples_jsonl": str(examples_path),
            "training_summary": str(training_summary_path),
            "evaluation": str(evaluation_path),
            "dataset_summary": str(summary_path),
            "component_features_csv": str(output_dir / "lambda_component_features.csv"),
            "prediction_rows_csv": str(prediction_rows_path),
            "nll_rows_csv": str(nll_rows_path),
            "rate_metrics_csv": str(rate_metrics_path),
            "calibration_bins_csv": str(calibration_bins_path),
            "bootstrap_ci_csv": str(bootstrap_ci_path),
            "disease_stratified_metrics_csv": str(disease_metrics_path),
            "simulation_operating_characteristics_csv": str(simulation_path),
            "temporal_split_evaluation_json": str(temporal_summary_path),
            "temporal_prediction_rows_csv": str(temporal_predictions_path),
            "temporal_nll_rows_csv": str(temporal_nll_path),
            "report": str(report_path),
            "figures": [str(path) for path in figure_paths],
            "training_example_count": len(examples),
            "excluded_count": len(failures),
        },
        indent=2,
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-root", type=Path, default=DEFAULT_DB_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/retrospective_lambda_oncology"),
    )
    parser.add_argument(
        "--pipeline-results-jsonl",
        type=Path,
        default=None,
        help="Reuse existing pipeline_results.jsonl and skip Stage 1 pseudo-query generation.",
    )
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--endpoint-key", default="ORR")
    parser.add_argument("--max-queries", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--rerank-top-n", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--lambda0", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument(
        "--model-type",
        choices=lambda_training.LAMBDA_MODEL_TYPES,
        default="two_head_deepsets",
        help="Lambda scorer architecture. The default is the current optimized recommendation.",
    )
    parser.add_argument(
        "--listwise-eta",
        type=float,
        default=0.0,
        help="Weight for optional listwise borrowing-allocation auxiliary loss.",
    )
    parser.add_argument(
        "--listwise-temperature",
        type=float,
        default=1.0,
        help="Temperature for listwise candidate target distribution.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument(
        "--temporal-split-mode",
        choices=("fraction", "date_based", "rolling_origin"),
        default="fraction",
        help="Temporal validation split strategy. fraction preserves the previous behavior.",
    )
    parser.add_argument(
        "--temporal-train-end-date",
        default=None,
        help="Date cutoff for --temporal-split-mode date_based, e.g. 2020-12-31.",
    )
    parser.add_argument(
        "--temporal-eval-start-date",
        default=None,
        help="Optional eval-start date for date_based temporal validation.",
    )
    parser.add_argument(
        "--date-metadata-csv",
        type=Path,
        default=None,
        help="Optional clinicaltrials_date_rows.csv sidecar used to merge true CT.gov date metadata into query examples before temporal splitting.",
    )
    parser.add_argument(
        "--rolling-min-train-count",
        type=int,
        default=10,
        help="Minimum initial training examples for rolling-origin temporal validation.",
    )
    parser.add_argument(
        "--rolling-eval-window-size",
        type=int,
        default=10,
        help="Number of future examples per rolling-origin evaluation window.",
    )
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--simulation-iterations", type=int, default=500)
    parser.add_argument(
        "--include-pdf-text",
        action="store_true",
        help="Read protocol/SAP PDF excerpts. Disabled by default for full-data retrospective training throughput.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
