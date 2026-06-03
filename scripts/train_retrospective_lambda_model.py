import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "docs") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "docs"))

import mixture_prior  # noqa: E402
import oncology_trial_similarity_pipeline as pipeline  # noqa: E402

import torch


LAMBDA_FEATURE_NAMES = [
    "s_i",
    "disease_match_i",
    "regimen_match_i",
    "endpoint_match_i",
    "followup_match_i",
    "eligibility_match_i",
    "result_quality_i",
    "negative_redflag_severity_i",
    "log_n_i",
]


class LambdaScorer(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def save_model_artifact(
    path: str | Path,
    model: LambdaScorer,
    input_dim: int,
    hidden_dim: int,
    lambda0: float,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(input_dim),
            "hidden_dim": int(hidden_dim),
            "feature_names": list(LAMBDA_FEATURE_NAMES),
            "lambda0": float(lambda0),
        },
        output,
    )


def load_model_artifact(path: str | Path) -> dict[str, Any]:
    artifact = torch.load(Path(path), map_location="cpu", weights_only=False)
    if artifact.get("feature_names") != LAMBDA_FEATURE_NAMES:
        raise ValueError("lambda model feature_names do not match current feature order")
    if int(artifact.get("input_dim", -1)) != len(LAMBDA_FEATURE_NAMES):
        raise ValueError("lambda model input_dim does not match current feature order")
    return artifact


def _log_choose(n: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return torch.lgamma(n + 1.0) - torch.lgamma(k + 1.0) - torch.lgamma(n - k + 1.0)


def _log_beta(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)


def beta_binomial_log_predictive(
    count: torch.Tensor,
    denominator: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    return (
        _log_choose(denominator, count)
        + _log_beta(count + alpha, denominator - count + beta)
        - _log_beta(alpha, beta)
    )


def _finite_float(name: str, value: Any) -> float:
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite")
    return numeric_value


def _validated_example_tensors(example: dict[str, Any]) -> dict[str, Any]:
    features = torch.tensor(example["features"], dtype=torch.float32)
    if features.ndim != 2:
        raise ValueError("features must be 2D")
    if features.shape[0] == 0:
        raise ValueError("features must be non-empty")
    if features.shape[1] == 0:
        raise ValueError("features must have at least one column")
    if features.shape[1] != len(LAMBDA_FEATURE_NAMES):
        raise ValueError("features must have 9 columns")

    feature_names = example.get("feature_names")
    if feature_names != LAMBDA_FEATURE_NAMES:
        raise ValueError("feature_names must match")

    components = example["components"]
    if not components:
        raise ValueError("components must be non-empty")
    if features.shape[0] != len(components):
        raise ValueError("feature rows must equal component count")

    lambda_rule_values = example.get("lambda_rule")
    if lambda_rule_values is not None and len(lambda_rule_values) != len(components):
        raise ValueError("lambda_rule length must equal component count")

    lambda0 = _finite_float("lambda_0", example.get("lambda_0", 0.2))
    if lambda0 < 0.0 or lambda0 > 1.0:
        raise ValueError("lambda_0 must be in [0, 1]")

    query = example.get("query", example)
    count = _finite_float("query count", query["count"])
    denominator = _finite_float("query denominator", query["denominator"])
    if denominator <= 0.0:
        raise ValueError("query denominator must be greater than 0")
    if count < 0.0:
        raise ValueError("query count must be non-negative")
    if count > denominator:
        raise ValueError("query count must be less than or equal to denominator")

    alpha_values = []
    beta_values = []
    for component in components:
        alpha = _finite_float("component alpha", component["alpha"])
        if alpha <= 0.0:
            raise ValueError("component alpha must be positive")
        beta = _finite_float("component beta", component["beta"])
        if beta <= 0.0:
            raise ValueError("component beta must be positive")
        alpha_values.append(alpha)
        beta_values.append(beta)

    return {
        "features": features,
        "components": components,
        "alpha": torch.tensor(alpha_values, dtype=torch.float32),
        "beta": torch.tensor(beta_values, dtype=torch.float32),
        "gate": torch.tensor([component.get("gate", 1.0) for component in components], dtype=torch.float32),
        "discount": torch.tensor([component.get("discount", 0.0) for component in components], dtype=torch.float32),
        "component_denominator": torch.tensor(
            [component.get("denominator", 0.0) for component in components],
            dtype=torch.float32,
        ),
        "feature_names": feature_names,
        "lambda_rule_values": lambda_rule_values,
        "lambda0": lambda0,
        "query_count": count,
        "query_denominator": denominator,
    }


def predictive_loss_for_example(
    model: LambdaScorer,
    example: dict[str, Any],
    rho: float = 0.1,
    ess_cap: float = 100.0,
) -> torch.Tensor:
    tensors = _validated_example_tensors(example)
    features = tensors["features"]
    scores = model(features)
    if scores.ndim != 1 or scores.shape[0] != features.shape[0]:
        raise ValueError("model scores must have one value per component")

    alpha = tensors["alpha"]
    beta = tensors["beta"]
    gate = tensors["gate"]
    discount = tensors["discount"]
    component_denominator = tensors["component_denominator"]
    lambda0 = torch.tensor(tensors["lambda0"], dtype=torch.float32)

    positive_gate_mask = gate > 0.0
    if positive_gate_mask.any().item():
        candidate_budget = torch.clamp(1.0 - lambda0, min=0.0)
        log_raw = scores + gate.clamp_min(1e-12).log()
        masked_log_raw = torch.where(
            positive_gate_mask,
            log_raw,
            torch.full_like(log_raw, -torch.inf),
        )
        lambda_i = candidate_budget * torch.softmax(masked_log_raw, dim=0)
        lambda0_tensor = lambda0
    else:
        candidate_budget = torch.tensor(0.0, dtype=torch.float32)
        lambda_i = torch.zeros_like(scores)
        lambda0_tensor = torch.tensor(1.0, dtype=torch.float32)

    count = torch.tensor(tensors["query_count"], dtype=torch.float32)
    denominator = torch.tensor(tensors["query_denominator"], dtype=torch.float32)
    weak_log_predictive = beta_binomial_log_predictive(
        count,
        denominator,
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(1.0, dtype=torch.float32),
    )
    component_log_predictive = beta_binomial_log_predictive(count, denominator, alpha, beta)

    mixture_terms = torch.cat(
        [
            (lambda0_tensor.clamp_min(1e-12).log() + weak_log_predictive).reshape(1),
            lambda_i.clamp_min(1e-12).log() + component_log_predictive,
        ]
    )
    loss = -torch.logsumexp(mixture_terms, dim=0)

    lambda_rule_values = tensors["lambda_rule_values"]
    if lambda_rule_values is not None:
        lambda_rule = torch.tensor(lambda_rule_values, dtype=torch.float32)
        rule_sum = lambda_rule.sum()
        if rule_sum.item() > 0.0:
            lambda_rule = lambda_rule / rule_sum * candidate_budget
            kl = (
                lambda_rule
                * (lambda_rule.clamp_min(1e-12).log() - lambda_i.clamp_min(1e-12).log())
            ).sum()
            loss = loss + float(rho) * kl

    ess = (lambda_i * discount * component_denominator).sum()
    cap = torch.tensor(float(ess_cap), dtype=torch.float32)
    ess_penalty = 1e-4 * torch.relu(ess - cap).pow(2)
    return loss + ess_penalty


def weak_only_loss_for_example(example: dict[str, Any]) -> torch.Tensor:
    tensors = _validated_example_tensors(example)
    count = torch.tensor(tensors["query_count"], dtype=torch.float32)
    denominator = torch.tensor(tensors["query_denominator"], dtype=torch.float32)
    weak_log_predictive = beta_binomial_log_predictive(
        count,
        denominator,
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(1.0, dtype=torch.float32),
    )
    return -weak_log_predictive


def learned_lambda_loss_for_example(model: LambdaScorer, example: dict[str, Any]) -> torch.Tensor:
    tensors = _validated_example_tensors(example)
    features = tensors["features"]
    scores = model(features)
    if scores.ndim != 1 or scores.shape[0] != features.shape[0]:
        raise ValueError("model scores must have one value per component")

    lambda0 = torch.tensor(tensors["lambda0"], dtype=torch.float32)
    gate = tensors["gate"]
    positive_gate_mask = gate > 0.0
    if positive_gate_mask.any().item():
        candidate_budget = torch.clamp(1.0 - lambda0, min=0.0)
        log_raw = scores + gate.clamp_min(1e-12).log()
        masked_log_raw = torch.where(
            positive_gate_mask,
            log_raw,
            torch.full_like(log_raw, -torch.inf),
        )
        lambda_i = candidate_budget * torch.softmax(masked_log_raw, dim=0)
        lambda0_tensor = lambda0
    else:
        lambda_i = torch.zeros_like(scores)
        lambda0_tensor = torch.tensor(1.0, dtype=torch.float32)

    count = torch.tensor(tensors["query_count"], dtype=torch.float32)
    denominator = torch.tensor(tensors["query_denominator"], dtype=torch.float32)
    weak_log_predictive = beta_binomial_log_predictive(
        count,
        denominator,
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(1.0, dtype=torch.float32),
    )
    component_log_predictive = beta_binomial_log_predictive(
        count,
        denominator,
        tensors["alpha"],
        tensors["beta"],
    )
    mixture_terms = torch.cat(
        [
            (lambda0_tensor.clamp_min(1e-12).log() + weak_log_predictive).reshape(1),
            lambda_i.clamp_min(1e-12).log() + component_log_predictive,
        ]
    )
    return -torch.logsumexp(mixture_terms, dim=0)


def rule_lambda_loss_for_example(example: dict[str, Any]) -> torch.Tensor:
    tensors = _validated_example_tensors(example)
    lambda_rule_values = tensors["lambda_rule_values"]
    if lambda_rule_values is None:
        return weak_only_loss_for_example(example)

    lambda_rule = torch.tensor(lambda_rule_values, dtype=torch.float32)
    rule_sum = lambda_rule.sum()
    if rule_sum.item() <= 0.0:
        return weak_only_loss_for_example(example)

    lambda0 = torch.tensor(tensors["lambda0"], dtype=torch.float32)
    candidate_budget = torch.clamp(1.0 - lambda0, min=0.0)
    lambda_rule = lambda_rule / rule_sum * candidate_budget

    count = torch.tensor(tensors["query_count"], dtype=torch.float32)
    denominator = torch.tensor(tensors["query_denominator"], dtype=torch.float32)
    weak_log_predictive = beta_binomial_log_predictive(
        count,
        denominator,
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(1.0, dtype=torch.float32),
    )
    component_log_predictive = beta_binomial_log_predictive(
        count,
        denominator,
        tensors["alpha"],
        tensors["beta"],
    )
    mixture_terms = torch.cat(
        [
            (lambda0.clamp_min(1e-12).log() + weak_log_predictive).reshape(1),
            lambda_rule.clamp_min(1e-12).log() + component_log_predictive,
        ]
    )
    return -torch.logsumexp(mixture_terms, dim=0)


def load_examples(path: str | Path) -> list[dict[str, Any]]:
    examples = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def redflag_severity(red_flags: list[Any]) -> float:
    raw = 0.0
    for flag in red_flags:
        text = str(flag).lower()
        if "low disease" in text or "low endpoint" in text:
            raw += 1.0
        elif "no primary endpoint-family overlap" in text:
            raw += 1.0
        elif "no normalized regimen-backbone overlap" in text:
            raw += 0.8
        elif "no posted results" in text or "no arm-level count/denominator" in text:
            raw += 0.8
        else:
            raw += 0.25
    return min(raw / 3.0, 1.0)


def _component_source_row(component: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    component_id = component.get("candidate_nct_id") or component.get("nct_id")
    for row in rows:
        row_id = row.get("candidate_nct_id") or row.get("nct_id")
        if component_id and row_id == component_id:
            return row
    return {}


def _dimension_score(dimension_scores: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = dimension_scores.get(key)
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric_value):
            return numeric_value
    return 0.0


def build_training_example_from_pipeline_result(
    result: dict[str, Any],
    endpoint_key: str = "ORR",
    lambda0: float = 0.2,
) -> dict[str, Any]:
    query_observations = pipeline.query_endpoint_observations(result["query_summary"])
    query = query_observations.get(endpoint_key)
    if query is None:
        raise ValueError(f"query_summary does not contain endpoint {endpoint_key}")

    count = query.get("treatment_count")
    denominator = query.get("treatment_denominator")
    if count is None or denominator is None:
        raise ValueError(f"query endpoint {endpoint_key} is missing treatment count/denominator")

    rows = result.get("reranked_top_matches") or result.get("reranked_top10") or []
    mixture = mixture_prior.components_from_reranked_rows(
        rows,
        endpoint_key=endpoint_key,
        lambda0=lambda0,
    )
    features = []
    components = []
    lambda_rule = []

    for component in mixture["components"]:
        source_row = _component_source_row(component, rows)
        dimension_scores = source_row.get("dimension_scores")
        if not isinstance(dimension_scores, dict):
            dimension_scores = {}
        red_flags = source_row.get("red_flags")
        if not isinstance(red_flags, list):
            red_flags = []
        denominator_i = float(component.get("denominator", 0.0))
        discount = float(component.get("discount", 0.0))
        gate = float(component.get("gate", 0.0))
        features.append(
            [
                float(component.get("overall_similarity_score", 0.0)) / 100.0,
                _dimension_score(
                    dimension_scores,
                    "disease_population_match",
                    "disease_biology_match",
                )
                / 5.0,
                _dimension_score(dimension_scores, "treatment_regimen_match") / 5.0,
                _dimension_score(dimension_scores, "endpoint_estimand_match") / 5.0,
                _dimension_score(
                    dimension_scores,
                    "safety_and_followup_relevance",
                    "outcome_assessment_followup",
                )
                / 5.0,
                _dimension_score(dimension_scores, "eligibility_criteria_overlap") / 5.0,
                _dimension_score(dimension_scores, "result_usability") / 5.0,
                -redflag_severity(red_flags),
                math.log1p(denominator_i),
            ]
        )
        components.append(
            {
                "alpha": float(component["alpha"]),
                "beta": float(component["beta"]),
                "gate": gate,
                "denominator": denominator_i,
                "discount": discount,
            }
        )
        lambda_rule.append(float(component.get("lambda_rule", 0.0)))

    return {
        "query": {"count": int(count), "denominator": int(denominator)},
        "lambda_0": float(mixture["lambda_0"]),
        "feature_names": LAMBDA_FEATURE_NAMES,
        "features": features,
        "components": components,
        "lambda_rule": lambda_rule,
    }


def features_from_mixture_components(components: list[dict[str, Any]]) -> list[list[float]]:
    features = []
    for component in components:
        explicit = component.get("lambda_features")
        if explicit is not None:
            if len(explicit) != len(LAMBDA_FEATURE_NAMES):
                raise ValueError("lambda_features must have 9 values")
            features.append([_finite_float("lambda feature", value) for value in explicit])
            continue

        denominator = _finite_float("component denominator", component.get("denominator", 0.0))
        features.append(
            [
                _finite_float(
                    "s_i",
                    component.get(
                        "s_i",
                        float(component.get("overall_similarity_score", 0.0)) / 100.0,
                    ),
                ),
                _finite_float("disease_match_i", component.get("disease_match_i", 0.0)),
                _finite_float("regimen_match_i", component.get("regimen_match_i", 0.0)),
                _finite_float("endpoint_match_i", component.get("endpoint_match_i", 0.0)),
                _finite_float("followup_match_i", component.get("followup_match_i", 0.0)),
                _finite_float("eligibility_match_i", component.get("eligibility_match_i", 0.0)),
                _finite_float("result_quality_i", component.get("result_quality_i", 0.0)),
                _finite_float(
                    "negative_redflag_severity_i",
                    component.get("negative_redflag_severity_i", 0.0),
                ),
                _finite_float("log_n_i", component.get("log_n_i", math.log1p(denominator))),
            ]
        )
    return features


def load_examples_from_pipeline_results(
    path: str | Path,
    endpoint_key: str = "ORR",
    lambda0: float = 0.2,
) -> list[dict[str, Any]]:
    examples = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(
                    build_training_example_from_pipeline_result(
                        json.loads(line),
                        endpoint_key=endpoint_key,
                        lambda0=lambda0,
                    )
                )
    return examples


def train_model(
    examples: list[dict[str, Any]],
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
    model_output: str | Path | None = None,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("examples must not be empty")
    if epochs <= 0:
        raise ValueError("epochs must be greater than 0")
    first_tensors = _validated_example_tensors(examples[0])

    input_dim = first_tensors["features"].shape[1]
    model = LambdaScorer(input_dim=input_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []

    for _ in range(epochs):
        optimizer.zero_grad()
        losses = [predictive_loss_for_example(model, example) for example in examples]
        mean_loss = torch.stack(losses).mean()
        mean_loss.backward()
        optimizer.step()
        loss_history.append(float(mean_loss.detach().item()))

    final_loss = loss_history[-1] if loss_history else math.nan
    if model_output is not None:
        save_model_artifact(
            model_output,
            model,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            lambda0=first_tensors["lambda0"],
        )

    summary = {
        "epochs": epochs,
        "final_loss": final_loss,
        "loss_history": loss_history,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "model": model,
    }
    if model_output is not None:
        summary["model_output"] = str(model_output)
    return summary


def serializable_training_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "model"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--examples-jsonl", type=Path, default=None)
    source_group.add_argument("--pipeline-results-jsonl", type=Path, default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--model-output", type=Path, default=None)
    args = parser.parse_args(argv)

    examples = []
    if args.examples_jsonl is not None:
        examples.extend(load_examples(args.examples_jsonl))
    if args.pipeline_results_jsonl is not None:
        examples.extend(load_examples_from_pipeline_results(args.pipeline_results_jsonl))

    summary = train_model(
        examples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        model_output=args.model_output,
    )
    serializable_summary = serializable_training_summary(summary)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(serializable_summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(serializable_summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
