from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "docs") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "docs"))

import mixture_prior  # noqa: E402


class SimulationScenario:
    def __init__(
        self,
        name: str,
        description: str,
        historical_rate_fn: Callable[[float, int, int], float],
    ) -> None:
        self.name = name
        self.description = description
        self.historical_rate_fn = historical_rate_fn


def _clip_probability(value: float) -> float:
    return max(1e-6, min(1.0 - 1e-6, float(value)))


def _logit(value: float) -> float:
    value = _clip_probability(value)
    return math.log(value / (1.0 - value))


def _expit(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def default_scenarios() -> list[SimulationScenario]:
    return [
        SimulationScenario(
            "exchangeable",
            "Historical and current trials share the same response probability.",
            lambda current, index, count: current,
        ),
        SimulationScenario(
            "mild_optimistic_conflict",
            "Historical trials are mildly more optimistic than the current trial on the logit scale.",
            lambda current, index, count: _expit(_logit(current) + 0.35),
        ),
        SimulationScenario(
            "strong_optimistic_conflict",
            "Historical trials are strongly more optimistic than the current trial on the logit scale.",
            lambda current, index, count: _expit(_logit(current) + 1.0),
        ),
        SimulationScenario(
            "mild_pessimistic_conflict",
            "Historical trials are mildly more pessimistic than the current trial on the logit scale.",
            lambda current, index, count: _expit(_logit(current) - 0.35),
        ),
        SimulationScenario(
            "mixture_historical_conflict",
            "Half of historical trials are exchangeable and half are optimistic-conflicting.",
            lambda current, index, count: current if index < max(1, count // 2) else _expit(_logit(current) + 1.0),
        ),
        SimulationScenario(
            "heterogeneous_historical",
            "Historical trials alternate around the current rate to mimic between-trial heterogeneity.",
            lambda current, index, count: _expit(_logit(current) + (-0.45 if index % 2 == 0 else 0.45)),
        ),
    ]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def select_template_examples(
    examples: list[dict[str, Any]],
    max_examples: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Return a deterministic template subset for higher-iteration simulations."""
    if max_examples is None or max_examples <= 0 or max_examples >= len(examples):
        return examples
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    selected = sorted(indices[:max_examples])
    return [examples[index] for index in selected]


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


def _binomial(rng: random.Random, n: int, p: float) -> int:
    p = _clip_probability(p)
    if hasattr(rng, "binomialvariate"):
        return int(rng.binomialvariate(n, p))
    return sum(1 for _ in range(n) if rng.random() < p)


def _beta_mean(alpha: float, beta: float) -> float:
    return alpha / (alpha + beta)


def _beta_var(alpha: float, beta: float) -> float:
    total = alpha + beta
    return alpha * beta / (total * total * (total + 1.0))


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _component_lambdas(example: dict[str, Any], source: str) -> list[float]:
    components = list(example.get("components") or [])
    values = []
    for index, component in enumerate(components):
        if source == "model":
            raw = component.get("lambda_model")
            if raw is None:
                raw = component.get("lambda_active")
        else:
            raw = component.get("lambda_rule")
        if raw is None and source == "rule":
            rule_values = example.get("lambda_rule") or []
            raw = rule_values[index] if index < len(rule_values) else 0.0
        try:
            values.append(max(0.0, float(raw)))
        except (TypeError, ValueError):
            values.append(0.0)
    return values


def _normalize_component_lambdas(values: list[float], lambda0: float) -> tuple[float, list[float]]:
    lambda0 = max(0.0, min(1.0, float(lambda0)))
    total = sum(values)
    if total <= 0.0:
        return 1.0, [0.0 for _ in values]
    budget = max(0.0, 1.0 - lambda0)
    return lambda0, [budget * value / total for value in values]


def _method_prior(
    example: dict[str, Any],
    simulated_histories: list[dict[str, float]],
    method: str,
    fixed_discount: float,
) -> dict[str, Any]:
    method = str(method)
    lambda0 = float(example.get("lambda_0", 0.2))
    components = []
    if method == "weak_only":
        return {"lambda_0": 1.0, "components": []}

    lambda_source = "model" if method.startswith("model") else "rule"
    raw_lambdas = _component_lambdas(example, lambda_source)
    lambda0, lambdas = _normalize_component_lambdas(raw_lambdas, lambda0=lambda0)
    for index, history in enumerate(simulated_histories):
        discount = history.get("discount", fixed_discount)
        if "fixed_discount" in method:
            discount = fixed_discount
        alpha = 1.0 + float(discount) * float(history["count"])
        beta = 1.0 + float(discount) * (float(history["denominator"]) - float(history["count"]))
        components.append(
            {
                "candidate_nct_id": history.get("candidate_nct_id", f"candidate_{index}"),
                "alpha": alpha,
                "beta": beta,
                "lambda": lambdas[index] if index < len(lambdas) else 0.0,
                "lambda_active": lambdas[index] if index < len(lambdas) else 0.0,
            }
        )
    return {"lambda_0": lambda0, "components": components}


def _posterior_summary(
    prior: dict[str, Any],
    y: int,
    n: int,
    decision_threshold: float,
    use_sam: bool,
) -> dict[str, float]:
    active_prior = prior
    sam_triggered = 0.0
    if use_sam:
        active_prior = mixture_prior.apply_sam_conflict_adapter(prior, y=y, n=n)
        conflict = active_prior.get("sam_prior_data_conflict") or {}
        sam_triggered = 1.0 if conflict.get("status") == "conflict_downweighted" else 0.0

    lambda0 = float(active_prior.get("lambda_0", 1.0))
    terms = [
        {
            "weight": lambda0,
            "alpha": 1.0,
            "beta": 1.0,
        }
    ]
    for component in active_prior.get("components") or []:
        terms.append(
            {
                "weight": float(component.get("lambda_active", component.get("lambda", 0.0))),
                "alpha": float(component["alpha"]),
                "beta": float(component["beta"]),
            }
        )

    predictive_weights = []
    for term in terms:
        predictive = mixture_prior.beta_binomial_predictive_probability(
            y,
            n,
            term["alpha"],
            term["beta"],
        )
        predictive_weights.append(max(0.0, term["weight"]) * predictive)
    total = sum(predictive_weights)
    if total <= 0.0:
        responsibilities = [1.0] + [0.0 for _ in terms[1:]]
    else:
        responsibilities = [value / total for value in predictive_weights]

    posterior_terms = []
    posterior_mean = 0.0
    for responsibility, term in zip(responsibilities, terms):
        alpha_post = term["alpha"] + y
        beta_post = term["beta"] + n - y
        mean = _beta_mean(alpha_post, beta_post)
        var = _beta_var(alpha_post, beta_post)
        posterior_terms.append((responsibility, mean, var))
        posterior_mean += responsibility * mean

    second_moment = sum(resp * (var + mean * mean) for resp, mean, var in posterior_terms)
    posterior_var = max(0.0, second_moment - posterior_mean * posterior_mean)
    posterior_sd = math.sqrt(posterior_var)
    lower = max(0.0, posterior_mean - 1.96 * posterior_sd)
    upper = min(1.0, posterior_mean + 1.96 * posterior_sd)
    if posterior_sd <= 0.0:
        prob_gt = 1.0 if posterior_mean > decision_threshold else 0.0
    else:
        prob_gt = 1.0 - _normal_cdf((decision_threshold - posterior_mean) / posterior_sd)

    historical_mass = sum(float(component.get("lambda_active", component.get("lambda", 0.0))) for component in active_prior.get("components") or [])
    return {
        "posterior_mean": posterior_mean,
        "interval_lower": lower,
        "interval_upper": upper,
        "interval_width": upper - lower,
        "probability_above_threshold": prob_gt,
        "historical_mass": historical_mass,
        "sam_triggered": sam_triggered,
    }


def _simulated_histories(
    rng: random.Random,
    example: dict[str, Any],
    scenario: SimulationScenario,
    current_rate: float,
) -> list[dict[str, float]]:
    histories = []
    source_components = list(example.get("components") or [])
    for index, component in enumerate(source_components):
        denominator = int(float(component.get("denominator") or component.get("n") or 0))
        if denominator <= 0:
            beta = float(component.get("beta", 1.0))
            alpha = float(component.get("alpha", 1.0))
            denominator = max(1, int(round(alpha + beta - 2.0)))
        p_hist = scenario.historical_rate_fn(current_rate, index, len(source_components))
        count = _binomial(rng, denominator, p_hist)
        discount = component.get("discount_model", component.get("discount_active", component.get("discount", 0.5)))
        histories.append(
            {
                "candidate_nct_id": str(component.get("candidate_nct_id") or component.get("nct_id") or f"candidate_{index}"),
                "count": float(count),
                "denominator": float(denominator),
                "discount": float(discount),
            }
        )
    return histories


def simulate_operating_characteristics(
    examples: list[dict[str, Any]],
    scenarios: list[SimulationScenario] | None = None,
    methods: tuple[str, ...] = ("weak_only", "rule", "model", "rule_sam"),
    iterations: int = 1000,
    seed: int = 20260607,
    null_rate: float = 0.2,
    alternative_rate: float = 0.35,
    decision_threshold: float = 0.3,
    posterior_probability_cutoff: float = 0.95,
    fixed_discount: float = 0.25,
) -> list[dict[str, Any]]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not examples:
        raise ValueError("at least one example is required")
    rng = random.Random(seed)
    scenarios = scenarios or default_scenarios()
    rows = []
    for scenario in scenarios:
        for method in methods:
            null_decisions = []
            alternative_decisions = []
            biases = []
            squared_errors = []
            coverages = []
            interval_widths = []
            historical_masses = []
            sam_triggers = []
            for truth_label, truth_rate, decision_store in (
                ("null", null_rate, null_decisions),
                ("alternative", alternative_rate, alternative_decisions),
            ):
                for _ in range(iterations):
                    for example in examples:
                        n_current = int(float((example.get("query") or {}).get("denominator") or 0))
                        if n_current <= 0:
                            continue
                        histories = _simulated_histories(rng, example, scenario, truth_rate)
                        y_current = _binomial(rng, n_current, truth_rate)
                        use_sam = method.endswith("_sam")
                        base_method = method.replace("_sam", "")
                        prior = _method_prior(example, histories, base_method, fixed_discount=fixed_discount)
                        summary = _posterior_summary(
                            prior,
                            y=y_current,
                            n=n_current,
                            decision_threshold=decision_threshold,
                            use_sam=use_sam,
                        )
                        decision = 1.0 if summary["probability_above_threshold"] >= posterior_probability_cutoff else 0.0
                        decision_store.append(decision)
                        if truth_label == "alternative":
                            error = summary["posterior_mean"] - truth_rate
                            biases.append(error)
                            squared_errors.append(error * error)
                            coverages.append(1.0 if summary["interval_lower"] <= truth_rate <= summary["interval_upper"] else 0.0)
                            interval_widths.append(summary["interval_width"])
                            historical_masses.append(summary["historical_mass"])
                            sam_triggers.append(summary["sam_triggered"])
            rows.append(
                {
                    "scenario": scenario.name,
                    "scenario_description": scenario.description,
                    "method": method,
                    "iterations": iterations,
                    "example_count": len(examples),
                    "simulated_trial_count": len(alternative_decisions),
                    "null_rate": null_rate,
                    "alternative_rate": alternative_rate,
                    "decision_threshold": decision_threshold,
                    "posterior_probability_cutoff": posterior_probability_cutoff,
                    "type_i_error": _mean(null_decisions),
                    "power": _mean(alternative_decisions),
                    "bias": _mean(biases),
                    "mse": _mean(squared_errors),
                    "coverage": _mean(coverages),
                    "mean_interval_width": _mean(interval_widths),
                    "sam_trigger_rate": _mean(sam_triggers),
                    "mean_historical_mass": _mean(historical_masses),
                }
            )
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Formal Borrowing Operating Characteristics Simulation",
        "",
        "These simulations use leakage-free retrospective lambda examples as candidate-set templates,",
        "then generate current and historical binomial outcomes under prespecified exchangeability",
        "and conflict scenarios. They do not use expert borrowability labels.",
        "",
        "| Scenario | Method | Type I error | Power | Bias | MSE | Coverage | SAM trigger |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {method} | {type_i_error:.4f} | {power:.4f} | {bias:.4f} | {mse:.4f} | {coverage:.4f} | {sam_trigger_rate:.4f} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run formal simulation operating characteristics for borrowing priors.")
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/operating_characteristics_simulation"))
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--methods", nargs="+", default=["weak_only", "rule", "model", "rule_sam", "model_sam", "fixed_discount"])
    parser.add_argument("--null-rate", type=float, default=0.2)
    parser.add_argument("--alternative-rate", type=float, default=0.35)
    parser.add_argument("--decision-threshold", type=float, default=0.3)
    parser.add_argument("--posterior-probability-cutoff", type=float, default=0.95)
    parser.add_argument("--fixed-discount", type=float, default=0.25)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional deterministic template subsample size for higher-iteration simulations.",
    )
    args = parser.parse_args(argv)

    examples = select_template_examples(
        read_jsonl(args.examples_jsonl),
        max_examples=args.max_examples,
        seed=args.seed,
    )
    rows = simulate_operating_characteristics(
        examples=examples,
        methods=tuple(args.methods),
        iterations=args.iterations,
        seed=args.seed,
        null_rate=args.null_rate,
        alternative_rate=args.alternative_rate,
        decision_threshold=args.decision_threshold,
        posterior_probability_cutoff=args.posterior_probability_cutoff,
        fixed_discount=args.fixed_discount,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "simulation_operating_characteristics.csv", rows)
    (args.output_dir / "simulation_scenarios.json").write_text(
        json.dumps(
            [{"name": scenario.name, "description": scenario.description} for scenario in default_scenarios()],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_report(args.output_dir / "simulation_operating_characteristics_report.md", rows)


if __name__ == "__main__":
    main()
