"""Pure Python mixture-prior math utilities."""

from __future__ import annotations

import math
from typing import Any


def _validate_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")


def _validate_finite_number(name: str, value: float) -> float:
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite")
    return numeric_value


def _validate_observation(y: int, n: int) -> None:
    _validate_integer("y", y)
    _validate_integer("n", n)
    if n < 0:
        raise ValueError("n must be non-negative")
    if y < 0 or y > n:
        raise ValueError("y must satisfy 0 <= y <= n")


def _validate_beta_parameters(alpha: float, beta: float) -> None:
    alpha = _validate_finite_number("alpha", alpha)
    beta = _validate_finite_number("beta", beta)
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")


def _validate_lambda0(lambda0: float) -> float:
    lambda0 = _validate_finite_number("lambda0", lambda0)
    if lambda0 < 0 or lambda0 > 1:
        raise ValueError("lambda0 must be in [0, 1]")
    return lambda0


def _validate_component_lambda(component_lambda: float) -> float:
    component_lambda = _validate_finite_number("component lambda", component_lambda)
    if component_lambda < 0:
        raise ValueError("component lambda must be non-negative")
    return component_lambda


def _optional_finite_number(value: Any) -> float | None:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric_value):
        return None
    return numeric_value


def _canonical_endpoint_key(
    family: str | None,
    title: str | None,
    time_frame: str | None = "",
) -> str | None:
    combined = " ".join(str(value).lower() for value in (family, title, time_frame) if value)
    if not combined:
        return None

    if "orr" in combined or "response" in combined or "complete response" in combined:
        return "ORR"
    if "pfs6" in combined or (
        "progression" in combined and "6" in combined and "month" in combined
    ):
        return "PFS6"
    return None


def _selected_treatment_observation(rows: Any) -> tuple[float, float, float] | None:
    if not isinstance(rows, list):
        return None

    excluded_arm_terms = ("placebo", "control", "comparator")
    for row in rows:
        if not isinstance(row, dict):
            continue

        arm = str(row.get("arm") or "").lower()
        if any(term in arm for term in excluded_arm_terms):
            continue

        count = _optional_finite_number(row.get("count"))
        denominator = _optional_finite_number(row.get("denominator"))
        if count is None or denominator is None:
            continue
        if count < 0 or denominator <= 0 or count > denominator:
            continue

        return count, denominator, count / denominator

    return None


def conservative_gate(dimension_scores: dict[str, Any], red_flags: list[Any]) -> float:
    endpoint = _optional_finite_number(dimension_scores.get("endpoint_estimand_match")) or 0.0
    result = _optional_finite_number(dimension_scores.get("result_usability")) or 0.0
    disease = _optional_finite_number(dimension_scores.get("disease_biology_match")) or 0.0

    if endpoint < 1.5 or result <= 0:
        return 0.0

    gate = 1.0
    if disease < 1.5:
        gate *= 0.2
    elif disease < 2.5:
        gate *= 0.6

    if any("Low " in str(flag) for flag in red_flags):
        gate *= 0.5

    return gate


def _log_choose(n: int, y: int) -> float:
    """Return log of n choose y."""
    _validate_observation(y, n)
    return math.lgamma(n + 1) - math.lgamma(y + 1) - math.lgamma(n - y + 1)


def _log_beta(alpha: float, beta: float) -> float:
    """Return log of the beta function."""
    _validate_beta_parameters(alpha, beta)
    return math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)


def beta_binomial_predictive_probability(
    y: int,
    n: int,
    alpha: float,
    beta: float,
) -> float:
    """Compute the beta-binomial predictive probability for y successes in n trials."""
    _validate_observation(y, n)
    _validate_beta_parameters(alpha, beta)

    log_probability = (
        _log_choose(n, y)
        + _log_beta(y + alpha, n - y + beta)
        - _log_beta(alpha, beta)
    )
    return math.exp(log_probability)


def normalize_lambdas(raw_weights: list[float], lambda0: float = 0.2) -> dict[str, float | list[float]]:
    """Reserve lambda0 for the weak component and normalize non-negative candidate weights."""
    lambda0 = _validate_lambda0(lambda0)

    clipped_weights = [
        max(0.0, _validate_finite_number("raw weight", weight))
        for weight in raw_weights
    ]
    total = sum(clipped_weights)
    if total <= 0:
        return {"lambda_0": 1.0, "lambda_i": [0.0 for _ in clipped_weights]}

    candidate_budget = 1.0 - lambda0
    lambdas = [candidate_budget * weight / total for weight in clipped_weights]
    return {"lambda_0": float(lambda0), "lambda_i": lambdas}


def components_from_reranked_rows(
    rows: list[dict[str, Any]],
    endpoint_key: str,
    lambda0: float = 0.2,
) -> dict[str, Any]:
    """Build rule-weighted beta components from Stage-2 reranked candidate rows."""
    target_endpoint_key = str(endpoint_key).strip().upper()
    components: list[dict[str, Any]] = []
    raw_weights: list[float] = []

    for row in rows[:10]:
        if not isinstance(row, dict):
            continue

        discount = _optional_finite_number(row.get("suggested_borrowing_discount"))
        if discount is None:
            discount = 0.0
        discount = max(0.0, min(1.0, discount))

        selected_observation = None
        quantities = row.get("borrowable_quantities")
        if not isinstance(quantities, list):
            quantities = []

        for quantity in quantities:
            if not isinstance(quantity, dict):
                continue

            candidate_key = _canonical_endpoint_key(
                quantity.get("endpoint_family"),
                quantity.get("endpoint"),
                quantity.get("time_frame", ""),
            )
            if candidate_key != target_endpoint_key:
                continue

            selected_observation = _selected_treatment_observation(quantity.get("arm_results"))
            if selected_observation is not None:
                break

        if selected_observation is None:
            continue

        y_i, n_i, rate = selected_observation
        dimension_scores = row.get("dimension_scores")
        if not isinstance(dimension_scores, dict):
            dimension_scores = {}
        red_flags = row.get("red_flags")
        if not isinstance(red_flags, list):
            red_flags = []

        overall = _optional_finite_number(row.get("overall_similarity_score")) or 0.0
        gate = conservative_gate(dimension_scores, red_flags)
        raw_weight = gate * discount * max(0.0, overall) / 100.0 * math.log1p(n_i)

        components.append(
            {
                "candidate_nct_id": row.get("candidate_nct_id"),
                "endpoint_key": target_endpoint_key,
                "count": y_i,
                "denominator": n_i,
                "rate": rate,
                "discount": discount,
                "gate": gate,
                "raw_weight": raw_weight,
                "alpha": 1.0 + discount * y_i,
                "beta": 1.0 + discount * (n_i - y_i),
            }
        )
        raw_weights.append(raw_weight)

    normalized = normalize_lambdas(raw_weights, lambda0=lambda0)
    lambdas = normalized["lambda_i"]
    for component, lambda_rule in zip(components, lambdas):
        component["lambda_rule"] = lambda_rule

    return {"lambda_0": normalized["lambda_0"], "components": components}


def mixture_predictive_probability(
    y: int,
    n: int,
    lambda0: float,
    weak_alpha: float,
    weak_beta: float,
    components: list[dict[str, float]],
) -> float:
    """Return the lambda-weighted predictive probability across weak and candidate priors."""
    lambda0 = _validate_lambda0(lambda0)
    weak_probability = beta_binomial_predictive_probability(y, n, weak_alpha, weak_beta)
    total_probability = lambda0 * weak_probability

    for component in components:
        component_lambda = _validate_component_lambda(component["lambda"])
        component_probability = beta_binomial_predictive_probability(
            y,
            n,
            component["alpha"],
            component["beta"],
        )
        total_probability += component_lambda * component_probability

    return total_probability


def posterior_component_weights(
    y: int,
    n: int,
    lambda0: float,
    weak_alpha: float,
    weak_beta: float,
    components: list[dict[str, float]],
) -> dict[str, float | list[float]]:
    """Return posterior responsibility weights for weak and candidate components."""
    lambda0 = _validate_lambda0(lambda0)
    weak_probability = beta_binomial_predictive_probability(y, n, weak_alpha, weak_beta)
    weak_weight = lambda0 * weak_probability

    component_weights = []
    for component in components:
        component_lambda = _validate_component_lambda(component["lambda"])
        component_probability = beta_binomial_predictive_probability(
            y,
            n,
            component["alpha"],
            component["beta"],
        )
        component_weights.append(component_lambda * component_probability)

    normalizer = weak_weight + sum(component_weights)
    if normalizer <= 0:
        return {
            "lambda_0_post": 1.0,
            "lambda_i_post": [0.0 for _ in components],
        }

    return {
        "lambda_0_post": weak_weight / normalizer,
        "lambda_i_post": [weight / normalizer for weight in component_weights],
    }
