"""Pure Python mixture-prior math utilities."""

from __future__ import annotations

import math


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
