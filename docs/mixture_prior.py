"""Pure Python mixture-prior math utilities."""

from __future__ import annotations

import math


def _validate_observation(y: int, n: int) -> None:
    if n < 0:
        raise ValueError("n must be non-negative")
    if y < 0 or y > n:
        raise ValueError("y must satisfy 0 <= y <= n")


def _validate_beta_parameters(alpha: float, beta: float) -> None:
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")


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


def normalize_lambdas(raw_weights: list[float], lambda0: float = 0.2) -> tuple[float, list[float]]:
    """Reserve lambda0 for the weak component and normalize non-negative candidate weights."""
    if lambda0 < 0 or lambda0 > 1:
        raise ValueError("lambda0 must be in [0, 1]")

    clipped_weights = [max(0.0, float(weight)) for weight in raw_weights]
    total = sum(clipped_weights)
    if total <= 0:
        return 1.0, [0.0 for _ in clipped_weights]

    candidate_budget = 1.0 - lambda0
    lambdas = [candidate_budget * weight / total for weight in clipped_weights]
    return float(lambda0), lambdas


def mixture_predictive_probability(
    y: int,
    n: int,
    lambda0: float,
    weak_alpha: float,
    weak_beta: float,
    components: list[dict[str, float]],
) -> float:
    """Return the lambda-weighted predictive probability across weak and candidate priors."""
    weak_probability = beta_binomial_predictive_probability(y, n, weak_alpha, weak_beta)
    total_probability = lambda0 * weak_probability

    for component in components:
        component_probability = beta_binomial_predictive_probability(
            y,
            n,
            component["alpha"],
            component["beta"],
        )
        total_probability += component["lambda"] * component_probability

    return total_probability


def posterior_component_weights(
    y: int,
    n: int,
    lambda0: float,
    weak_alpha: float,
    weak_beta: float,
    components: list[dict[str, float]],
) -> tuple[float, list[float]]:
    """Return posterior responsibility weights for weak and candidate components."""
    weak_probability = beta_binomial_predictive_probability(y, n, weak_alpha, weak_beta)
    weak_weight = lambda0 * weak_probability

    component_weights = []
    for component in components:
        component_probability = beta_binomial_predictive_probability(
            y,
            n,
            component["alpha"],
            component["beta"],
        )
        component_weights.append(component["lambda"] * component_probability)

    normalizer = weak_weight + sum(component_weights)
    if normalizer <= 0:
        raise ValueError("posterior weights cannot be normalized from zero probability")

    return weak_weight / normalizer, [weight / normalizer for weight in component_weights]
