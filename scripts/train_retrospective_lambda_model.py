import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


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


def _validated_example_tensors(example: dict[str, Any]) -> dict[str, Any]:
    features = torch.tensor(example["features"], dtype=torch.float32)
    if features.ndim != 2:
        raise ValueError("features must be 2D")
    if features.shape[0] == 0:
        raise ValueError("features must be non-empty")
    if features.shape[1] == 0:
        raise ValueError("features must have at least one column")

    components = example["components"]
    if not components:
        raise ValueError("components must be non-empty")
    if features.shape[0] != len(components):
        raise ValueError("feature rows must equal component count")

    lambda_rule_values = example.get("lambda_rule")
    if lambda_rule_values is not None and len(lambda_rule_values) != len(components):
        raise ValueError("lambda_rule length must equal component count")

    return {
        "features": features,
        "components": components,
        "alpha": torch.tensor([component["alpha"] for component in components], dtype=torch.float32),
        "beta": torch.tensor([component["beta"] for component in components], dtype=torch.float32),
        "gate": torch.tensor([component.get("gate", 1.0) for component in components], dtype=torch.float32),
        "discount": torch.tensor([component.get("discount", 0.0) for component in components], dtype=torch.float32),
        "component_denominator": torch.tensor(
            [component.get("denominator", 0.0) for component in components],
            dtype=torch.float32,
        ),
        "lambda_rule_values": lambda_rule_values,
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
    lambda0 = torch.tensor(float(example.get("lambda_0", 0.2)), dtype=torch.float32)

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

    query = example.get("query", example)
    count = torch.tensor(float(query["count"]), dtype=torch.float32)
    denominator = torch.tensor(float(query["denominator"]), dtype=torch.float32)
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


def load_examples(path: str | Path) -> list[dict[str, Any]]:
    examples = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def train_model(
    examples: list[dict[str, Any]],
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
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
    return {
        "epochs": epochs,
        "final_loss": final_loss,
        "loss_history": loss_history,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=16)
    args = parser.parse_args(argv)

    examples = load_examples(args.examples_jsonl)
    summary = train_model(
        examples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
