"""Selective two-head DeepSets borrowing model (Path-A pivot experiment).

Two architectural changes relative to the canonical two-head model, both
motivated by the review and by the pathfinding analysis
(artifacts_pathfinding/PATHFINDING_REPORT.md):

1. **Learnable total borrowing.** The weak component joins the candidate
   softmax through a set-level head computed from the pooled DeepSets
   context, so the model can decline to borrow on a per-query basis
   (previously ``lambda_0`` was fixed at 0.2, forcing 80% historical mass
   whenever any candidate passed its gate).
2. **Empirical-Bayes anchor.** The weak component is Beta(a0, b0) fitted by
   beta-binomial maximum likelihood on training-era queries only, replacing
   the uninformative Beta(1, 1). The evaluation reference baseline is the
   same EB prior alone.

The KL regulariser toward the rule allocation is applied to the *normalised*
candidate allocation (identical to the canonical candidate softmax), so it
constrains relative allocation without biasing the learned total borrowing.
Discount head, gating, ESS penalty, seeds, optimiser, and epochs match the
canonical training loop.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

_PIPELINE_DIR = Path(__file__).resolve().parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

import train_retrospective_lambda_model as LT  # noqa: E402


class SelectiveTwoHeadDeepSetsLambdaScorer(LT.TwoHeadDeepSetsLambdaScorer):
    """Two-head DeepSets scorer with a set-level weak-component logit head."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim)
        self.weak_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def weak_logit(self, features: torch.Tensor) -> torch.Tensor:
        set_context = self.phi(features).mean(dim=0)
        return self.weak_head(set_context).squeeze(-1)


def selective_lambda_weights(
    model: torch.nn.Module,
    features: torch.Tensor,
    gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (lambda0, lambda_i) from a joint softmax over weak + candidates."""
    scores = LT._model_scores(model, features)
    positive_gate_mask = gate > 0.0
    if not positive_gate_mask.any().item():
        return torch.tensor(1.0, dtype=torch.float32), torch.zeros_like(scores)
    log_raw = scores + gate.clamp_min(1e-12).log()
    masked_log_raw = torch.where(
        positive_gate_mask, log_raw, torch.full_like(log_raw, -torch.inf)
    )
    w0 = model.weak_logit(features).reshape(1)
    joint = torch.cat([w0, masked_log_raw])
    lam = torch.softmax(joint, dim=0)
    return lam[0], lam[1:]


def selective_loss_for_example(
    model: torch.nn.Module,
    example: dict[str, Any],
    anchor: tuple[float, float],
    rho: float = 0.1,
    ess_cap: float = 100.0,
) -> torch.Tensor:
    tensors = LT._validated_example_tensors(example)
    features = tensors["features"]
    gate = tensors["gate"]
    lambda0, lambda_i = selective_lambda_weights(model, features, gate)

    alpha, beta, active_discount = LT._component_alpha_beta_for_model(
        model, features, tensors
    )
    count = torch.tensor(tensors["query_count"], dtype=torch.float32)
    denominator = torch.tensor(tensors["query_denominator"], dtype=torch.float32)
    weak_log_predictive = LT.beta_binomial_log_predictive(
        count,
        denominator,
        torch.tensor(float(anchor[0]), dtype=torch.float32),
        torch.tensor(float(anchor[1]), dtype=torch.float32),
    )
    component_log_predictive = LT.beta_binomial_log_predictive(
        count, denominator, alpha, beta
    )
    mixture_terms = torch.cat(
        [
            (lambda0.clamp_min(1e-12).log() + weak_log_predictive).reshape(1),
            lambda_i.clamp_min(1e-12).log() + component_log_predictive,
        ]
    )
    loss = -torch.logsumexp(mixture_terms, dim=0)

    # KL toward the rule allocation over candidates only (normalised), which
    # matches the canonical regulariser's allocation target while leaving the
    # learned total borrowing unpenalised.
    lambda_rule_values = tensors["lambda_rule_values"]
    if lambda_rule_values is not None:
        lambda_rule = torch.tensor(lambda_rule_values, dtype=torch.float32)
        rule_sum = lambda_rule.sum()
        cand_sum = lambda_i.sum()
        if rule_sum.item() > 0.0 and cand_sum.item() > 0.0:
            rule_norm = lambda_rule / rule_sum
            cand_norm = lambda_i / cand_sum.clamp_min(1e-12)
            kl = (
                rule_norm
                * (rule_norm.clamp_min(1e-12).log() - cand_norm.clamp_min(1e-12).log())
            ).sum()
            loss = loss + float(rho) * kl

    ess = (lambda_i * active_discount * tensors["component_denominator"]).sum()
    cap = torch.tensor(float(ess_cap), dtype=torch.float32)
    loss = loss + 1e-4 * torch.relu(ess - cap).pow(2)
    return loss


def selective_details(
    model: torch.nn.Module,
    example: dict[str, Any],
    anchor: tuple[float, float],
) -> dict[str, Any]:
    """Pure predictive NLL plus allocation diagnostics (no regularisers)."""
    with torch.no_grad():
        tensors = LT._validated_example_tensors(example)
        features = tensors["features"]
        gate = tensors["gate"]
        lambda0, lambda_i = selective_lambda_weights(model, features, gate)
        alpha, beta, active_discount = LT._component_alpha_beta_for_model(
            model, features, tensors
        )
        count = torch.tensor(tensors["query_count"], dtype=torch.float32)
        denominator = torch.tensor(tensors["query_denominator"], dtype=torch.float32)
        weak_lp = LT.beta_binomial_log_predictive(
            count,
            denominator,
            torch.tensor(float(anchor[0]), dtype=torch.float32),
            torch.tensor(float(anchor[1]), dtype=torch.float32),
        )
        comp_lp = LT.beta_binomial_log_predictive(count, denominator, alpha, beta)
        mixture_terms = torch.cat(
            [
                (lambda0.clamp_min(1e-12).log() + weak_lp).reshape(1),
                lambda_i.clamp_min(1e-12).log() + comp_lp,
            ]
        )
        nll = float(-torch.logsumexp(mixture_terms, dim=0).item())
        return {
            "nll": nll,
            "lambda0": float(lambda0.item()),
            "borrowed_mass": float(lambda_i.sum().item()),
            "lambda_i": [float(v) for v in lambda_i],
            "discount_i": [float(v) for v in active_discount],
        }


def mean_selective_nll(model, examples, anchor) -> float:
    return sum(selective_details(model, ex, anchor)["nll"] for ex in examples) / len(examples)


def fit_eb_anchor(data: list[tuple[float, float]]) -> tuple[float, float]:
    """Beta-binomial MLE for an intercept-only Beta prior (grid + refine)."""

    def mnll(a: float, b: float) -> float:
        total = 0.0
        for y, m in data:
            total -= (
                math.lgamma(m + 1) - math.lgamma(y + 1) - math.lgamma(m - y + 1)
                + math.lgamma(y + a) + math.lgamma(m - y + b) - math.lgamma(m + a + b)
                + math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
            )
        return total / len(data)

    best = (1.0, 1.0, mnll(1.0, 1.0))
    for mu in [x / 100 for x in range(5, 61, 2)]:
        for k in [0.5, 1, 2, 3, 5, 7, 10, 15, 25, 50]:
            a, b = mu * k, (1 - mu) * k
            v = mnll(a, b)
            if v < best[2]:
                best = (a, b, v)
    a0, b0 = best[0], best[1]
    for _ in range(80):
        improved = False
        for da, db in [(1.02, 1), (0.98, 1), (1, 1.02), (1, 0.98),
                       (1.004, 1), (0.996, 1), (1, 1.004), (1, 0.996)]:
            a, b = a0 * da, b0 * db
            v = mnll(a, b)
            if v < best[2]:
                best = (a, b, v)
                a0, b0 = a, b
                improved = True
        if not improved:
            break
    return best[0], best[1]


def train_selective(
    examples: list[dict[str, Any]],
    train_indices: list[int],
    eval_indices: list[int],
    anchor: tuple[float, float],
    epochs: int = 100,
    learning_rate: float = 0.01,
    hidden_dim: int = 16,
    seed: int = 20260603,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    input_dim = len(LT.LAMBDA_FEATURE_NAMES)
    model = SelectiveTwoHeadDeepSetsLambdaScorer(input_dim=input_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_examples = [examples[i] for i in train_indices]
    eval_examples = [examples[i] for i in eval_indices]
    history = {"loss": [], "train_nll": [], "eval_nll": []}
    for _ in range(epochs):
        optimizer.zero_grad()
        losses = [selective_loss_for_example(model, ex, anchor) for ex in train_examples]
        objective = torch.stack(losses).mean()
        objective.backward()
        optimizer.step()
        history["loss"].append(float(objective.detach().item()))
        history["train_nll"].append(mean_selective_nll(model, train_examples, anchor))
        history["eval_nll"].append(
            mean_selective_nll(model, eval_examples, anchor) if eval_examples else float("nan")
        )
    return {"model": model, "history": history, "anchor": [float(anchor[0]), float(anchor[1])]}


def save_selective_artifact(path: Path, model, hidden_dim: int, anchor, extra=None) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "input_dim": len(LT.LAMBDA_FEATURE_NAMES),
        "hidden_dim": int(hidden_dim),
        "model_type": "selective_two_head_deepsets",
        "feature_names": list(LT.LAMBDA_FEATURE_NAMES),
        "anchor": [float(anchor[0]), float(anchor[1])],
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_selective_artifact(path: Path):
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    model = SelectiveTwoHeadDeepSetsLambdaScorer(
        input_dim=int(artifact["input_dim"]), hidden_dim=int(artifact["hidden_dim"])
    )
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model, tuple(artifact["anchor"])


def read_examples(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sort_date(example: dict[str, Any]) -> str:
    return (example.get("query_metadata") or {}).get("temporal_sort_date") or ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["primary", "fold"], default="primary")
    parser.add_argument("--train-end-date", default=None, help="fold mode cutoff")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260603)
    args = parser.parse_args()

    examples = read_examples(args.examples_jsonl)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "fold":
        if not args.train_end_date:
            raise SystemExit("--train-end-date required in fold mode")
        train_idx = [i for i, ex in enumerate(examples)
                     if sort_date(ex) and sort_date(ex) <= args.train_end_date]
        eval_idx = [i for i, ex in enumerate(examples)
                    if sort_date(ex) and sort_date(ex) > args.train_end_date]
        tag = f"fold_{args.train_end_date}"
    else:
        import evaluate_retrospective_lambda_model as LE
        train_idx, eval_idx = LE.deterministic_split_indices(
            len(examples), train_fraction=args.train_fraction, seed=args.seed
        )
        tag = "primary"

    anchor = fit_eb_anchor(
        [(examples[i]["query"]["count"], examples[i]["query"]["denominator"]) for i in train_idx]
    )
    result = train_selective(
        examples, train_idx, eval_idx, anchor,
        epochs=args.epochs, learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim, seed=args.seed,
    )
    model = result["model"]
    model_path = args.output_dir / f"selective_model_{tag}.pt"
    save_selective_artifact(model_path, model, args.hidden_dim, anchor,
                            extra={"train_count": len(train_idx), "eval_count": len(eval_idx)})
    summary = {
        "tag": tag,
        "train_count": len(train_idx),
        "eval_count": len(eval_idx),
        "anchor": result["anchor"],
        "seed": args.seed,
        "epochs": args.epochs,
        "final_loss": result["history"]["loss"][-1],
        "final_train_nll": result["history"]["train_nll"][-1],
        "final_eval_nll": result["history"]["eval_nll"][-1],
        "model_path": str(model_path),
    }
    (args.output_dir / f"selective_summary_{tag}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
