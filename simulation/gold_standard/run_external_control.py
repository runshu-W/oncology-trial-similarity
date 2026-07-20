"""External / synthetic control arm simulation (手稿修改方案 v3, section 4).

The current trial randomises to a treatment arm and a SMALL internal control
arm. Candidate donors are historical control arms retrieved from the
literature; borrowing augments the internal control to form a hybrid control.
The estimand is the treatment effect theta_trt - theta_ctl, and the decision
rule declares success when P(theta_trt > theta_ctl | data) > SUCCESS_THRESHOLD.

This is the setting regulators scrutinise most closely, because the failure mode
is directional: if the borrowed external controls are systematically WORSE than
the internal control, the hybrid control is dragged down, the apparent treatment
effect is inflated, and type I error rises. The design worlds below are built to
expose exactly that, by sweeping the drift both downward and upward.

Usage:
    python3 run_external_control.py --replicates 1000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dgm import (ExternalControlConfig, simulate_external_control_dataset)  # noqa: E402
from methods import (  # noqa: E402
    TwoHeadDeepSets, apply_sam, pooling, power_prior, robust_map,
    rule_components, train_two_head, two_head_prior, uip_dirichlet, uip_js,
    weak_only,
)
from metrics import (  # noqa: E402
    SUCCESS_THRESHOLD, classification_counts, mcse_mean, mcse_prop,
    oracle_weights, pr_auc, roc_auc, weight_agreement,
)

FIXED_DISCOUNT = 0.35
DRIFT_GRID = [-1.2, -0.8, -0.4, 0.0, 0.4, 0.8]
EFFECT_NULL = 0.0
EFFECT_ALT = 0.15


def seed_for(name: str, base: int) -> int:
    return base + (zlib.crc32(name.encode("utf-8")) % 100000)


def method_table(models):
    base, pro, pro_fixed = models["base"], models["pro"], models["pro_fixed"]
    return {
        # The regulatory reference: use the internal control arm only.
        "internal_only": (lambda c, y0, n0: weak_only(c), False),
        "rule": (lambda c, y0, n0: rule_components(c), False),
        "two_head": (lambda c, y0, n0: two_head_prior(base, c), False),
        "two_head_pro": (
            lambda c, y0, n0: two_head_prior(pro, c, prospective=True), False),
        "two_head_pro_sam": (
            lambda c, y0, n0: two_head_prior(pro, c, prospective=True), True),
        "two_head_pro_fixdisc": (
            lambda c, y0, n0: two_head_prior(pro_fixed, c, prospective=True,
                                             fixed_discount=FIXED_DISCOUNT), False),
        "robust_map_w0.5": (lambda c, y0, n0: robust_map(c, w=0.5), False),
        "power_prior": (lambda c, y0, n0: power_prior(c, a0=0.30), False),
        "uip_dirichlet": (lambda c, y0, n0: uip_dirichlet(c, y0, n0), False),
        "uip_js": (lambda c, y0, n0: uip_js(c, y0, n0), False),
        "pooling": (lambda c, y0, n0: pooling(c), False),
    }


def summarise(rows):
    eff_bias = np.array([r["effect_bias"] for r in rows])
    ctl_bias = np.array([r["control_bias"] for r in rows])
    rej = np.array([r["reject"] for r in rows], dtype=float)
    cov = np.array([r["covered"] for r in rows], dtype=float)
    ess = np.array([r["prior_ess"] for r in rows])
    n = len(rows)

    aucs = np.array([r["auc"] for r in rows if not np.isnan(r["auc"])])
    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows); tn = sum(r["tn"] for r in rows)
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    fpr = 1 - spec if spec == spec else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")

    osp = np.array([r["oracle_spearman"] for r in rows
                    if not np.isnan(r.get("oracle_spearman", np.nan))])
    return {
        "n_replicates": n,
        "reject_rate": float(rej.mean()), "reject_rate_mcse": mcse_prop(rej.mean(), n),
        "effect_bias": float(eff_bias.mean()), "effect_bias_mcse": mcse_mean(eff_bias),
        "effect_rmse": float(np.sqrt((eff_bias ** 2).mean())),
        "control_bias": float(ctl_bias.mean()), "control_bias_mcse": mcse_mean(ctl_bias),
        "effect_coverage95": float(cov.mean()), "effect_coverage95_mcse": mcse_prop(cov.mean(), n),
        "prior_ess": float(ess.mean()), "prior_ess_mcse": mcse_mean(ess),
        "roc_auc": float(aucs.mean()) if aucs.size else float("nan"),
        "roc_auc_mcse": mcse_mean(aucs) if aucs.size > 1 else float("nan"),
        "sensitivity": sens, "specificity": spec, "fpr": fpr, "ppv": ppv,
        "fpr_mcse": mcse_prop(fpr, tn + fp) if (tn + fp) else float("nan"),
        "oracle_spearman": float(osp.mean()) if osp.size else float("nan"),
    }


def evaluate(dataset, models, cfg):
    table = method_table(models)
    results = {name: [] for name in table}
    for q in dataset:
        cands = q["candidates"]
        labels = np.array([c["borrowable"] for c in cands])
        y_ctl, n_ctl = q["y_control"], q["n_control"]
        y_trt, n_trt = q["y_treatment"], q["n_treatment"]
        theta_ctl, true_eff = q["theta_control"], q["true_effect"]
        oracle = oracle_weights(cands, theta_ctl, cfg.epsilon)

        for name, (build, use_sam) in table.items():
            prior = build(cands, y_ctl, n_ctl)
            if use_sam:
                prior, _ = apply_sam(prior, y_ctl, n_ctl)
            eff, lo, hi, p_sup, ctl_mean = prior.treatment_effect(
                y_ctl, n_ctl, y_trt, n_trt)
            tp, fp, fn, tn = classification_counts(prior.scores, labels, len(cands))
            osp, _ = weight_agreement(prior.scores, oracle)
            results[name].append({
                "effect_bias": eff - true_eff,
                "control_bias": ctl_mean - theta_ctl,
                "reject": float(p_sup > SUCCESS_THRESHOLD),
                "covered": bool(lo <= true_eff <= hi),
                "prior_ess": prior.prior_ess(),
                "auc": roc_auc(prior.scores, labels),
                "prauc": pr_auc(prior.scores, labels),
                "oracle_spearman": osp,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            })
    return {name: summarise(rows) for name, rows in results.items()}


def _dump(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--train-size", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260720)
    ap.add_argument("--output", type=str,
                    default="../../artifacts/gold_standard_simulation")
    ap.add_argument("--cell", type=str, default=None,
                    help="run only this cell, e.g. null_-0.8 or pcomp_0.5")
    ap.add_argument("--no-combine", action="store_true")
    ap.add_argument("--skip-train", action="store_true",
                    help="reuse previously trained EC models")
    args = ap.parse_args()

    out = Path(args.output)
    if not out.is_absolute():
        out = (Path(__file__).resolve().parent / out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Train the two-head variants on an independent EXTERNAL-CONTROL training
    # draw. The borrowing target here is a control-arm rate, so the model is
    # trained on the task it is evaluated on rather than transferred from the
    # treatment-arm task. Training uses a neutral drift mixture so the model is
    # not tuned to any single drift level it is later tested at.
    train_cfg = ExternalControlConfig(name="ec_train", drift_shift=-0.6,
                                      p_comparable=0.5)
    specs = [("base", 9, dict(prospective=False)),
             ("pro", 10, dict(prospective=True)),
             ("pro_fixed", 10, dict(prospective=True, fixed_discount=FIXED_DISCOUNT))]
    models = {}
    if args.skip_train and all((out / f"ec_two_head_{k}.npz").exists() for k, _, _ in specs):
        for key, dim, _ in specs:
            model = TwoHeadDeepSets(input_dim=dim)
            z = np.load(out / f"ec_two_head_{key}.npz")
            model.load([z[f"p{i}"] for i in range(len(model.params))])
            models[key] = model
    else:
        train = simulate_external_control_dataset(train_cfg, args.train_size, args.seed + 1)
        for key, _, kwargs in specs:
            print(f"Training EC two-head [{key}] ...", flush=True)
            m = train_two_head(train, epochs=args.epochs, seed=20260603,
                               verbose=True, **kwargs)
            m.save_npz(out / f"ec_two_head_{key}.npz")
            models[key] = m

    # Each cell is checkpointed so an interrupted run resumes.
    parts = out / "ec_parts"
    parts.mkdir(exist_ok=True)
    for world, effect in (("null", EFFECT_NULL), ("alt", EFFECT_ALT)):
        for drift in DRIFT_GRID:
            cell = f"{world}_{drift}"
            if args.cell and args.cell != cell:
                continue
            part = parts / f"{cell}.csv"
            if part.exists():
                continue
            cfg = ExternalControlConfig(
                name=f"EC_{world}_drift{drift}", treatment_effect=effect,
                drift_shift=drift)
            print(f"EC world={world} drift={drift} ...", flush=True)
            data = simulate_external_control_dataset(
                cfg, args.replicates, seed_for(cfg.name, args.seed))
            _dump([{"world": world, "drift_shift": drift,
                    "treatment_effect": effect, "method": mname, **stats}
                   for mname, stats in evaluate(data, models, cfg).items()], part)

    # Sensitivity to how much of the candidate pool is genuinely comparable.
    for p_comp in (0.25, 0.50, 0.75):
        cell = f"pcomp_{p_comp}"
        if args.cell and args.cell != cell:
            continue
        part = parts / f"{cell}.csv"
        if part.exists():
            continue
        cfg = ExternalControlConfig(
            name=f"EC_pcomp{p_comp}", treatment_effect=EFFECT_NULL,
            drift_shift=-0.8, p_comparable=p_comp)
        print(f"EC p_comparable={p_comp} ...", flush=True)
        data = simulate_external_control_dataset(
            cfg, args.replicates, seed_for(cfg.name, args.seed))
        _dump([{"p_comparable": p_comp, "method": mname, **stats}
               for mname, stats in evaluate(data, models, cfg).items()], part)

    if not args.no_combine:
        rows, rows2 = [], []
        for p in sorted(parts.glob("*.csv")):
            with open(p, newline="", encoding="utf-8") as fh:
                (rows2 if p.stem.startswith("pcomp") else rows).extend(
                    list(csv.DictReader(fh)))
        _dump(rows, out / "external_control.csv")
        _dump(rows2, out / "external_control_pcomp.csv")

    (out / "external_control_config.json").write_text(json.dumps({
        "replicates": args.replicates, "seed": args.seed,
        "drift_grid": DRIFT_GRID, "effect_null": EFFECT_NULL,
        "effect_alt": EFFECT_ALT, "success_threshold": SUCCESS_THRESHOLD,
        "train_drift_shift": train_cfg.drift_shift,
    }, indent=2), encoding="utf-8")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
