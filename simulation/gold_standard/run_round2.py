"""Round-2 gold-standard simulation: selective methods + theta-level labels.

Extends the released runners without disturbing them:
  - datasets are drawn with the SAME per-cell seeds as the published run
    (verified bit-identical), so previously reported OC rows must reproduce;
  - primary discrimination labels are now parameter-level (|theta_true -
    theta_0| <= eps); the released observation-process labels are kept as
    secondary *_obs columns (review round 2, Major point 3);
  - five new methods join every cell: selective (pro), selective without the
    prospective signal, selective with fixed discount, selective+SAM (anchored),
    and the intercept-only EB reference (review round 2, Major point 2).

Usage:
  python3 run_round2.py --which gold --replicates 2000 --output <dir>
  python3 run_round2.py --which ec   --replicates 2000 --output <dir>
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

from dgm import (ExternalControlConfig, SimConfig, label_summary,  # noqa: E402
                 simulate_dataset, simulate_external_control_dataset)
from methods import (  # noqa: E402
    TwoHeadDeepSets, apply_sam, pooling, power_prior, robust_map,
    rule_components, train_two_head, two_head_prior, uip_dirichlet, uip_js,
    weak_only,
)
from metrics import (  # noqa: E402
    DESIGN_DELTA, POWER_MARGIN, SUCCESS_THRESHOLD, THETA_NULL,
    classification_counts, ess_response, mcse_mean, mcse_prop, oracle_weights,
    pr_auc, roc_auc, summarise_replicates, weight_agreement,
)
from run_external_control import (  # noqa: E402
    DRIFT_GRID, EFFECT_ALT, EFFECT_NULL, summarise as ec_summarise)
from run_simulation import (  # noqa: E402
    CONFLICT_GRID, FIXED_DISCOUNT, design_world, scenarios, seed_for)
from selective_methods import (  # noqa: E402
    apply_sam_anchored, eb_only, fit_eb_anchor, load_selective_npz,
    save_selective_npz, selective_prior, train_selective)


def oracle_weights_param(cands, theta0, epsilon):
    w = np.array([(c["n"] if abs(c["theta_true"] - theta0) <= epsilon else 0.0)
                  for c in cands], dtype=float)
    total = w.sum()
    return w / total if total > 0 else w


# ---------------------------------------------------------------------------
# Training / loading
# ---------------------------------------------------------------------------
def train_models_gold(args, out):
    train = simulate_dataset(SimConfig(name="train"), args.train_size, args.seed + 1)
    anchor = fit_eb_anchor([(q["y_query"], q["n_query"]) for q in train])
    print(f"EB anchor (gold train draw): Beta({anchor[0]:.3f}, {anchor[1]:.3f})", flush=True)
    models = {}
    for key, kwargs in [("base", dict(prospective=False)),
                        ("pro", dict(prospective=True)),
                        ("pro_fixed", dict(prospective=True, fixed_discount=FIXED_DISCOUNT))]:
        p = out / f"two_head_{key}.npz"
        if p.exists():
            dim = 9 if key == "base" else 10
            m = TwoHeadDeepSets(input_dim=dim)
            z = np.load(p)
            m.load([z[f"p{i}"] for i in range(len(m.params))])
        else:
            print(f"Training two-head [{key}] ...", flush=True)
            m = train_two_head(train, epochs=args.epochs, seed=20260603,
                               verbose=True, **kwargs)
            m.save_npz(p)
        models[key] = m
    for key, kwargs in [("sel_pro", dict(prospective=True)),
                        ("sel_base", dict(prospective=False)),
                        ("sel_pro_fixed", dict(prospective=True, fixed_discount=FIXED_DISCOUNT))]:
        p = out / f"selective_{key}.npz"
        if p.exists():
            dim = 9 if key == "sel_base" else 10
            m = load_selective_npz(p, dim)
        else:
            print(f"Training selective [{key}] ...", flush=True)
            m = train_selective(train, anchor, epochs=args.epochs, seed=20260603,
                                verbose=True, **kwargs)
            save_selective_npz(m, p)
        models[key] = m
    (out / "anchor.json").write_text(json.dumps({"gold": list(anchor)}))
    return models, anchor


def train_models_ec(args, out):
    cfg = ExternalControlConfig(name="ec_train", drift_shift=-0.6, p_comparable=0.5)
    train = simulate_external_control_dataset(cfg, args.train_size, args.seed + 1)
    anchor = fit_eb_anchor([(q["y_control"], q["n_control"]) for q in train])
    print(f"EB anchor (EC train draw): Beta({anchor[0]:.3f}, {anchor[1]:.3f})", flush=True)
    models = {}
    for key, kwargs in [("base", dict(prospective=False)),
                        ("pro", dict(prospective=True)),
                        ("pro_fixed", dict(prospective=True, fixed_discount=FIXED_DISCOUNT))]:
        p = out / f"ec_two_head_{key}.npz"
        if p.exists():
            dim = 9 if key == "base" else 10
            m = TwoHeadDeepSets(input_dim=dim)
            z = np.load(p)
            m.load([z[f"p{i}"] for i in range(len(m.params))])
        else:
            print(f"Training EC two-head [{key}] ...", flush=True)
            m = train_two_head(train, epochs=args.epochs, seed=20260603,
                               verbose=True, **kwargs)
            m.save_npz(p)
        models[key] = m
    for key, kwargs in [("sel_pro", dict(prospective=True))]:
        p = out / f"ec_selective_{key}.npz"
        if p.exists():
            m = load_selective_npz(p, 10)
        else:
            print(f"Training EC selective [{key}] ...", flush=True)
            m = train_selective(train, anchor, epochs=args.epochs, seed=20260603,
                                verbose=True, **kwargs)
            save_selective_npz(m, p)
        models[key] = m
    (out / "anchor_ec.json").write_text(json.dumps({"ec": list(anchor)}))
    return models, anchor


# ---------------------------------------------------------------------------
# Method tables
# ---------------------------------------------------------------------------
def table_gold(models, anchor):
    base, pro, pro_fixed = models["base"], models["pro"], models["pro_fixed"]
    sp, sb, spf = models["sel_pro"], models["sel_base"], models["sel_pro_fixed"]
    t = {
        "weak_only": (lambda c, y0, n0: weak_only(c), None),
        "eb_only": (lambda c, y0, n0: eb_only(c, anchor), None),
        "rule": (lambda c, y0, n0: rule_components(c), None),
        "rule_sam": (lambda c, y0, n0: rule_components(c), "flat"),
        "two_head": (lambda c, y0, n0: two_head_prior(base, c), None),
        "two_head_pro": (lambda c, y0, n0: two_head_prior(pro, c, prospective=True), None),
        "two_head_pro_sam": (lambda c, y0, n0: two_head_prior(pro, c, prospective=True), "flat"),
        "two_head_pro_fixdisc": (lambda c, y0, n0: two_head_prior(
            pro_fixed, c, prospective=True, fixed_discount=FIXED_DISCOUNT), None),
        "selective_pro": (lambda c, y0, n0: selective_prior(sp, c, anchor, prospective=True), None),
        "selective_base": (lambda c, y0, n0: selective_prior(sb, c, anchor, prospective=False), None),
        "selective_pro_fixdisc": (lambda c, y0, n0: selective_prior(
            spf, c, anchor, prospective=True, fixed_discount=FIXED_DISCOUNT), None),
        "selective_pro_sam": (lambda c, y0, n0: selective_prior(sp, c, anchor, prospective=True), "anchored"),
        "robust_map_w0.5": (lambda c, y0, n0: robust_map(c, w=0.5), None),
        "robust_map_w0.9": (lambda c, y0, n0: robust_map(c, w=0.9), None),
        "power_prior": (lambda c, y0, n0: power_prior(c, a0=0.30), None),
        "uip_dirichlet": (lambda c, y0, n0: uip_dirichlet(c, y0, n0), None),
        "uip_js": (lambda c, y0, n0: uip_js(c, y0, n0), None),
        "pooling": (lambda c, y0, n0: pooling(c), None),
    }
    return t


def table_ec(models, anchor):
    base, pro, pro_fixed = models["base"], models["pro"], models["pro_fixed"]
    sp = models["sel_pro"]
    return {
        "internal_only": (lambda c, y0, n0: weak_only(c), None),
        "eb_only": (lambda c, y0, n0: eb_only(c, anchor), None),
        "rule": (lambda c, y0, n0: rule_components(c), None),
        "two_head": (lambda c, y0, n0: two_head_prior(base, c), None),
        "two_head_pro": (lambda c, y0, n0: two_head_prior(pro, c, prospective=True), None),
        "two_head_pro_sam": (lambda c, y0, n0: two_head_prior(pro, c, prospective=True), "flat"),
        "two_head_pro_fixdisc": (lambda c, y0, n0: two_head_prior(
            pro_fixed, c, prospective=True, fixed_discount=FIXED_DISCOUNT), None),
        "selective_pro": (lambda c, y0, n0: selective_prior(sp, c, anchor, prospective=True), None),
        "selective_pro_sam": (lambda c, y0, n0: selective_prior(sp, c, anchor, prospective=True), "anchored"),
        "robust_map_w0.5": (lambda c, y0, n0: robust_map(c, w=0.5), None),
        "power_prior": (lambda c, y0, n0: power_prior(c, a0=0.30), None),
        "uip_dirichlet": (lambda c, y0, n0: uip_dirichlet(c, y0, n0), None),
        "uip_js": (lambda c, y0, n0: uip_js(c, y0, n0), None),
        "pooling": (lambda c, y0, n0: pooling(c), None),
    }


def _sam(prior, kind, y0, n0):
    if kind == "flat":
        return apply_sam(prior, y0, n0)
    if kind == "anchored":
        return apply_sam_anchored(prior, y0, n0)
    return prior, 0.0


# ---------------------------------------------------------------------------
# Evaluation (dual labels)
# ---------------------------------------------------------------------------
def evaluate_gold(dataset, table, cfg):
    results = {name: [] for name in table}
    design_world_run = cfg.theta_query_fixed is not None
    for q in dataset:
        cands = q["candidates"]
        labels_p = np.array([c["borrowable_param"] for c in cands])
        labels_o = np.array([c["borrowable"] for c in cands])
        y0, n0, theta0 = q["y_query"], q["n_query"], q["theta_query"]
        oracle_p = oracle_weights_param(cands, theta0, cfg.epsilon)
        oracle_o = oracle_weights(cands, theta0, cfg.epsilon)
        for name, (build, sam_kind) in table.items():
            prior = build(cands, y0, n0)
            prior, sam_fired = _sam(prior, sam_kind, y0, n0)
            post_mean, lo, hi, p_calib, p_power = prior.posterior_summary(
                y0, n0, theta0, max(theta0 - POWER_MARGIN, 0.001))
            reject_design = np.nan
            if design_world_run:
                reject_design = float(
                    prior.posterior_prob_greater(y0, n0, THETA_NULL) > SUCCESS_THRESHOLD)
            tp, fp, fn, tn = classification_counts(prior.scores, labels_p, len(cands))
            osp, okl = weight_agreement(prior.scores, oracle_p)
            osp_o, _ = weight_agreement(prior.scores, oracle_o)
            esums, ecounts = ess_response(prior.discounts, cands, theta0)
            results[name].append({
                "bias": post_mean - theta0,
                "covered": lo <= theta0 <= hi,
                "width": hi - lo,
                "reject_null": p_calib > SUCCESS_THRESHOLD,
                "reject_power": p_power > SUCCESS_THRESHOLD,
                "calibration_hit": float(p_calib > SUCCESS_THRESHOLD),
                "reject_design": reject_design,
                "prior_ess": prior.prior_ess(),
                "hist_mass": prior.historical_mass(),
                "nll": -prior.log_predictive(y0, n0),
                "auc": roc_auc(prior.scores, labels_p),
                "prauc": pr_auc(prior.scores, labels_p),
                "auc_obs": roc_auc(prior.scores, labels_o),
                "oracle_spearman": osp, "oracle_kl": okl,
                "oracle_spearman_obs": osp_o,
                "ess_sums": esums, "ess_counts": ecounts,
                "sam_fired": sam_fired,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            })
    out = {}
    for name, rows in results.items():
        s = summarise_replicates(rows, cfg.n_candidates)
        aucs_o = np.array([r["auc_obs"] for r in rows if not np.isnan(r["auc_obs"])])
        osp_o = np.array([r["oracle_spearman_obs"] for r in rows
                          if not np.isnan(r.get("oracle_spearman_obs", np.nan))])
        s["roc_auc_obs"] = float(aucs_o.mean()) if aucs_o.size else float("nan")
        s["oracle_spearman_obs"] = float(osp_o.mean()) if osp_o.size else float("nan")
        out[name] = s
    return out


def evaluate_ec(dataset, table, cfg):
    results = {name: [] for name in table}
    for q in dataset:
        cands = q["candidates"]
        labels_p = np.array([c["borrowable_param"] for c in cands])
        labels_o = np.array([c["borrowable"] for c in cands])
        y_ctl, n_ctl = q["y_control"], q["n_control"]
        y_trt, n_trt = q["y_treatment"], q["n_treatment"]
        theta_ctl, true_eff = q["theta_control"], q["true_effect"]
        oracle_p = oracle_weights_param(cands, theta_ctl, cfg.epsilon)
        for name, (build, sam_kind) in table.items():
            prior = build(cands, y_ctl, n_ctl)
            prior, _fired = _sam(prior, sam_kind, y_ctl, n_ctl)
            eff, lo, hi, p_sup, ctl_mean = prior.treatment_effect(
                y_ctl, n_ctl, y_trt, n_trt)
            tp, fp, fn, tn = classification_counts(prior.scores, labels_p, len(cands))
            osp, _ = weight_agreement(prior.scores, oracle_p)
            results[name].append({
                "effect_bias": eff - true_eff,
                "control_bias": ctl_mean - theta_ctl,
                "reject": float(p_sup > SUCCESS_THRESHOLD),
                "covered": bool(lo <= true_eff <= hi),
                "prior_ess": prior.prior_ess(),
                "auc": roc_auc(prior.scores, labels_p),
                "prauc": pr_auc(prior.scores, labels_p),
                "auc_obs": roc_auc(prior.scores, labels_o),
                "oracle_spearman": osp,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            })
    out = {}
    for name, rows in results.items():
        s = ec_summarise(rows)
        aucs_o = np.array([r["auc_obs"] for r in rows if not np.isnan(r["auc_obs"])])
        s["roc_auc_obs"] = float(aucs_o.mean()) if aucs_o.size else float("nan")
        out[name] = s
    return out


def _dump(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["gold", "ec"], required=True)
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--train-size", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260720)
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.which == "gold":
        models, anchor = train_models_gold(args, out)
        table = table_gold(models, anchor)
        sparts = out / "scenario_parts"
        sparts.mkdir(exist_ok=True)
        diagnostics = {}
        for sname, cfg in scenarios().items():
            part = sparts / f"{sname}.csv"
            if part.exists():
                print(f"Scenario {sname} cached", flush=True)
                continue
            print(f"Scenario {sname} ...", flush=True)
            data = simulate_dataset(cfg, args.replicates, seed_for(sname, args.seed))
            diagnostics[sname] = label_summary(data)
            diagnostics[sname]["borrowable_param_rate"] = float(np.mean(
                [c["borrowable_param"] for q in data for c in q["candidates"]]))
            _dump([{"scenario": sname, "method": mname, **stats}
                   for mname, stats in evaluate_gold(data, table, cfg).items()], part)
            (out / "dgm_diagnostics.json").write_text(
                json.dumps(diagnostics, indent=2), encoding="utf-8")
        rows = []
        for sname in scenarios():
            part = sparts / f"{sname}.csv"
            if part.exists():
                with open(part, newline="", encoding="utf-8") as fh:
                    rows.extend(list(csv.DictReader(fh)))
        _dump(rows, out / "scenario_results.csv")

        parts = out / "design_parts"
        parts.mkdir(exist_ok=True)
        for world in ("null", "alt"):
            for shift in CONFLICT_GRID:
                part = parts / f"{world}_{shift}.csv"
                if part.exists():
                    print(f"Design {world} shift={shift} cached", flush=True)
                    continue
                cfg = design_world(world, shift)
                print(f"Design {world} shift={shift} ...", flush=True)
                data = simulate_dataset(cfg, args.replicates, seed_for(cfg.name, args.seed))
                _dump([{"world": world, "conflict_shift": shift,
                        "theta_true": cfg.theta_query_fixed, "method": mname, **stats}
                       for mname, stats in evaluate_gold(data, table, cfg).items()], part)
        design_rows = []
        for world in ("null", "alt"):
            for shift in CONFLICT_GRID:
                part = parts / f"{world}_{shift}.csv"
                if part.exists():
                    with open(part, newline="", encoding="utf-8") as fh:
                        design_rows.extend(list(csv.DictReader(fh)))
        _dump(design_rows, out / "design_worlds.csv")

    else:
        models, anchor = train_models_ec(args, out)
        table = table_ec(models, anchor)
        parts = out / "ec_parts"
        parts.mkdir(exist_ok=True)
        for world, effect in (("null", EFFECT_NULL), ("alt", EFFECT_ALT)):
            for drift in DRIFT_GRID:
                part = parts / f"{world}_{drift}.csv"
                if part.exists():
                    print(f"EC {world} drift={drift} cached", flush=True)
                    continue
                cfg = ExternalControlConfig(
                    name=f"EC_{world}_drift{drift}", treatment_effect=effect,
                    drift_shift=drift)
                print(f"EC {world} drift={drift} ...", flush=True)
                data = simulate_external_control_dataset(
                    cfg, args.replicates, seed_for(cfg.name, args.seed))
                _dump([{"world": world, "drift_shift": drift,
                        "treatment_effect": effect, "method": mname, **stats}
                       for mname, stats in evaluate_ec(data, table, cfg).items()], part)
        for p_comp in (0.25, 0.50, 0.75):
            part = parts / f"pcomp_{p_comp}.csv"
            if part.exists():
                continue
            cfg = ExternalControlConfig(
                name=f"EC_pcomp{p_comp}", treatment_effect=EFFECT_NULL,
                drift_shift=-0.8, p_comparable=p_comp)
            print(f"EC p_comparable={p_comp} ...", flush=True)
            data = simulate_external_control_dataset(
                cfg, args.replicates, seed_for(cfg.name, args.seed))
            _dump([{"p_comparable": p_comp, "method": mname, **stats}
                   for mname, stats in evaluate_ec(data, table, cfg).items()], part)
        rows, rows2 = [], []
        for p in sorted(parts.glob("*.csv")):
            with open(p, newline="", encoding="utf-8") as fh:
                (rows2 if p.stem.startswith("pcomp") else rows).extend(
                    list(csv.DictReader(fh)))
        _dump(rows, out / "external_control.csv")
        _dump(rows2, out / "external_control_pcomp.csv")

    (out / f"run_config_{args.which}.json").write_text(json.dumps({
        "replicates": args.replicates, "train_size": args.train_size,
        "epochs": args.epochs, "seed": args.seed,
        "labels": "primary=|theta_true - theta0|<=eps; secondary(_obs)=|theta_obs - theta0|<=eps",
    }, indent=2), encoding="utf-8")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
