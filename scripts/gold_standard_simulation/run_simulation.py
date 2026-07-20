"""Driver for the gold-standard borrowability simulation (ADEMP structured).

Aims                  : (i) can a borrowing method identify which historical
                        trials are genuinely exchangeable; (ii) does that
                        translate into better operating characteristics for
                        theta_0; (iii) does the prospective set-level conflict
                        signal buy anything over features alone.
Data-generating mech. : dgm.py (NSCLC-anchored, non-linear, deliberately NOT
                        the fitted model)
Estimand              : theta_0, the query trial's true response rate; the
                        per-donor borrowable/not gold-standard label; and the
                        oracle allocation over donors
Methods               : see method_table() -- no-borrowing, rule-based,
                        two-head variants, robust-MAP, power prior, UIP, pooling
Performance measures  : metrics.py (OC + discrimination + oracle agreement,
                        all with MCSE)

Two families of scenario are run:

*   Estimation scenarios (S1-S5) leave theta_0 free and measure bias, RMSE,
    coverage, borrowing discrimination and oracle agreement.
*   Design worlds pin theta_0 to a fixed design benchmark. The null world sets
    theta_0 = THETA_NULL, so the rejection rate IS the type I error; the
    alternative world sets theta_0 = THETA_NULL + DESIGN_DELTA, so the rejection
    rate is power. Both are swept over increasing prior-data conflict.

Usage:
    python3 run_simulation.py --replicates 1000 --output ../../artifacts/gold_standard_simulation
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

from dgm import SimConfig, label_summary, simulate_dataset  # noqa: E402
from methods import (  # noqa: E402
    TwoHeadDeepSets, apply_sam, pooling, power_prior, robust_map,
    rule_components, train_two_head, two_head_prior, uip_dirichlet, uip_js,
    weak_only,
)
from metrics import (  # noqa: E402
    DESIGN_DELTA, POWER_MARGIN, SUCCESS_THRESHOLD, THETA_NULL,
    classification_counts, ess_response, oracle_weights, pr_auc, roc_auc,
    summarise_replicates, weight_agreement,
)

FIXED_DISCOUNT = 0.35


def seed_for(name: str, base: int) -> int:
    """Deterministic per-scenario seed.

    Python's built-in ``hash`` is randomised per process unless PYTHONHASHSEED
    is set, so using it here would make runs irreproducible. CRC32 is stable
    across processes and platforms.
    """
    return base + (zlib.crc32(name.encode("utf-8")) % 100000)


def scenarios() -> dict[str, SimConfig]:
    """Estimation scenarios: theta_0 free.

    S1 is deliberately favourable to the simple pooled competitors -- almost
    every donor is exchangeable, so full pooling should win on RMSE. Including
    it guards against cherry-picking (手稿修改方案 v3, section 3.3).
    """
    return {
        "S1_exchangeable": SimConfig(
            name="S1_exchangeable", p_same_line=0.95, p_same_surface=0.60,
            conflict_shift=0.0, p_endpoint_incompatible=0.05, p_poor_result_quality=0.05),
        "S2_trap_heavy": SimConfig(
            name="S2_trap_heavy", p_same_line=0.20, p_same_surface=0.85,
            conflict_shift=0.0),
        "S3_hidden_gem": SimConfig(
            name="S3_hidden_gem", p_same_line=0.70, p_same_surface=0.20,
            conflict_shift=0.0),
        "S4_conflict": SimConfig(
            name="S4_conflict", p_same_line=0.45, p_same_surface=0.55,
            conflict_shift=1.0),
        "S5_heterogeneous": SimConfig(
            name="S5_heterogeneous", p_same_line=0.45, p_same_surface=0.55,
            tau=0.70),
    }


CONFLICT_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


def design_world(world: str, shift: float) -> SimConfig:
    """Null / alternative design world at a given conflict level."""
    theta = THETA_NULL if world == "null" else THETA_NULL + DESIGN_DELTA
    return SimConfig(
        name=f"{world}_shift{shift}", conflict_shift=shift, theta_query_fixed=theta)


def method_table(models):
    """name -> (prior builder taking (cands, y0, n0), uses SAM)."""
    base, pro, pro_fixed = models["base"], models["pro"], models["pro_fixed"]
    return {
        "weak_only": (lambda c, y0, n0: weak_only(c), False),
        "rule": (lambda c, y0, n0: rule_components(c), False),
        "rule_sam": (lambda c, y0, n0: rule_components(c), True),
        # Contribution 2 ablation: identical model, with and without the
        # prospective set-level conflict feature.
        "two_head": (lambda c, y0, n0: two_head_prior(base, c), False),
        "two_head_pro": (
            lambda c, y0, n0: two_head_prior(pro, c, prospective=True), False),
        "two_head_pro_sam": (
            lambda c, y0, n0: two_head_prior(pro, c, prospective=True), True),
        # Contribution 3 ablation: learned per-source discount vs a constant.
        "two_head_pro_fixdisc": (
            lambda c, y0, n0: two_head_prior(pro_fixed, c, prospective=True,
                                             fixed_discount=FIXED_DISCOUNT), False),
        "robust_map_w0.5": (lambda c, y0, n0: robust_map(c, w=0.5), False),
        "robust_map_w0.9": (lambda c, y0, n0: robust_map(c, w=0.9), False),
        "power_prior": (lambda c, y0, n0: power_prior(c, a0=0.30), False),
        # Closed-form adaptive competitors (Jin & Yin 2021).
        "uip_dirichlet": (lambda c, y0, n0: uip_dirichlet(c, y0, n0), False),
        "uip_js": (lambda c, y0, n0: uip_js(c, y0, n0), False),
        "pooling": (lambda c, y0, n0: pooling(c), False),
    }


def evaluate(dataset, models, cfg):
    table = method_table(models)
    results = {name: [] for name in table}
    design_world_run = cfg.theta_query_fixed is not None
    for q in dataset:
        cands = q["candidates"]
        labels = np.array([c["borrowable"] for c in cands])
        y0, n0, theta0 = q["y_query"], q["n_query"], q["theta_query"]
        oracle = oracle_weights(cands, theta0, cfg.epsilon)
        for name, (build, use_sam) in table.items():
            prior = build(cands, y0, n0)
            sam_fired = 0.0
            if use_sam:
                prior, sam_fired = apply_sam(prior, y0, n0)

            # Estimation summaries plus the legacy calibration diagnostic.
            post_mean, lo, hi, p_calib, p_power = prior.posterior_summary(
                y0, n0, theta0, max(theta0 - POWER_MARGIN, 0.001))

            # Decision against the FIXED design null. Only meaningful when the
            # world pins theta_0, which is what makes the rate a type I error
            # (null world) or power (alternative world).
            reject_design = np.nan
            if design_world_run:
                reject_design = float(
                    prior.posterior_prob_greater(y0, n0, THETA_NULL) > SUCCESS_THRESHOLD)

            tp, fp, fn, tn = classification_counts(prior.scores, labels, len(cands))
            osp, okl = weight_agreement(prior.scores, oracle)
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
                "auc": roc_auc(prior.scores, labels),
                "prauc": pr_auc(prior.scores, labels),
                "oracle_spearman": osp, "oracle_kl": okl,
                "ess_sums": esums, "ess_counts": ecounts,
                "sam_fired": sam_fired,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            })
    return {name: summarise_replicates(rows, cfg.n_candidates)
            for name, rows in results.items()}


def _dump(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def train_all(args, out):
    """Train the three two-head variants on an independent training draw."""
    train = simulate_dataset(SimConfig(name="train"), args.train_size, args.seed + 1)
    models = {}
    for key, kwargs in [
        ("base", dict(prospective=False)),
        ("pro", dict(prospective=True)),
        ("pro_fixed", dict(prospective=True, fixed_discount=FIXED_DISCOUNT)),
    ]:
        print(f"Training two-head [{key}] ...", flush=True)
        m = train_two_head(train, epochs=args.epochs, seed=20260603,
                           verbose=True, **kwargs)
        m.save_npz(out / f"two_head_{key}.npz")
        models[key] = m
    return models


def load_all(out):
    models = {}
    for key, dim in [("base", 9), ("pro", 10), ("pro_fixed", 10)]:
        model = TwoHeadDeepSets(input_dim=dim)
        z = np.load(out / f"two_head_{key}.npz")
        model.load([z[f"p{i}"] for i in range(len(model.params))])
        models[key] = model
    return models


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "scenarios", "design", "all"],
                    default="all")
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--design-replicates", type=int, default=None,
                    help="replicates for the null/alternative worlds "
                         "(defaults to --replicates)")
    ap.add_argument("--train-size", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260720)
    ap.add_argument("--output", type=str, default="../../artifacts/gold_standard_simulation")
    ap.add_argument("--cell", type=str, default=None,
                    help="run only this design cell, e.g. null_0.5")
    ap.add_argument("--no-combine", action="store_true",
                    help="skip writing the combined CSV (for parallel workers)")
    args = ap.parse_args()

    out = Path(args.output)
    if not out.is_absolute():
        out = (Path(__file__).resolve().parent / out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    n_design = args.design_replicates or args.replicates

    models = train_all(args, out) if args.mode in ("train", "all") else load_all(out)
    if args.mode == "train":
        return 0

    # ---- estimation scenarios ------------------------------------------
    # Checkpointed per scenario so an interrupted run resumes.
    if args.mode in ("scenarios", "all"):
        sparts = out / "scenario_parts"
        sparts.mkdir(exist_ok=True)
        diagnostics = {}
        dpath = out / "dgm_diagnostics.json"
        if dpath.exists():
            diagnostics = json.loads(dpath.read_text(encoding="utf-8"))
        for sname, cfg in scenarios().items():
            part = sparts / f"{sname}.csv"
            if part.exists():
                print(f"Scenario {sname} cached", flush=True)
                continue
            print(f"Scenario {sname} ...", flush=True)
            data = simulate_dataset(cfg, args.replicates, seed_for(sname, args.seed))
            diagnostics[sname] = label_summary(data)
            _dump([{"scenario": sname, "method": mname, **stats}
                   for mname, stats in evaluate(data, models, cfg).items()], part)
            dpath.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        rows = []
        for sname in scenarios():
            part = sparts / f"{sname}.csv"
            if part.exists():
                with open(part, newline="", encoding="utf-8") as fh:
                    rows.extend(list(csv.DictReader(fh)))
        _dump(rows, out / "scenario_results.csv")

    # ---- design worlds: type I error and power under conflict ----------
    # Each world-shift cell is checkpointed to its own part file so that an
    # interrupted run resumes instead of restarting.
    if args.mode in ("design", "all"):
        parts = out / "design_parts"
        parts.mkdir(exist_ok=True)
        for world in ("null", "alt"):
            for shift in CONFLICT_GRID:
                # --cell lets one process own exactly one world-shift cell, so
                # several cells can be run in parallel without racing on the
                # same checkpoint file.
                if args.cell and args.cell != f"{world}_{shift}":
                    continue
                part = parts / f"{world}_{shift}.csv"
                if part.exists():
                    print(f"Design world {world} shift={shift} cached", flush=True)
                    continue
                cfg = design_world(world, shift)
                print(f"Design world {world} shift={shift} ...", flush=True)
                data = simulate_dataset(cfg, n_design, seed_for(cfg.name, args.seed))
                cell = [{"world": world, "conflict_shift": shift,
                         "theta_true": cfg.theta_query_fixed, "method": mname, **stats}
                        for mname, stats in evaluate(data, models, cfg).items()]
                _dump(cell, part)
        if not args.no_combine:
            design_rows = []
            for world in ("null", "alt"):
                for shift in CONFLICT_GRID:
                    part = parts / f"{world}_{shift}.csv"
                    if part.exists():
                        with open(part, newline="", encoding="utf-8") as fh:
                            design_rows.extend(list(csv.DictReader(fh)))
            _dump(design_rows, out / "design_worlds.csv")

    (out / "run_config.json").write_text(json.dumps({
        "replicates": args.replicates, "design_replicates": n_design,
        "train_size": args.train_size, "epochs": args.epochs, "seed": args.seed,
        "success_threshold": SUCCESS_THRESHOLD, "theta_null": THETA_NULL,
        "design_delta": DESIGN_DELTA, "fixed_discount": FIXED_DISCOUNT,
        "conflict_grid": CONFLICT_GRID,
    }, indent=2), encoding="utf-8")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
