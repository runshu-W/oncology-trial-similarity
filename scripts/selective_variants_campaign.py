#!/usr/bin/env python3
"""Round-2 review campaign: 2x2 ablation, learning-curve control, multi-seed.

Produces artifacts_round2/:
  models/                       trained variant checkpoints
  swap_check.json               two_head recompute verification + EB-anchor swap rows
  ablation_2x2.json             full 2x2 (anchor x total-borrowing) paired comparison
  learning_curve.json           fold-2020/2021/2022 models all scored on window 3 (+w2)
  multiseed.json                per-seed deltas + seed-ensemble prior results
  campaign_summary.md           human-readable digest

Design constraints honoured:
  - primary split ALWAYS from deterministic_split_indices(n, 0.8, 20260603);
    the training seed varies independently (reviewer point 4).
  - forward windows scored once, by the model trained at their own cutoff;
    learning-curve rows are the deliberate exception and are labelled as such.
  - paired bootstrap B=4000, rng seed 20260817; Bonferroni 98.33% CIs reported
    alongside 95% for the three per-window tests (reviewer point 5).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

import train_retrospective_lambda_model as LT  # noqa: E402
import two_head_selective as TS  # noqa: E402
from evaluate_retrospective_lambda_model import deterministic_split_indices  # noqa: E402

EXAMPLES = ROOT / "artifacts_rerun" / "examples_unitfix_with_true_dates.jsonl"
REWRITE = ROOT / "artifacts_pathfinding" / "rewrite_numbers_perexample.json"
SEL_DIR = ROOT / "artifacts_pathfinding" / "selective"
WITH_MODEL_PRIMARY = ROOT / "artifacts_rerun" / "examples_unitfix_with_model.jsonl"
WITH_MODEL_FOLD = {
    c: ROOT / "artifacts_pathfinding" / f"with_model_fold_{c}.jsonl"
    for c in ("2020-12-31", "2021-12-31", "2022-12-31")
}
OUT = ROOT / "artifacts_round2"
MODELS = OUT / "models"
CUTOFFS = ("2020-12-31", "2021-12-31", "2022-12-31")
SPLIT_SEED = 20260603
SEEDS = [20260603, 11, 907, 4242, 77777]
BOOT_B = 4000
BOOT_SEED = 20260817

log_path = None


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if log_path is not None:
        with open(log_path, "a") as f:
            f.write(line + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def sort_date(ex: dict) -> str:
    return TS.sort_date(ex)


def window_of(date: str) -> str | None:
    if not date:
        return None
    if date <= "2020-12-31":
        return None
    if date <= "2021-12-31":
        return "w1"
    if date <= "2022-12-31":
        return "w2"
    return "w3"


def bb_log_pmf(y: float, n: float, a: float, b: float) -> float:
    y, n = float(y), float(n)
    return (
        math.lgamma(n + 1) - math.lgamma(y + 1) - math.lgamma(n - y + 1)
        + math.lgamma(y + a) + math.lgamma(n - y + b) - math.lgamma(n + a + b)
        + math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    )


def two_head_nll(ex: dict, anchor: tuple[float, float]) -> float:
    """Fixed-budget two-head prior NLL from stored per-candidate model outputs.

    Replicates run_borrowing_baseline_comparison semantics: lambda_0 fixed at
    the stored value (0.2), candidate lambdas = stored lambda_model normalised
    to sum to 1 - lambda_0 over gate-positive candidates, discounts clipped to
    [0, 1], component Beta(1 + a*y, 1 + a*(n - y)).
    """
    y0 = float(ex["query"]["count"])
    n0 = float(ex["query"]["denominator"])
    lam0 = float(ex.get("lambda_0", 0.2))
    comps = ex["components"]
    lams = np.array([float(c.get("lambda_model", 0.0)) for c in comps])
    gates = np.array([float(c.get("gate", 0.0)) for c in comps])
    lams = np.where(gates > 0, lams, 0.0)
    s = lams.sum()
    terms = [math.log(max(lam0, 1e-300)) + bb_log_pmf(y0, n0, anchor[0], anchor[1])]
    if s > 0:
        lams = lams * (1.0 - lam0) / s
        for c, lam in zip(comps, lams):
            if lam <= 0:
                continue
            a = min(max(float(c.get("discount_model", 0.0)), 0.0), 1.0)
            yi, ni = float(c["count"]), float(c["denominator"])
            terms.append(math.log(lam) + bb_log_pmf(y0, n0, 1 + a * yi, 1 + a * (ni - yi)))
    else:
        # no gated candidates: all mass on the weak component
        terms = [bb_log_pmf(y0, n0, anchor[0], anchor[1])]
    m = max(terms)
    return -(m + math.log(sum(math.exp(t - m) for t in terms)))


def boot_ci(diffs: np.ndarray, rng: np.random.Generator, b: int = BOOT_B,
            levels=(0.95, 0.98333)) -> dict:
    n = len(diffs)
    idx = rng.integers(0, n, size=(b, n))
    means = diffs[idx].mean(axis=1)
    out = {"mean": float(diffs.mean()), "n": int(n),
           "win": float((diffs < 0).mean())}
    for lv in levels:
        lo, hi = np.quantile(means, [(1 - lv) / 2, 1 - (1 - lv) / 2])
        key = "ci95" if abs(lv - 0.95) < 1e-9 else f"ci{lv:.5f}".rstrip("0")
        out[key] = [float(lo), float(hi)]
    return out


def train_variant(examples, train_idx, eval_idx, anchor, seed) -> torch.nn.Module:
    res = TS.train_selective(
        examples, train_idx, eval_idx, anchor,
        epochs=100, learning_rate=0.01, hidden_dim=16, seed=seed,
    )
    return res["model"]


def eval_rows(model, examples, idxs, anchor) -> dict[int, dict]:
    out = {}
    for i in idxs:
        d = TS.selective_details(model, examples[i], anchor)
        out[i] = {"nll": d["nll"], "lambda0": d["lambda0"], "mass": d["borrowed_mass"]}
    return out


def main() -> None:
    global log_path
    OUT.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    log_path = OUT / "campaign_log.txt"

    examples = read_jsonl(EXAMPLES)
    rw = json.load(open(REWRITE))
    n = len(examples)
    log(f"loaded {n} examples")

    # --- index verification against published rewrite rows -------------------
    train_idx, test_idx = deterministic_split_indices(n, train_fraction=0.8, seed=SPLIT_SEED)
    assert set(test_idx) == {r["i"] for r in rw["test"]}, "test index mismatch"
    win = {i: window_of(sort_date(ex)) for i, ex in enumerate(examples)}
    fwd_idx = {w: sorted(i for i, v in win.items() if v == w) for w in ("w1", "w2", "w3")}
    assert [len(fwd_idx[w]) for w in ("w1", "w2", "w3")] == [181, 204, 390], "window sizes"
    assert {r["i"] for r in rw["fwd"]} == set().union(*fwd_idx.values()), "fwd index mismatch"
    fold_train_idx = {
        c: [i for i, ex in enumerate(examples) if sort_date(ex) and sort_date(ex) <= c]
        for c in CUTOFFS
    }
    assert [len(fold_train_idx[c]) for c in CUTOFFS] == [632, 813, 1017], "fold train sizes"
    log("index verification OK (test 281; windows 181/204/390; folds 632/813/1017)")

    # NOTE: a query can appear in BOTH scopes (test-split member whose date
    # falls in a forward window) with different scores (primary model +
    # train-split EB vs fold model + past-only EB). Scope maps stay separate.
    eb_test = {r["i"]: r["eb"] for r in rw["test"]}
    eb_fwd = {r["i"]: r["eb"] for r in rw["fwd"]}
    sel_test = {r["i"]: r["selective"] for r in rw["test"]}
    sel_fwd = {r["i"]: r["selective"] for r in rw["fwd"]}
    old_test = {r["i"]: r["two_head_old"] for r in rw["test"]}
    old_fwd = {r["i"]: r["two_head_old"] for r in rw["fwd"]}
    overlap = len(set(test_idx) & set(eb_fwd))
    log(f"scope overlap (test members inside forward windows): {overlap}")

    # --- sanity: reproduce published selective NLLs from stored artifacts ----
    prim_model, prim_anchor = TS.load_selective_artifact(SEL_DIR / "selective_model_primary.pt")
    check = eval_rows(prim_model, examples, test_idx[:25], prim_anchor)
    worst = max(abs(check[i]["nll"] - sel_test[i]) for i in check)
    assert worst < 1e-4, f"selective replication drift {worst}"
    log(f"selective primary replication OK (max drift {worst:.2e})")

    fold_models = {}
    for c in CUTOFFS:
        m, a = TS.load_selective_artifact(SEL_DIR / f"selective_model_fold_{c}.pt")
        fold_models[c] = (m, a)

    # --- swap check: recompute two_head flat, then EB-anchor swap ------------
    log("stage: swap verification + EB/fixed rows")
    wm_primary = {ex["query_nct_id"]: ex for ex in read_jsonl(WITH_MODEL_PRIMARY)}
    wm_fold = {c: {ex["query_nct_id"]: ex for ex in read_jsonl(WITH_MODEL_FOLD[c])} for c in CUTOFFS}
    id_of = {i: examples[i]["query_nct_id"] for i in range(n)}

    def wm_row(i: int, scope: str) -> tuple[dict, tuple[float, float]]:
        if scope == "test":
            return wm_primary[id_of[i]], prim_anchor
        cut = {"w1": CUTOFFS[0], "w2": CUTOFFS[1], "w3": CUTOFFS[2]}[win[i]]
        return wm_fold[cut][id_of[i]], fold_models[cut][1]

    swap = {"flat_recompute_maxdrift": 0.0, "rows": {"test": {}, "fwd": {}}}
    fwd_all_pre = [i for w in ("w1", "w2", "w3") for i in fwd_idx[w]]
    for scope, idxs, ref in (("test", list(test_idx), old_test),
                             ("fwd", fwd_all_pre, old_fwd)):
        for i in idxs:
            ex_wm, eb_anchor = wm_row(i, scope)
            flat = two_head_nll(ex_wm, (1.0, 1.0))
            drift = abs(flat - ref[i])
            swap["flat_recompute_maxdrift"] = max(swap["flat_recompute_maxdrift"], drift)
            swap["rows"][scope][str(i)] = {
                "flat": flat, "ebfixed": two_head_nll(ex_wm, eb_anchor)}
    log(f"two_head flat recompute max drift = {swap['flat_recompute_maxdrift']:.3e}")
    json.dump(swap, open(OUT / "swap_check.json", "w"))
    if swap["flat_recompute_maxdrift"] > 5e-3:
        log("WARNING: two_head recompute drift above tolerance; EB/fixed cell suspect")

    # --- training campaign ----------------------------------------------------
    fits: dict[tuple, dict[int, dict]] = {}

    def anchor_for(variant: str, idxs: list[int]) -> tuple[float, float]:
        if variant == "flat":
            return (1.0, 1.0)
        return TS.fit_eb_anchor(
            [(examples[i]["query"]["count"], examples[i]["query"]["denominator"]) for i in idxs]
        )

    todo = []
    for variant in ("eb", "flat"):
        for seed in SEEDS:
            if variant == "eb" and seed == SPLIT_SEED:
                continue  # canonical fits already exist
            todo.append((variant, seed))
    log(f"training campaign: {len(todo)} (variant,seed) pairs x 4 fits each")

    for variant, seed in todo:
        t0 = time.time()
        anc = anchor_for(variant, train_idx)
        m = train_variant(examples, train_idx, test_idx, anc, seed)
        torch.save(m.state_dict(), MODELS / f"{variant}_s{seed}_primary.pt")
        fits[(variant, seed, "primary")] = eval_rows(m, examples, test_idx, anc)
        for c in CUTOFFS:
            anc_c = anchor_for(variant, fold_train_idx[c])
            ev = [i for i, ex in enumerate(examples) if sort_date(ex) and sort_date(ex) > c]
            mc = train_variant(examples, fold_train_idx[c], ev, anc_c, seed)
            torch.save(mc.state_dict(), MODELS / f"{variant}_s{seed}_fold_{c}.pt")
            fits[(variant, seed, f"fold_{c}")] = eval_rows(mc, examples, ev, anc_c)
        log(f"trained {variant} seed {seed} (4 fits) in {time.time()-t0:.0f}s")

    # canonical EB fits reuse published artifacts
    fits[("eb", SPLIT_SEED, "primary")] = eval_rows(prim_model, examples, test_idx, prim_anchor)
    for c in CUTOFFS:
        m, a = fold_models[c]
        ev = [i for i, ex in enumerate(examples) if sort_date(ex) and sort_date(ex) > c]
        fits[("eb", SPLIT_SEED, f"fold_{c}")] = eval_rows(m, examples, ev, a)
    log("canonical EB fits evaluated from stored artifacts")

    def pooled_fwd(variant: str, seed: int) -> dict[int, float]:
        out = {}
        for w, c in zip(("w1", "w2", "w3"), CUTOFFS):
            rows = fits[(variant, seed, f"fold_{c}")]
            for i in fwd_idx[w]:
                out[i] = rows[i]["nll"]
        return out

    rng = np.random.default_rng(BOOT_SEED)

    def paired(a: dict[int, float], b: dict[int, float], idxs) -> dict:
        d = np.array([a[i] - b[i] for i in idxs])
        r = boot_ci(d, rng)
        r["mean_a"] = float(np.mean([a[i] for i in idxs]))
        r["mean_b"] = float(np.mean([b[i] for i in idxs]))
        return r

    # --- 2x2 ablation (seed 20260603) ----------------------------------------
    log("stage: 2x2 ablation assembly")
    cells_test: dict[str, dict[int, float]] = {
        "flat_fixed": dict(old_test),
        "eb_fixed": {i: swap["rows"]["test"][str(i)]["ebfixed"] for i in test_idx},
        "flat_learned": {i: fits[("flat", SPLIT_SEED, "primary")][i]["nll"] for i in test_idx},
        "eb_learned": dict(sel_test),
    }
    fwd_all = [i for w in ("w1", "w2", "w3") for i in fwd_idx[w]]
    cells_fwd: dict[str, dict[int, float]] = {
        "flat_fixed": dict(old_fwd),
        "eb_fixed": {i: swap["rows"]["fwd"][str(i)]["ebfixed"] for i in fwd_all},
        "flat_learned": pooled_fwd("flat", SPLIT_SEED),
        "eb_learned": dict(sel_fwd),
    }
    ablation = {"note": "paired vs intercept-only EB reference; negative = better than EB",
                "test": {}, "fwd": {}, "vs_flat_fixed": {"test": {}, "fwd": {}}}
    for cell in cells_test:
        ablation["test"][cell] = paired(cells_test[cell], eb_test, test_idx)
        ablation["fwd"][cell] = paired(cells_fwd[cell], eb_fwd, fwd_all)
        ablation["vs_flat_fixed"]["test"][cell] = paired(cells_test[cell], cells_test["flat_fixed"], test_idx)
        ablation["vs_flat_fixed"]["fwd"][cell] = paired(cells_fwd[cell], cells_fwd["flat_fixed"], fwd_all)
    json.dump(ablation, open(OUT / "ablation_2x2.json", "w"), indent=1)
    log("ablation_2x2.json written")

    # --- learning-curve control (fixed composition: window 3) ----------------
    log("stage: learning-curve control")
    lc = {"note": "all three fold models scored on the SAME window-3 queries "
                  "(390); composition fixed, training size varies 632/813/1017. "
                  "w2 rows: fold-2020 vs fold-2021 on the same 204 queries.",
          "w3": {}, "w2": {}, "pairwise_w3": {}}
    w3_rows = {}
    for c in CUTOFFS:
        m, a = fold_models[c]
        rows = (fits[("eb", SPLIT_SEED, f"fold_{c}")]
                if c == CUTOFFS[2] else eval_rows(m, examples, fwd_idx["w3"], a))
        w3_rows[c] = {i: rows[i]["nll"] for i in fwd_idx["w3"]}
        lc["w3"][c] = paired(w3_rows[c], eb_fwd, fwd_idx["w3"])
    for c in CUTOFFS[:2]:
        lc["pairwise_w3"][f"{CUTOFFS[2]}_minus_{c}"] = paired(
            w3_rows[CUTOFFS[2]], w3_rows[c], fwd_idx["w3"])
    w2_rows = {}
    for c in CUTOFFS[:2]:
        m, a = fold_models[c]
        rows = (fits[("eb", SPLIT_SEED, f"fold_{c}")]
                if c == CUTOFFS[1] else eval_rows(m, examples, fwd_idx["w2"], a))
        w2_rows[c] = {i: rows[i]["nll"] for i in fwd_idx["w2"]}
        lc["w2"][c] = paired(w2_rows[c], eb_fwd, fwd_idx["w2"])
    lc["pairwise_w2"] = {f"{CUTOFFS[1]}_minus_{CUTOFFS[0]}": paired(
        w2_rows[CUTOFFS[1]], w2_rows[CUTOFFS[0]], fwd_idx["w2"])}
    json.dump(lc, open(OUT / "learning_curve.json", "w"), indent=1)
    log("learning_curve.json written")

    # --- multi-seed + ensemble ------------------------------------------------
    log("stage: multi-seed assembly")
    ms = {"seeds": SEEDS, "per_seed": {}, "ensemble": {}, "bonferroni_note":
          "per-window CIs also given at 98.33% (Bonferroni for 3 windows)"}
    for variant in ("eb", "flat"):
        ms["per_seed"][variant] = {}
        for seed in SEEDS:
            t = {i: fits[(variant, seed, "primary")][i]["nll"] for i in test_idx}
            f = pooled_fwd(variant, seed)
            entry = {
                "test_vs_eb": paired(t, eb_test, test_idx),
                "fwd_vs_eb": paired(f, eb_fwd, fwd_all),
                "fwd_vs_flat_fixed": paired(f, old_fwd, fwd_all),
                "w3_vs_eb": paired({i: f[i] for i in fwd_idx["w3"]}, eb_fwd, fwd_idx["w3"]),
                "mass_quartiles_fwd": [float(q) for q in np.quantile(
                    [fits[(variant, seed, f"fold_{c}")][i]["mass"]
                     for w, c in zip(("w1", "w2", "w3"), CUTOFFS) for i in fwd_idx[w]],
                    [0.25, 0.5, 0.75])],
            }
            ms["per_seed"][variant][str(seed)] = entry
        # seed-ensemble prior: predictive pmf averaged over seeds
        ens_t, ens_f = {}, {}
        for i in test_idx:
            ens_t[i] = -math.log(np.mean(
                [math.exp(-fits[(variant, s, "primary")][i]["nll"]) for s in SEEDS]))
        for w, c in zip(("w1", "w2", "w3"), CUTOFFS):
            for i in fwd_idx[w]:
                ens_f[i] = -math.log(np.mean(
                    [math.exp(-fits[(variant, s, f"fold_{c}")][i]["nll"]) for s in SEEDS]))
        ms["ensemble"][variant] = {
            "test_vs_eb": paired(ens_t, eb_test, test_idx),
            "fwd_vs_eb": paired(ens_f, eb_fwd, fwd_all),
            "w1_vs_eb": paired({i: ens_f[i] for i in fwd_idx["w1"]}, eb_fwd, fwd_idx["w1"]),
            "w2_vs_eb": paired({i: ens_f[i] for i in fwd_idx["w2"]}, eb_fwd, fwd_idx["w2"]),
            "w3_vs_eb": paired({i: ens_f[i] for i in fwd_idx["w3"]}, eb_fwd, fwd_idx["w3"]),
            "fwd_vs_flat_fixed": paired(ens_f, old_fwd, fwd_all),
        }
    # persist per-example ensemble rows for downstream triage analysis
    json.dump({"ens_eb_test": {str(i): v for i, v in ens_t.items()}},
              open(OUT / "_ens_flat_last.json", "w"))
    per_ex = {}
    for variant in ("eb",):
        for seed in SEEDS:
            for c in CUTOFFS:
                key = f"{variant}_s{seed}_fold_{c}"
                per_ex[key] = {str(i): fits[(variant, seed, f"fold_{c}")][i]
                               for i in fits[(variant, seed, f"fold_{c}")]}
    json.dump(per_ex, open(OUT / "multiseed_perexample.json", "w"))
    json.dump(ms, open(OUT / "multiseed.json", "w"), indent=1)
    log("multiseed.json written")

    # --- digest ---------------------------------------------------------------
    def fmt(r):
        return (f"mean_delta={r['mean']:+.4f} CI95=[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]"
                f" win={r['win']:.2f}")

    with open(OUT / "campaign_summary.md", "w") as f:
        f.write("# Round-2 campaign digest\n\n## 2x2 ablation vs EB (seed 20260603)\n\n")
        for scope in ("test", "fwd"):
            for cell in ("flat_fixed", "eb_fixed", "flat_learned", "eb_learned"):
                f.write(f"- {scope} {cell}: {fmt(ablation[scope][cell])}\n")
        f.write("\n## Learning curve on fixed window 3\n\n")
        for c in CUTOFFS:
            f.write(f"- model@{c} vs EB on w3: {fmt(lc['w3'][c])}\n")
        for k, v in lc["pairwise_w3"].items():
            f.write(f"- {k}: {fmt(v)}\n")
        f.write("\n## Multi-seed (selective, EB anchor)\n\n")
        for seed in SEEDS:
            e = ms["per_seed"]["eb"][str(seed)]
            f.write(f"- seed {seed}: fwd_vs_eb {fmt(e['fwd_vs_eb'])}; "
                    f"w3 {fmt(e['w3_vs_eb'])}\n")
        f.write(f"- ENSEMBLE: fwd_vs_eb {fmt(ms['ensemble']['eb']['fwd_vs_eb'])}; "
                f"w3 {fmt(ms['ensemble']['eb']['w3_vs_eb'])} "
                f"(98.33% CI {ms['ensemble']['eb']['w3_vs_eb'].get('ci0.98333')})\n")
    log("campaign complete")


if __name__ == "__main__":
    main()
