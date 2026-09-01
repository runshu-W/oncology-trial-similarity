#!/usr/bin/env python3
"""Round-9 programmatic verification: every number cited in the round-9
edits must match the artifact JSONs; mandated wordings present, banned
wordings absent."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = (ROOT / "manuscript" / "manuscript_full_draft_pharm_stats.tex").read_text()
S = (ROOT / "manuscript" / "supplement.tex").read_text()
T = json.load(open(ROOT / "artifacts_round4" / "stress" / "round9_tail_robustness.json"))
C = json.load(open(ROOT / "artifacts_round4" / "calibration" / "round9_crossfit_diag.json"))
R8 = json.load(open(ROOT / "artifacts_round4" / "stress" / "round8_direct_paired.json"))

ok = fail = 0


def chk(name, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print("FAIL:", name)


def near(a, b, tol=5e-4):
    return abs(a - b) <= tol


def r(x, d):
    return round(x, d)


# ---- primary tail statistics --------------------------------------------
b = T["primaryB_pooled"]["binding"]
chk("bind n 50", T["primaryB_pooled"]["n_binding"] == 50)
chk("bind mean -0.340", near(r(b["mean"], 3), -0.340))
chk("bind median -0.030", near(r(b["median"], 3), -0.030))
chk("bind IQR", near(r(b["q25"], 2), -0.39) and near(r(b["q75"], 2), 0.07))
chk("bind 5-95", near(r(b["q05"], 2), -2.06) and near(r(b["q95"], 2), 0.36))
chk("bind range", near(r(b["min"], 2), -4.66) and near(r(b["max"], 2), 0.44))
chk("bind 30 neg", b["n_negative"] == 30)
chk("trim10 -0.154", near(r(b["trimmed10_mean"], 3), -0.154))
chk("wins5 -0.273", near(r(b["winsor5_mean"], 3), -0.273))
chk("poolwins1 -0.017", near(r(T["primaryB_pooled"]["pooled_winsor1_mean"], 3), -0.017))
lq = T["primaryB_pooled"]["loo_query"]
lb = T["primaryB_pooled"]["loo_block"]
chk("LOOq [-0.030,-0.021]", near(r(lq["min"], 3), -0.030) and near(r(lq["max"], 3), -0.021))
chk("LOOb [-0.030,-0.021]", near(r(lb["min"], 3), -0.030) and near(r(lb["max"], 3), -0.021))
chk("LOO all negative", lq["all_negative"] and lb["all_negative"])
chk("infl row -4.66 -> -0.021", near(r(lq["most_influential_d"], 2), -4.66)
    and near(r(lq["mean_without_most_influential"], 3), -0.021))
W = T["primaryB_pooled"]["windows"]
chk("w1 -0.012/2", near(r(W["w1"]["mean"], 3), -0.012) and W["w1"]["n_binding"] == 2)
chk("w1 blockCI", near(r(W["w1"]["ci95_block"][0], 3), -0.037)
    and near(r(W["w1"]["ci95_block"][1], 3), 0.000))
chk("w2 +0.004/9", near(r(W["w2"]["mean"], 3), 0.004) and W["w2"]["n_binding"] == 9)
chk("w2 blockCI", near(r(W["w2"]["ci95_block"][0], 3), -0.003)
    and near(r(W["w2"]["ci95_block"][1], 3), 0.014))
chk("w3 -0.053/39", near(r(W["w3"]["mean"], 3), -0.053) and W["w3"]["n_binding"] == 39)
chk("w3 blockCI", near(r(W["w3"]["ci95_block"][0], 3), -0.095)
    and near(r(W["w3"]["ci95_block"][1], 3), -0.020))
le = T["primaryB_engaged"]
chk("engaged LOOq", near(r(le["loo_query"]["min"], 3), -0.148)
    and near(r(le["loo_query"]["max"], 3), -0.105))
chk("engaged LOOb", near(r(le["loo_block"]["min"], 3), -0.152)
    and near(r(le["loo_block"]["max"], 3), -0.105))

# ---- forward / title-screened tails -------------------------------------
f = T["fwd"]["binding"]
chk("fwd med +0.021", near(r(f["median"], 3), 0.021))
chk("fwd IQR", near(r(f["q25"], 2), -0.06) and near(r(f["q75"], 2), 0.11))
chk("fwd q05 -0.69", near(r(f["q05"], 2), -0.69))
chk("fwd 5-95 hi", near(r(f["q95"], 2), 0.30))
chk("fwd range", near(r(f["min"], 2), -2.16) and near(r(f["max"], 2), 0.52))
chk("fwd trim +0.017", near(r(f["trimmed10_mean"], 3), 0.017))
chk("fwd trim sign-flipped", f["trimmed10_mean"] > 0 > T["fwd"]["mean"])
chk("fwd LOOq", near(r(T["fwd"]["loo_query"]["min"], 3), -0.015)
    and near(r(T["fwd"]["loo_query"]["max"], 3), -0.012))
chk("fwd LOOb", near(r(T["fwd"]["loo_block"]["min"], 3), -0.015)
    and near(r(T["fwd"]["loo_block"]["max"], 3), -0.012))
for k, med, tr, lo_q, hi_q, lo_b, hi_b in (
        ("strictA", 0.020, 0.021, -0.016, -0.009, -0.017, -0.009),
        ("strictC", 0.022, 0.016, -0.012, -0.008, -0.012, -0.007)):
    chk(f"{k} med", near(r(T[k]["binding"]["median"], 3), med))
    chk(f"{k} trim", near(r(T[k]["binding"]["trimmed10_mean"], 3), tr))
    chk(f"{k} LOOq", near(r(T[k]["loo_query"]["min"], 3), lo_q)
        and near(r(T[k]["loo_query"]["max"], 3), hi_q))
    chk(f"{k} LOOb", near(r(T[k]["loo_block"]["min"], 3), lo_b)
        and near(r(T[k]["loo_block"]["max"], 3), hi_b))
chk("strictA IQR", near(r(T["strictA"]["binding"]["q25"], 2), -0.08)
    and near(r(T["strictA"]["binding"]["q75"], 2), 0.17))
chk("strictA 5-95", near(r(T["strictA"]["binding"]["q05"], 2), -0.70)
    and near(r(T["strictA"]["binding"]["q95"], 2), 0.40))
chk("strictA range", near(r(T["strictA"]["binding"]["min"], 2), -3.48)
    and near(r(T["strictA"]["binding"]["max"], 2), 0.58))
chk("strictC IQR", near(r(T["strictC"]["binding"]["q25"], 2), -0.05)
    and near(r(T["strictC"]["binding"]["q75"], 2), 0.11))
chk("strictC 5-95", near(r(T["strictC"]["binding"]["q05"], 2), -0.69)
    and near(r(T["strictC"]["binding"]["q95"], 2), 0.29))
chk("strictC range", near(r(T["strictC"]["binding"]["min"], 2), -1.65)
    and near(r(T["strictC"]["binding"]["max"], 2), 0.49))
# means still equal round-8
for k in ("primaryB_pooled", "primaryB_engaged", "fwd", "strictA", "strictC"):
    chk(f"{k} mean == round8", abs(T[k]["mean"] - R8[k]["mean"]) < 1e-6)

# ---- forward windows vs EB (figure F2b + caption) -----------------------
fw = T["fwd_windows_vs_eb"]
pooled = sum(fw[w]["capped_nll"] * fw[w]["n"] for w in fw) / sum(fw[w]["n"] for w in fw)
chk("F2 pooled capped NLL 2.803", near(r(pooled, 3), 2.803))
chk("F2 w3 capped -0.045", near(r(fw["w3"]["capped_delta"], 3), -0.045))
chk("F2 w3 capped CI", near(r(fw["w3"]["capped_ci"][0], 3), -0.074)
    and near(r(fw["w3"]["capped_ci"][1], 3), -0.016))

# ---- cross-fit diagnostics ----------------------------------------------
chk("n_tail 8", C["n_tail"] == 8)
chk("tail range", near(min(C["tail_estimates"]), 0.0023, 5e-5)
    and near(max(C["tail_estimates"]), 0.0057, 1e-4))
chk("corr 0.91", near(r(C["corr_estimate_evalsizegap"], 2), 0.91))
chk("tau_w 4 unique", C["tau_w"]["n_unique"] == 4)
chk("weak eval tail 0.0249", near(r(C["eval_size_weak"]["tail_mean"], 4), 0.0249))
chk("weak eval rest 0.0257", near(r(C["eval_size_weak"]["rest_mean"], 4), 0.0257))
chk("cap50 eval stable", near(r(C["eval_size_cap50"]["tail_mean"], 5), 0.02515)
    and near(r(C["eval_size_cap50"]["rest_mean"], 5), 0.02514))
chk("residual 0.0013", near(r(C["est_size_residual_max"], 4), 0.0013))
a1, a5 = C["alt_mixings"]["N(0,1)"], C["alt_mixings"]["N(0,0.5)"]
chk("N01 target 0.02535", a1["target"] == 0.02535)
chk("N01 mean +0.0097", near(r(a1["mean"], 4), 0.0097))
chk("N01 median +0.0094", near(r(a1["median"], 4), 0.0094))
chk("N01 range", near(r(a1["q"]["0.025"], 4), 0.0090)
    and near(r(a1["q"]["0.975"], 4), 0.0149))
chk("N01 all positive", a1["frac_positive"] == 1.0)
chk("N05 target 0.0262", a5["target"] == 0.0262)
chk("N05 mean +0.0382", near(r(a5["mean"], 4), 0.0382))
chk("N05 median +0.0373", near(r(a5["median"], 4), 0.0373))
chk("N05 range", near(r(a5["q"]["0.025"], 4), 0.0365)
    and near(r(a5["q"]["0.975"], 4), 0.0432))
chk("N05 all positive", a5["frac_positive"] == 1.0)

# ---- textual presence ----------------------------------------------------
for frag in (
        "reduced the mean NLL} in this post hoc comparison",
        "conditionally on\nbinding} is a mean difference of $-0.340$",
        "median\n$-0.030$, IQR $[-0.39,+0.07]$",
        "$[-2.06,+0.36]$", "$[-4.66,+0.44]$", "$30$ of $50$ rows negative",
        "$10\\%$-trimmed mean is $-0.154$", "$5\\%$-winsorized mean $-0.273$",
        "percentiles leaves\n$-0.017$", "$[-0.030,-0.021]$",
        "single largest improvement, $-4.66$, leaves $-0.021$",
        "binding-row median $+0.021$", "trimmed mean $+0.017$, sign-flipped",
        "improve by more than $0.69$", "$[-0.015,-0.012]$",
        "central $95\\%$ split-quantile range",
        "correlates $0.91$", "four distinct values over the $400$",
        "average $0.0257$ against the\n$0.0251$ target",
        "($0.0249$)", "($0.02514$--$0.02515$)",
        "$N(0,1)$: mean $+0.0097$", "$[+0.0090,+0.0149]$",
        "$N(0,0.5)$: mean $+0.0382$", "$[+0.0365,+0.0432]$",
        "in two populations that we keep",
        "no endpoint screening",
        "compatible with both modest improvement and modest",
        "no directional conclusion is supported there",
        "tail-driven,\ninsurance-like effect",
        "reduce in mean", "reduced the mean harm",
        "Figure~S1"):
    chk(f"M has {frag[:38]!r}", frag in M)
for frag in ("at least as favourable", "did not detectably tax",
             "cost nothing detectable", "split interval $[",
             "$95\\%$ split interval"):
    chk(f"M lacks {frag!r}", frag not in M)
for frag in ("tab:s-tail", "fig:s-cf", "FS1_crossfit_dist",
             "round9\\_tail\\_robustness.json", "round9\\_crossfit\\_diag.json",
             "round9\\_rows.json", "Round-9 additions (this revision)",
             "$+0.0382$ with all $200$ splits",
             "$[-0.148,-0.105]$", "$[-0.152,-0.105]$",
             "$[-2.16,+0.52]$", "$[-3.48,+0.58]$", "$[-1.65,+0.49]$",
             "thefigure}{S\\arabic{figure}}",
             "target $0.02535$", "target $0.02620$"):
    chk(f"S has {frag[:38]!r}", frag in S)
for frag in ("did not detectably tax", "improves, not taxes"):
    chk(f"S lacks {frag!r}", frag not in S)
# F2 caption numbers
for frag in ("capped\n  selective, $2.803$", "final design $-0.045$",
             "CI $[-0.074,-0.016]$"):
    chk(f"M caption has {frag[:30]!r}", frag in M)

print(f"\n{ok} OK, {fail} FAIL")
sys.exit(1 if fail else 0)
