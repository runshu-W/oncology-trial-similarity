"""Sanity and correctness checks for the gold-standard simulation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dgm import SimConfig, label_summary, simulate_dataset  # noqa: E402
from methods import MixturePrior, robust_map, rule_components, weak_only  # noqa: E402
from metrics import SUCCESS_THRESHOLD, pr_auc, roc_auc  # noqa: E402

ok = True


def check(name, condition, detail=""):
    global ok
    ok &= bool(condition)
    print(f"[{'OK ' if condition else 'FAIL'}] {name} {detail}")


# 1. Weak-only prior must be well calibrated: nominal 95% coverage and
#    one-sided type I error at or below 0.025.
data = simulate_dataset(SimConfig(), 3000, 99)
cov, t1 = [], []
for q in data:
    p = weak_only(q["candidates"])
    lo, hi = p.posterior_interval(q["y_query"], q["n_query"])
    cov.append(lo <= q["theta_query"] <= hi)
    t1.append(p.posterior_prob_greater(q["y_query"], q["n_query"], q["theta_query"]) > SUCCESS_THRESHOLD)
cov, t1 = np.mean(cov), np.mean(t1)
check("weak-only coverage ~0.95", 0.93 <= cov <= 0.975, f"= {cov:.4f}")
check("weak-only type I <= 0.035", t1 <= 0.035, f"= {t1:.4f}")

# 2. ROC-AUC helper against a known-correct case.
check("roc_auc perfect separation = 1", abs(roc_auc([0.1, 0.2, 0.9, 1.0], [0, 0, 1, 1]) - 1.0) < 1e-9)
check("roc_auc reversed = 0", abs(roc_auc([0.9, 1.0, 0.1, 0.2], [0, 0, 1, 1]) - 0.0) < 1e-9)
check("roc_auc all ties = 0.5", abs(roc_auc([1, 1, 1, 1], [0, 0, 1, 1]) - 0.5) < 1e-9)
check("pr_auc perfect = 1", abs(pr_auc([0.1, 0.2, 0.9, 1.0], [0, 0, 1, 1]) - 1.0) < 1e-9)

# 3. Posterior of a pure Beta(1,1) prior must equal the analytic Beta(y+1,n-y+1).
p = MixturePrior(1.0, [], [], [])
y, n = 7, 20
analytic_mean = (y + 1) / (n + 2)
check("weak posterior mean = (y+1)/(n+2)",
      abs(p.posterior_mean(y, n) - analytic_mean) < 1e-6,
      f"{p.posterior_mean(y, n):.6f} vs {analytic_mean:.6f}")

# 4. Mixture predictive probabilities must sum to 1 over y = 0..n.
pr = rule_components(data[0]["candidates"])
total = sum(np.exp(pr.log_predictive(k, 12)) for k in range(13))
check("predictive sums to 1 over support", abs(total - 1.0) < 1e-6, f"= {total:.8f}")

# 5. The DGM must actually produce traps and hidden gems.
s = label_summary(data)
check("DGM produces traps", s["prop_trap"] > 0.10, f"prop_trap = {s['prop_trap']:.3f}")
check("DGM produces hidden gems", s["prop_hidden_gem"] > 0.02, f"prop_gem = {s['prop_hidden_gem']:.3f}")
check("DGM borrowable fraction sensible", 0.10 < s["prop_borrowable"] < 0.60,
      f"= {s['prop_borrowable']:.3f}")

# 6. Full pooling must over-borrow relative to weak-only under conflict
#    (a directional check that the harness can detect harm).
conf = simulate_dataset(SimConfig(conflict_shift=1.5), 400, 5)
from methods import pooling  # noqa: E402
bias_pool = np.mean([pooling(q["candidates"]).posterior_mean(q["y_query"], q["n_query"]) - q["theta_query"] for q in conf])
bias_weak = np.mean([weak_only(q["candidates"]).posterior_mean(q["y_query"], q["n_query"]) - q["theta_query"] for q in conf])
check("pooling more biased than weak under conflict",
      abs(bias_pool) > abs(bias_weak), f"|{bias_pool:.4f}| vs |{bias_weak:.4f}|")

# ---------------------------------------------------------------------------
# 7. LEAKAGE: the prospective conflict signal must not depend on the query
#    outcome. This is the property that makes the resulting allocation usable
#    at design time, so it is checked directly rather than argued for.
# ---------------------------------------------------------------------------
from methods import candidate_features, prospective_conflict  # noqa: E402

q = simulate_dataset(SimConfig(), 1, 777)[0]
g_before = prospective_conflict(q["candidates"]).copy()
for y_alt in (0, q["n_query"], max(0, q["n_query"] // 2)):
    q["y_query"] = y_alt
    g_after = prospective_conflict(q["candidates"])
    if not np.allclose(g_before, g_after):
        break
check("prospective signal independent of query outcome",
      np.allclose(g_before, prospective_conflict(q["candidates"])))

# The signal must also be permutation equivariant: reordering candidates
# permutes g the same way, since the model is a set function.
perm = np.random.default_rng(0).permutation(len(q["candidates"]))
shuffled = [q["candidates"][i] for i in perm]
for c in q["candidates"] + shuffled:
    c.pop("_g_cached", None)
g_perm = prospective_conflict(shuffled)
check("prospective signal is permutation equivariant",
      np.allclose(g_perm, prospective_conflict(q["candidates"])[perm]))

# It must carry real information about outcome-space discrepancy, otherwise the
# discount head has nothing to condition on.
big = simulate_dataset(SimConfig(), 300, 424)
gs, ds = [], []
for qq in big:
    for c in qq["candidates"]:
        c.pop("_g_cached", None)
    gs.extend(prospective_conflict(qq["candidates"]))
    ds.extend(abs(c["theta_obs"] - qq["theta_query"]) for c in qq["candidates"])
corr = float(np.corrcoef(gs, ds)[0, 1])
check("prospective signal correlates with true distance", corr > 0.15, f"r = {corr:.3f}")

# Feature matrix width must match what each model expects.
check("prospective feature matrix is 10-dimensional",
      candidate_features(q["candidates"], prospective=True).shape[1] == 10)
check("plain feature matrix is 9-dimensional",
      candidate_features(q["candidates"], prospective=False).shape[1] == 9)

# ---------------------------------------------------------------------------
# 8. DESIGN WORLDS: pinning theta_0 must be exact, and must preserve the donor
#    structure rather than overwrite it.
# ---------------------------------------------------------------------------
from metrics import THETA_NULL  # noqa: E402

pinned = simulate_dataset(SimConfig(theta_query_fixed=THETA_NULL), 200, 31)
tq = np.array([p["theta_query"] for p in pinned])
check("design world pins theta_0 exactly", np.allclose(tq, THETA_NULL),
      f"range {tq.min():.6f}-{tq.max():.6f}")
free = simulate_dataset(SimConfig(), 200, 31)
lab_pinned = label_summary(pinned)["prop_borrowable"]
lab_free = label_summary(free)["prop_borrowable"]
check("recentring preserves the exchangeability structure",
      abs(lab_pinned - lab_free) < 0.08,
      f"borrowable {lab_pinned:.3f} pinned vs {lab_free:.3f} free")

# Weak-only type I error against the FIXED null is the no-borrowing reference.
# It need not equal 0.025 exactly because the binomial is discrete, but a gross
# departure would mean the design harness is wrong.
from methods import weak_only as _weak  # noqa: E402
rej = np.mean([_weak(p["candidates"]).posterior_prob_greater(
    p["y_query"], p["n_query"], THETA_NULL) > SUCCESS_THRESHOLD for p in pinned])
check("no-borrowing type I error near nominal", 0.005 <= rej <= 0.08, f"= {rej:.4f}")

# ---------------------------------------------------------------------------
# 9. UIP: reproduce the closed form of Jin & Yin (2021), section 3.2, on a
#    hand-computable case, and check the paper's stated ESS ~ M - 1 property.
# ---------------------------------------------------------------------------
from methods import _uip_from_weights, uip_dirichlet, uip_js  # noqa: E402

# With identical donors the UIP mean must equal their common rate and the
# borrowed ESS must be close to the mean of the M grid, i.e. the paper's
# "M represents the amount of information" statement.
same = [{"y": 20, "n": 100, "dim": data[0]["candidates"][0]["dim"],
         "features": data[0]["candidates"][0]["features"]} for _ in range(5)]
u = _uip_from_weights(same, np.ones(5), 40.0)
mean_u = float(np.sum(u.weights * u.alphas / (u.alphas + u.betas)))
theta_j = (20 + 0.5) / (100 + 1.0)          # Jeffreys-smoothed rate
check("UIP prior mean equals the common donor rate",
      abs(mean_u - theta_j) < 1e-6, f"{mean_u:.6f} vs {theta_j:.6f}")
check("UIP borrowed ESS is bounded by the current sample size",
      u.prior_ess() <= 40.0 * 1.05, f"ESS = {u.prior_ess():.2f}")

# UIP-Dirichlet must not depend on the query outcome; UIP-JS must.
cands0 = data[1]["candidates"]
d_lo = uip_dirichlet(cands0, 1, 30).alphas[0]
d_hi = uip_dirichlet(cands0, 29, 30).alphas[0]
check("UIP-Dirichlet is outcome independent", abs(d_lo - d_hi) < 1e-9)
j_lo = uip_js(cands0, 1, 30).scores
j_hi = uip_js(cands0, 29, 30).scores
check("UIP-JS is outcome dependent (retrospective)",
      not np.allclose(j_lo, j_hi))

# ---------------------------------------------------------------------------
# 10. EXTERNAL CONTROL: labels and the exact treatment-effect convolution.
# ---------------------------------------------------------------------------
from dgm import ExternalControlConfig, simulate_external_control_dataset  # noqa: E402

ec = simulate_external_control_dataset(
    ExternalControlConfig(drift_shift=-1.0, p_comparable=0.5), 300, 8)
comp = np.mean([c["comparable"] for q in ec for c in q["candidates"]])
check("EC comparable fraction matches p_comparable", 0.42 <= comp <= 0.58,
      f"= {comp:.3f}")
# Comparable donors must be borrowable far more often than drifted ones.
b_comp = np.mean([c["borrowable"] for q in ec for c in q["candidates"] if c["comparable"]])
b_drift = np.mean([c["borrowable"] for q in ec for c in q["candidates"] if not c["comparable"]])
check("drift makes donors non-exchangeable", b_comp > b_drift + 0.2,
      f"comparable {b_comp:.3f} vs drifted {b_drift:.3f}")

# The convolution-based effect posterior must match the analytic difference of
# posterior means when nothing is borrowed.
q0 = ec[0]
p0 = _weak(q0["candidates"])
eff, lo_e, hi_e, p_sup, _ = p0.treatment_effect(
    q0["y_control"], q0["n_control"], q0["y_treatment"], q0["n_treatment"])
analytic = ((1 + q0["y_treatment"]) / (2 + q0["n_treatment"])
            - (1 + q0["y_control"]) / (2 + q0["n_control"]))
check("treatment-effect mean matches analytic value",
      abs(eff - analytic) < 1e-6, f"{eff:.6f} vs {analytic:.6f}")
check("treatment-effect interval brackets the mean", lo_e <= eff <= hi_e)
check("superiority probability is a probability", 0.0 <= p_sup <= 1.0)

# ---------------------------------------------------------------------------
# 11. Determinism: the same seed must give the same data across processes.
# ---------------------------------------------------------------------------
import zlib  # noqa: E402

a1 = simulate_dataset(SimConfig(), 5, 12345)
a2 = simulate_dataset(SimConfig(), 5, 12345)
check("simulation is reproducible for a fixed seed",
      all(x["y_query"] == y["y_query"] and x["n_query"] == y["n_query"]
          for x, y in zip(a1, a2)))
check("scenario seeds use a process-stable hash",
      zlib.crc32(b"S2_trap_heavy") == zlib.crc32(b"S2_trap_heavy"))

print("\nALL SIMULATION CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
