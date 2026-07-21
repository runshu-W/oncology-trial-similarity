# Revision changelog — 2026-07-20

Implements 手稿修改方案 v3 under the two decisions taken: **Strategy A**
positioning (integration + evaluation protocol, method increments reported as
incremental) and **Pharm Stat / Stat Med** as target journals.

---

## 1. Corrected the type I error definition

**The problem.** `run_simulation.py` set the null threshold to `theta0` — the
replicate's own true value:

```python
post_mean, lo, hi, p_null, p_power = prior.posterior_summary(
    y0, n0, theta0, max(theta0 - POWER_MARGIN, 0.001))
```

`P(P(theta > theta_true | data) > 0.975)` is a one-sided posterior calibration
rate, not a frequentist type I error. No fixed null means no type I error.

**The fix.** A standard single-arm oncology design testing `H0: theta <= 0.20`
at the one-sided 0.025 level, evaluated in two worlds:

- **null world**, `theta_0 = 0.20` exactly → rejection rate *is* the type I error
- **alternative world**, `theta_0 = 0.35` → rejection rate *is* the power

`theta_0` is pinned by shifting the query *and every donor* by a common logit
offset, so all exchangeability relationships are preserved and only the location
moves. Verified: borrowable fraction 0.263 pinned vs 0.263 free.

The old quantity is retained as `calibration_rate` and labelled honestly.

Reported alongside the nominal 0.025 is the **no-borrowing reference** (0.037),
because a Bayesian test of a point null with a flat prior cannot attain the
nominal level exactly at these sample sizes — the binomial is discrete.

## 2. Fixed a reproducibility defect

Scenario seeds used `abs(hash(sname))`. Python randomises string hashing per
process unless `PYTHONHASHSEED` is set, so runs were not reproducible. Replaced
with `zlib.crc32`, which is stable across processes and platforms.

## 3. Contribution 2 — prospective set-level conflict signal

`g_i = |r_i - m_{-i}| / s_{-i}`, where `m_{-i}` is the sample-size-weighted
median of the *other* donors' observed rates and `s_{-i}` the corresponding
scaled MAD. Leave-one-out, so a dominant donor cannot define the consensus it is
measured against. Uses only donors' published outcomes — never the query
outcome — and is therefore available at design time. Enforced by a leakage test.

**Result — honest and partly negative.** Discrimination gains are negligible
(ROC-AUC changes −0.000 to +0.016 across five scenarios) and type I error does
not improve. The reason is structural: a consensus-based signal detects
*heterogeneity within* the donor set, not a *uniform displacement of the whole
set*. Under uniform conflict every design-time method falls **below chance**
(ROC-AUC 0.445); only outcome-dependent UIP-JS stays informative (0.839).

**But it does something else, which is the real finding.** Without `g_i` the
learned discount head is **inert**: fitted discount median 1.000, SD 0.002, and
per-source ESS flat across true-distance bins (43.7 / 43.9 / 44.1 / 43.8). With
`g_i` the discount spans its range (median 0.953, SD 0.427) and ESS declines
monotonically (34.9 / 31.3 / 24.2 / 20.3). Cause: the nine reranking dimensions
are all design/surface similarity and carry **no** outcome-space information, so
the discount head has nothing to condition on. `g_i` correlates +0.35 with the
true distance `|theta_k − theta_0|`.

Contributions 2 and 3 are therefore **coupled** — neither stands alone — and the
manuscript now says so.

## 4. Closed-form comparators

**UIP (Jin & Yin 2021, Stat Med 40(25):5657–5672), §3.2 binary case**, implemented
to the paper's formulas. `M` is marginalised on a grid rather than by MCMC,
turning the UIP into a finite beta mixture scored on the same footing as every
other prior. Both variants:

- **UIP-Dirichlet** — weights from `min(1, n_k/n)`; outcome-independent
- **UIP-JS** — weights ∝ `1/d_k` with `d_k` the JS divergence between
  Jeffreys-prior posteriors; **outcome-dependent**, therefore retrospective

That distinction is load-bearing and is flagged wherever UIP-JS is compared
against design-time methods.

**One deviation from the paper, documented.** Evaluating the unit information
`1/(theta(1-theta))` at the raw MLE explodes when `y_k = 0`, which is common in
small oncology arms: one zero-response donor inflated the borrowed ESS by an
order of magnitude (139 vs an expected ~16). We evaluate at the Jeffreys
posterior mean `(y+0.5)/(n+1)` instead — the same Jeffreys prior the paper
already uses for UIP-JS, so this is internally consistent rather than an ad hoc
patch. It restores the paper's own stated property that ESS ≈ M − 1 (measured
15.5 against a theoretical 16.4).

Also added: fixed-discount ablation isolating contribution 3.

## 5. Auxiliary gold standard

Oracle allocation (mass on exchangeable donors ∝ information, zero elsewhere),
compared to each method's weights by Spearman correlation and KL. Plus the
learned-ESS response curve binned by true distance.

## 6. External control arm — full pipeline

Internal treatment arm (n 40–70) + small internal control (n 15–30) + retrievable
external control arms, a fraction drifted. Estimand is the treatment effect;
success when `P(theta_trt > theta_ctl) > 0.975`. The difference law is computed
**exactly** as the discrete cross-correlation of the two posteriors
(`np.correlate`) — matches the analytic value to 4e-13 — not by Monte Carlo.

Drift swept in both directions, plus a `p_comparable` sensitivity.

Headline (drift −1.2, no true effect): internal-only 0.014 → pooling 0.211,
power prior 0.150, two-head+prospective 0.032, fixed-discount 0.025. Robust-MAP
looks safest (0.015) but its FPR for borrowing non-comparable controls is
**1.000** — it doesn't discriminate at all; its safety is purely from borrowing
almost nothing (ESS 7.7).

## 7. Data defect found while auditing for the real case study

`docs/oncology_trial_similarity_pipeline.py:296` computes
`proportion = count / denominator` unconditionally. `count` is the value *in the
reported unit*. When ClinicalTrials.gov reports ORR as "Percentage of
Participants", `26.0` means 26%, not 26 responders — the extraction produced
0.839.

| Scope | |
|---|---|
| Arm-level ORR rows in corpus | 3319 |
| Reported in percentage units | 2040 |
| Assigned a rate > 1 (provably wrong) | 804 (24.2%) |
| Pseudo-queries located / corrected | 93 of 116 / 9 |
| Corrected by > 0.10 absolute | 8 (max 0.579) |

Affected: aggregate real-data NLL, calibration, coverage.
**Not** affected: the two worked case studies (participant units, 6/26 and 5/32,
correct as printed), and both simulations, which never touch this path.

Added `scripts/fix_endpoint_units.py` (unit-aware, refuses to guess on
unrecognised units — 29 rows) and `scripts/audit_orr_units.py` (per-query
correction table). Full re-run needs the raw records, which were not available in
this environment; documented in the manuscript as a main-text limitation rather
than a supplement note, since a defect producing plausible-looking numbers is not
one a reader can catch from the results.

The planned real external-control case study was **deferred** rather than built
on an extraction path just found unreliable.

## 8. Performance

- Cached the prospective signal (was recomputed every forward pass, every epoch —
  this was making training appear to hang): 60 epochs now ~14 s
- Posterior grid 1500 → 800 (resolution 0.00125, far finer than anything reported)
- UIP M-grid 24 → 12
- Vectorised the treatment-effect convolution
- Per-cell checkpointing so interrupted runs resume

Net: ~17 ms/replicate; 2000 replicates per cell throughout.

## 9. Manuscript

- Honest positioning section naming the precedents (UIP, PSPP, SPx,
  covariate-adjusted borrowing, SAM, DeepSets) and stating that no individual
  component is new; the defended contribution is the integration plus the
  gold-standard protocol
- New Methods: prospective signal, gold-standard simulation (ADEMP), external
  control scenario
- New Results: design-based OC, the prospective-signal ablation including its
  failure, external control
- Statistical nomenclature (sensitivity/specificity/PPV) rather than ML naming;
  trial-level and donor-level decisions kept separate
- New Table `tab:design-oc`, Table `tab:ec`, Figures F8–F11
- Main-text data-defect section
- References 19–25 added; details for refs 20, 22, 23 **corrected against source**
  after initial drafting introduced errors
- Compiles clean: 33 pages, 0 undefined references

## 10. Tests

30 checks pass, including: prospective signal independent of query outcome
(leakage), permutation equivariance, exact `theta_0` pinning, structure
preservation under recentring, UIP mean/ESS against the closed form, UIP-Dirichlet
outcome-independence vs UIP-JS outcome-dependence, EC drift labels, exact
treatment-effect convolution, and seed reproducibility. Gradient checks still pass.

---

## Open items

1. **Real-data re-run** with corrected units — needs `/Users/wang/PHD/clinic.gov/`
2. **Real external-control case study** — blocked on (1)
3. **S4 below-chance discrimination** — a real limitation, currently discussed, not solved
4. **Learned discount does not improve OC** over a fixed discount despite being
   adaptive — reported as such; worth understanding before claiming more for it
