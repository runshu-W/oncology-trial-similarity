# Path-A pathfinding: when does candidate-level borrowing beat an intercept-only EB prior?

Date: 2026-08-31. All comparisons paired per query, bootstrap 95% CIs (2,000 resamples).
EB reference: Beta fit by beta-binomial MLE on training data only (train split for the
20% test analysis; past-only data for each forward window). Forward analysis uses
disjoint future windows, each scored by the fold model trained strictly on the past
(w1 = 2021 window / model_2020, w2 = 2022 / model_2021, w3 = >2022 / model_2022),
so every query is evaluated exactly once, leakage-free. "EB-anchor" = the same mixture
with the weak component swapped from Beta(1,1) to the past-fit EB beta (post-hoc swap,
a lower bound on retraining with that anchor).

## 1. Aggregate: candidate borrowing adds nothing on average

Delta NLL vs EB (negative = borrowing better):

| comparison | 20% test (n=281) | disjoint forward pooled (n=775) |
|---|---|---|
| two_head vs EB | +0.029 [-0.041, +0.099] | +0.040 [-0.008, +0.088] |
| two_head (EB anchor) vs EB | -0.004 [-0.063, +0.059] | +0.000 [-0.040, +0.041] |
| rule vs EB | +0.153 [+0.073, +0.237] (worse) | +0.125 [+0.073, +0.178] (worse) |
| rule (EB anchor) vs EB | +0.102 (worse) | +0.052 [+0.010, +0.094] (worse) |

The anchor swap closes the two_head gap to an exact tie; rule-based borrowing is
significantly worse than the marginal prior. Reviewer Major-6 confirmed at the
aggregate level.

## 2. But a design-time-identifiable winning regime exists, and it replicates

Borrowable information = sum over candidates of gate x discount x n_i.

| view | subset | n | delta vs EB | 95% CI | win rate |
|---|---|---|---|---|---|
| forward pooled | info >= 8 | 291 | -0.080 | [-0.151, -0.007] | 60% |
| forward pooled | query n0 > 60 | 140 | -0.202 | [-0.300, -0.106] | 69% |
| forward pooled | Phase 3 | 38 | -0.365 | [-0.580, -0.146] | 71% |
| forward pooled | info >= 8 AND n0 > 60 | 70 | -0.267 | [-0.418, -0.106] | 70% |
| 20% test split | info >= 8 | 97 | -0.113 | [-0.217, -0.007] | 58% |
| rule-derivation windows (w1+w2) | info >= 16 | 80 | -0.217 | [-0.379, -0.058] | 75% |
| untouched window w3 | info >= 8 | 138 | -0.066 | [-0.164, +0.031] | 53% (directionally consistent) |
| untouched w3 | selective policy (borrow iff info >= 8, else EB) | 390 | -0.023 | [-0.058, +0.011] | — |

Monotone dose-response on the derivation windows (info >= 4/6/8/12/16/20 gives
-0.056/-0.082/-0.092/-0.106/-0.217/-0.249). Spearman correlations between delta and
design-time covariates are directionally coherent (info -0.13, max similarity -0.14,
n0 -0.14).

Symmetric losing regimes (design-time): max similarity < 0.55 (+0.064, CI excludes 0),
near-degenerate donor consensus SD < 0.05 (+0.157), Phase 1 (+0.231).

Caveats: thresholds were explored on multiple stratifiers (multiplicity); the
strongest cells have modest n; n0 is treated as approximately design-time (analyzed
denominator ~ planned size). The w3 validation is directionally consistent but not
individually significant.

## 3. Mechanism (outcome-dependent diagnostic, not usable as a selection rule)

| |donor-consensus mean - observed rate| | n | delta vs EB | win rate |
|---|---|---|---|
| < 0.05 | 133 | -0.316 | 84% |
| 0.05-0.15 | 210 | -0.248 | 82% |
| > 0.15 (56% of queries) | 432 | +0.218 | 34% |

The machinery wins exactly when the retrieved donors are actually informative about
the query truth; it loses when they are not — and the current architecture cannot
decline to borrow (lambda_0 fixed at 0.2 forces 79% historical mass on average).

## 4. Verdict and the Path-A pivot

Path A is viable, with the claim recentred from "learned borrowing beats baselines"
to **selective borrowing**:

1. Replace the Beta(1,1) weak component with a past-fit EB anchor.
2. Make total borrowing learnable (weak component inside the softmax, or a set-level
   total-borrowing head) — exactly reviewer Major-4's architectural fix. The learned
   policy should reproduce the info-based triage that the stratified analysis found
   by hand, converting regime-level wins into aggregate wins and shrinking the
   losing-regime penalty.
3. Report EB and stratified-EB as the reference baselines everywhere (Major-6), with
   design-time subgroup analyses pre-specified from this pathfinding.
4. This simultaneously supplies the legitimate, design-time replacement for what SAM
   was doing with the outcome (Major-1).

Pivotal next experiment: retrain the two-head model with (EB anchor + learnable
lambda_0), primary + three past-only fold refits, then rerun the EB-referenced
forward comparison. Success criterion: aggregate forward delta vs EB < 0 with CI
excluding 0, or at minimum no aggregate loss plus significant wins in the
pre-specified high-info regime and no significant loss elsewhere.
