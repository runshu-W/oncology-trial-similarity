# Selective borrowing experiment — verdict (2026-08-31)

Architecture: canonical two-head DeepSets + (i) set-level weak-component logit head
(weak joins the candidate softmax; total borrowing learnable per query) and (ii)
EB anchor (past-only/train-only beta-binomial MLE) replacing Beta(1,1). Canonical
hyperparameters and seeds (epochs 100, lr 0.01, hidden 16, seed 20260603); fold
train sets identical to the canonical splits (632/813/1,017; primary 1,126).
Evaluation: paired vs the EB reference, disjoint forward windows (leakage-free,
each query scored once by the freshest legitimately trained model) and the 20%
test split. 4,000-resample bootstrap CIs.

## Pre-registered criteria

- Primary (aggregate forward delta vs EB < 0, CI excluding 0): **not met, narrowly.**
  d = -0.0266, CI [-0.0575, +0.0045], win 55%. Test split: d = -0.0295,
  CI [-0.0710, +0.0122].
- Fallback (no aggregate loss + significant high-info win + no significant loss
  elsewhere): high-info subset d = -0.0585, CI [-0.1254, +0.0077] — right
  direction, borderline; low-info subset safely neutral (-0.0074,
  CI [-0.0369, +0.0229]). **Partially met.**

## What the experiment did establish

1. **The architectural fix is a real improvement over the published model.**
   Paired, leakage-free, CIs exclude zero:
   - selective vs old two_head (Beta(1,1) anchor, fixed lambda_0 = 0.2):
     d = -0.0665, CI [-0.0979, -0.0352];
   - selective vs old two_head with post-hoc EB anchor swap:
     d = -0.0268, CI [-0.0514, -0.0015].
   The aggregate relationship to EB moves from a significant-leaning loss
   (+0.040) to a lead (-0.027). Reviewer Major-4's proposed fix is validated.
2. **The learned lambda_0 performs design-time triage.** Mean borrowed mass
   0.310 when candidate information is poor (info < 8) vs 0.585 when rich
   (info >= 8); Spearman(borrowed mass, info) = +0.59; borrowed-mass quartiles
   0.15 / 0.36 / 0.65 — against a forced ~0.79 everywhere in the old
   architecture.
3. **The best-trained window wins significantly.** Window 3 (>2022-12-31,
   n = 390, trained on 1,017 past examples): d = -0.0461, CI [-0.0832, -0.0098].
   Windows 1-2 (632/813 training examples) are ~zero — a learning-curve
   pattern consistent with triage quality improving with training data.

## Honest claim now supportable

"A selective borrowing architecture — learnable total borrowing plus an
empirical-Bayes anchor — significantly improves on the fixed-budget two-head
model, matches or slightly exceeds an intercept-only EB prior on aggregate
forward validation (CI [-0.058, +0.005]), wins significantly in the
best-trained forward window, and learns an interpretable design-time triage
policy (borrowing mass tracks candidate-set information)." Combined with the
known-truth simulation showing when borrowing helps and harms, this is a
viable Path-A paper; the claim "beats the marginal prior on average" is not
yet demonstrable and must not be made.

## Remaining lever (pre-registered next step, if pursued)

Multi-seed prior ensembling (average the mixture prior over ~5 training seeds),
which reviewers independently requested for stability reporting. Mechanical,
no tuning contact with evaluation data. May or may not push the aggregate CI
past zero; report either way.
