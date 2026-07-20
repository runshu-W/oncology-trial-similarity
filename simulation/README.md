# 2. Simulation

Controlled evaluation of the borrowing layer, where the correct answer is known
by construction. This is what the retrospective analyses in `../pipeline/`
cannot provide: on real data the truth is unknown, so one can measure predictive
fit but not whether the method borrowed from the *right* trials.

## Contents

| Path | Purpose |
|---|---|
| `gold_standard/` | Main study. Parameter-level exchangeability known by construction; design-based type I error and power; borrowing-decision diagnostic accuracy; external control arm scenario. See its `README.md`. |
| `run_borrowing_operating_characteristics_simulation.py` | Earlier operating-characteristics study retained for continuity with previously reported results. |

## Our method versus other methods

Every prior is scored on identical footing — the same candidate sets, the same
estimand, the same decision rule — and lives in
`gold_standard/methods.py`.

**Ours**

- `two_head_pro` — learned set function producing both a mixture allocation and
  a per-source ESS discount, with the prospective set-level conflict signal
- `two_head` — same, without the prospective signal (ablation for contribution 2)
- `two_head_pro_fixdisc` — same, with a constant discount (ablation for contribution 3)
- `two_head_pro_sam` — with the outcome-dependent SAM adapter

**Others**

- `weak_only` — no borrowing, the reference
- `rule` / `rule_sam` — the rule-based mixture from the manuscript
- `robust_map_w0.5`, `robust_map_w0.9` — robustified pooled MAP
- `power_prior` — fixed power parameter
- `uip_dirichlet` — unit information prior, sample-size weights (Jin & Yin 2021); outcome-independent
- `uip_js` — unit information prior, JS-divergence weights; **uses the current outcome**, so it is retrospective and is labelled as such wherever it appears
- `pooling` — full pooling

The prospective/retrospective distinction is load-bearing. Methods that consult
the query outcome have strictly more information than methods usable at design
time, and comparing them as equals would overstate what a design-time method can
achieve.

## What the results say

Reported in full in `../results/` and in the manuscript. In brief: the learned
prior holds type I error at nominal without conflict and buys about five
percentage points of power over not borrowing, but borrows more aggressively
than the rule and robust-MAP baselines and pays for it in type I error as
prior-data conflict grows. The prospective conflict signal does not improve
donor discrimination, but is what makes the learned per-source ESS discount
respond to comparability at all — without it the discount head is inert. Both
findings, including the negative one, are reported as they came out.
