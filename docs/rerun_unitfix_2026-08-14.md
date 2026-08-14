# Corrected-units re-run of the retrospective borrowing evaluation — 2026-08-14

**Status:** evaluation chain fully re-run on unit-corrected data. The
manuscript's real-data numbers (head-to-head Table, forward validation,
calibration diagnostics, robust-MAP comparison) can now be updated and the
"provisional pending re-run" caveat retired. The manuscript text itself has
**not** been updated yet — that is the next step.

## What was re-run, and from what

The endpoint-unit defect (`docs/KNOWN_ISSUE_endpoint_units.md`) was fixed at
all four call sites in commit `0243795`. Because `trial_summaries.jsonl`
retains the reported unit for all 26,603 arm-level rows, and
`pipeline_results.jsonl` (June run, 1,470 leakage-controlled ORR
pseudo-queries) retains `borrowable_quantities` with per-outcome units, the
re-run regenerates everything downstream of retrieval **without touching the
raw ClinicalTrials.gov export and without re-running Stage 1/Stage 2** (the
unit fix does not enter retrieval or reranking scores).

Chain (all driven from the regenerated examples file):

1. `scripts/rebuild_lambda_examples_chunked.py` — chunked, resumable
   regeneration of training examples from the existing
   `pipeline_results.jsonl` (2.5 GB, worktree artifacts) using the canonical
   `build_examples_from_results` with the fixed conversion. 1,470 results →
   **1,407 examples**, 63 excluded (`failures_unitfix.jsonl`).
2. `pipeline/run_training_from_examples.py` (new driver, replays the
   orchestrator's post-example flow) — primary two-head DeepSets fit
   (80/20 deterministic split, epochs 100, lr 0.01, hidden 16, seed 20260603)
   plus three past-only fold refits for the forward validation.
3. `pipeline/two_head_inference.py` — attach model predictions
   (internal consistency check: mean |ΔNLL| vs training rows 6e-06).
4. `pipeline/run_borrowing_baseline_comparison.py`,
   `pipeline/run_robust_map_baselines.py`,
   `pipeline/run_calibration_diagnostics.py`,
   `pipeline/run_forward_validation.py` (refit mode, leakage-free).

Environment: Linux x86_64, Python 3.11.15, **torch 2.11.0** and
**numpy 2.4.4** (both matching `requirements.lock.txt` versions; platform
differs from the macOS arm64 environment of the July runs). All seeds fixed
(20260603 training, 20260707 evaluation bootstrap). Cross-platform numerical
differences should be negligible but a confirmation pass on the Mac `.venv`
is cheap now — see "Reproducing" below.

## How corrupted was the old data

Beyond the audited query-level errors (9 of 93 located queries changed), the
old examples contained **fractional "responder counts"** — percentage and
proportion values passed through unconverted into beta components, e.g.
`52.3/65`, `0.375/8`, `7.14/28`, `76.7/240` — and proportion-unit query
outcomes truncated to zero by `int()` (e.g. a held-out truth recorded as
`0/30` where the corrected value is `10/30`). Percentage rows whose value
exceeded the denominator had been silently dropped by the `0 ≤ y ≤ n` check;
they now convert correctly and re-enter the data. Net effect: June set 1,414
examples → corrected set 1,407.

## Headline results on corrected data (n = 1,407)

**Method ordering is preserved; every conclusion of the manuscript survives.**
NLL levels drop across the board (cleaner held-out targets), and the
two-head + SAM margin over the rule baseline is essentially unchanged.

### Borrowing head-to-head (mean held-out beta-binomial NLL)

| method | corrected | manuscript (pre-fix) |
|---|---|---|
| two_head_sam | **2.7938** | 2.9328 |
| two_head (trained) | 2.8957 | 3.0562 |
| rule_sam | 2.9006 | 2.9755 |
| fixed_discount | 2.9916 | — |
| rule | 3.0100 | 3.1800 |
| commensurate_like | 3.0325 | — |
| weak_only | 3.1641 | ~3.19 |
| map_like | 3.2219 | ~3.34 |
| power_prior_like | 3.2269 | ~3.36 |

95% interval coverage: two_head_sam 0.9964, two_head 0.9744, rule 0.9716.

**One narrative change to carry into the manuscript:** the trained two-head
prior (2.8957) now edges out rule_sam (2.9006), whereas pre-fix the order was
rule_sam < two_head_trained. The abstract sentence reporting that ordering
needs rewording; the overall message (conflict adaptation matters, learned +
SAM is best) is unchanged. The `two_head_trained` row loaded from
`lambda_nll_rows.csv` agrees with the embedded-prediction `two_head` row to
3e-7 — same internal consistency check as July. (Its `mean_historical_mass`
is blank by construction on that load path; cite the `two_head` row, which
carries mass 0.789.)

### Robust-MAP comparison

| method | mean NLL | coverage95 | mean ESS |
|---|---|---|---|
| two_head_sam | 2.7938 | 0.9964 | — |
| robust_map_w0.5 | 3.0019 | 0.9865 | 32.7 |
| robust_map_w0.8 | 3.0404 | 0.9616 | 32.7 |
| robust_map_w0.9 | 3.0996 | 0.9375 | 32.7 |

(Pre-fix reference: robust_map_w0.9 3.4784 / coverage 0.897.)

### Leakage-free rolling-origin forward validation (refit per cutoff)

Aggregate future-fold NLL: two_head_sam **2.7558** (was 2.880), rule_sam
2.8517 (2.923), two_head 2.8682 (2.984), rule 2.9563 (3.100), weak_only
3.0939 (3.113), map_like 3.1562, power_prior_like 3.1601. Coverage95 for the
SAM methods 0.9977.

Per-cutoff two_head_sam improvement vs rule, paired bootstrap 95% CI
(all exclude zero):

| cutoff | train / eval n | Δ NLL vs rule | 95% CI |
|---|---|---|---|
| 2020-12-31 | 632 / 775 | −0.1983 | [−0.2264, −0.1711] |
| 2021-12-31 | 813 / 594 | −0.2003 | [−0.2312, −0.1699] |
| 2022-12-31 | 1017 / 390 | −0.2028 | [−0.2463, −0.1639] |

rule_sam deltas: −0.1086 [−0.1313, −0.0880], −0.1049 [−0.1267, −0.0833],
−0.1004 [−0.1275, −0.0752].

Manuscript updates needed: fold sizes change from 629/810/1,017 (train) and
785/604/397 (eval) to **632/813/1,017** and **775/594/390**.

### Calibration diagnostics

Empirical coverage at nominal 0.50 / 0.80 / 0.95:
two_head 0.576 / 0.869 / 0.974; two_head_sam 0.622 / 0.915 / 0.996
(conservative direction); rule_sam 0.532 / 0.856 / 0.995;
map_like 0.389 / 0.689 / 0.945 and power_prior_like 0.399 / 0.680 / 0.942
(under-coverage for the naive borrowers — same qualitative story as before).
PIT histograms per method are in
`artifacts/rerun_unitfix_2026-08-14/calibration_diagnostics/`.

## Not affected / not re-run

- Gold-standard and external-control simulations: never touch the extraction
  path; their numbers stand.
- The two worked case studies (6/26 and 5/32, participant units): correct as
  printed.
- Stage 1 paired retrieval benchmark: retrieval scores do not use converted
  rates; unchanged.

## Artifacts

`artifacts/rerun_unitfix_2026-08-14/` (untracked, like all runtime
artifacts) contains: regenerated examples (`examples_unitfix.jsonl`, with
dates, with model), `failures_unitfix.jsonl`, primary training outputs
(`primary/`), fold models (`fold_models/`), and the four evaluation output
directories. The June artifacts in the worktree are untouched.

## Reproducing (Mac .venv confirmation pass)

```bash
cd oncology-trial-similarity
V=../.venv/bin/python
A=artifacts/rerun_unitfix_2026-08-14
$V pipeline/run_training_from_examples.py \
  --examples-jsonl $A/examples_unitfix.jsonl \
  --failures-jsonl $A/failures_unitfix.jsonl \
  --output-dir $A/primary_mac
for C in 2020-12-31 2021-12-31 2022-12-31; do
  $V pipeline/run_training_from_examples.py \
    --examples-jsonl $A/examples_unitfix_with_true_dates.jsonl \
    --output-dir $A/fold_models_mac --train-end-date $C
done
$V pipeline/two_head_inference.py --examples-jsonl $A/examples_unitfix.jsonl \
  --model-path $A/primary_mac/lambda_model.pt \
  --output-jsonl $A/examples_unitfix_with_model_mac.jsonl
$V pipeline/run_borrowing_baseline_comparison.py \
  --examples-jsonl $A/examples_unitfix_with_model_mac.jsonl \
  --output-dir $A/head_to_head_mac \
  --learned-nll-csv $A/primary_mac/lambda_nll_rows.csv
$V pipeline/run_forward_validation.py \
  --examples-jsonl $A/examples_unitfix_with_true_dates.jsonl \
  --output-dir $A/forward_mac \
  --model-path $A/primary_mac/lambda_model.pt \
  --fold-model-dir $A/fold_models_mac
```

To regenerate the examples themselves from the 2.5 GB pipeline results, see
`scripts/rebuild_lambda_examples_chunked.py`.

## Next step

Update `manuscript/manuscript_full_draft_pharm_stats.tex`: abstract and
Results numbers per the tables above, the two_head/rule_sam ordering
sentence, fold sizes, and rewrite the data-defect subsection from
"provisional pending re-run" to "corrected and re-run, pre-fix numbers
retained in supplement for transparency" (or drop the pre-fix numbers
entirely — author's call).
