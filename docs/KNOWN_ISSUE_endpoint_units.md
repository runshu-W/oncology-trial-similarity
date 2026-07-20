# Known issue: endpoint-unit handling when converting outcomes to rates

**Status:** open. Diagnosed and quantified; fix deliberately not applied
piecemeal (see "Why this is not yet fixed").

## The defect

ClinicalTrials.gov reports an outcome value together with a *unit of measure*.
For objective response rate the unit is sometimes `Participants` (the value is a
responder count) and sometimes `Percentage of Participants` (the value is
already a percentage). The pipeline converts the value to a rate by dividing by
the arm denominator unconditionally, which is only correct in the first case.

A row reading `26.0` with denominator `31` under `Percentage of Participants`
denotes an ORR of **0.26**. The current code produces **0.839**.

## Measured impact

From `scripts/audit_orr_units.py` over the ORR corpus:

| | |
|---|---|
| Arm-level ORR rows | 3319 |
| Reported in percentage units | 2040 |
| Reported in participant counts | 846 |
| Rows assigned a rate above 1 (provably wrong) | 804 (24.2%) |

The 804 impossible values are the *detectable* portion. Rows where the bad value
happens to land inside (0, 1) look entirely plausible and cannot be spotted from
the output — this is what makes the defect dangerous rather than merely obvious.

Within the 120-query borrowing dataset: of 93 queries whose originating row
could be located, **9 change** under corrected unit handling and **8 change by
more than 0.10** in absolute rate, the largest error being **0.579**
(NCT01178944: 0.839 used, 0.260 correct). Because the held-out query outcome is
the reference for predictive NLL, calibration and coverage, aggregate real-data
results are affected to that extent.

Not affected: the two worked case studies in the manuscript (participant units,
6/26 and 5/32), and the gold-standard and external-control simulations, which
never touch this path.

## Affected call sites

The conversion is not centralised; it is reimplemented independently at each of
these, and **all four must be fixed together**:

| Location | Code |
|---|---|
| `docs/oncology_trial_similarity_pipeline.py:296` | `result["proportion"] = round(float(result["count"]) / float(denominator), 6)` |
| `docs/oncology_trial_similarity_pipeline.py:1703` | `"rate": count / denominator` |
| `docs/mixture_prior.py:134` | `return count, denominator, count / denominator` |
| `scripts/train_retrospective_lambda_model.py:613` | derives the query rate from count/denominator |

Note that `unit` is currently read *after* the `arm_results` loop in
`extract_outcomes` (line 305), so the fix requires hoisting it above the loop.

## Why this is not yet fixed

Patching one site would leave the others wrong while making the codebase
internally inconsistent — some rates correct, some not, with no way to tell them
apart downstream. That is worse than a uniform, documented defect.

A correct fix is:

1. Centralise the conversion (`scripts/fix_endpoint_units.py` already implements
   and tests the correct dispatch).
2. Thread `unit` through to all four call sites.
3. Regenerate the affected artifacts from the raw ClinicalTrials.gov records.
4. Re-run the retrospective evaluation and update the reported aggregates.

Step 3 requires the raw export under `Oncology_All_Trials/`, which is not
tracked in this repository.

## Tooling already available

- `scripts/fix_endpoint_units.py` — unit-aware conversion. Dispatches on the
  reported unit and raises `UnitError` rather than guessing when the unit is
  missing or unrecognised (29 such rows in the corpus).
- `scripts/audit_orr_units.py` — reproduces the numbers above and writes a
  per-query correction table:

  ```bash
  python3 scripts/audit_orr_units.py \
      --corpus path/to/trial_summaries.jsonl \
      --lambda-features path/to/lambda_component_features.csv
  ```

  Outputs `artifacts/orr_unit_audit/{unit_audit_summary.json,query_corrections.csv}`.
