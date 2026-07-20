# 3. Case study

Worked single-query examples on real trials: take one query trial, rebuild the
historical mixture prior under every borrowing method, and score each against
that trial's held-out outcome.

This is the real-data counterpart to `../simulation/`. The crucial difference is
what can be concluded. In the simulation the correct borrowing decision is known,
so methods can be graded on *whether they borrowed from the right trials*. Here
there is no such truth, so methods are compared only on held-out predictive
performance and on how much information they borrowed. A case study can show
that a method behaves sensibly on real registry text; it cannot show that its
donor selection was correct.

## Running

```bash
python3 run_case_study.py \
    --query-dir  path/to/pseudo_query_results \
    --corpus     path/to/trial_summaries.jsonl \
    --model      ../artifacts/gold_standard_simulation/two_head_pro.npz \
    --top 2
```

- `--query-dir` / `--query-json` — pseudo-query results from
  `../pipeline/run_oncology_retrospective_lambda_training.py`
- `--corpus` — required in practice: the retrieval snapshot inside a pseudo-query
  file carries only denominators, so candidate arm-level outcomes are read from
  the structured corpus
- `--model` — optional; without it the learned two-head prior is skipped and only
  the closed-form comparators run

Outputs `case_study_comparison.csv`, `case_study_metadata.json` and
`table_case_studies.tex` under `artifacts/case_studies/`.

## Our method versus other methods

The prior implementations are imported from `../simulation/gold_standard/methods.py`
rather than reimplemented, so the case study and the simulation are provably
running the same code. The same prospective/retrospective labelling applies: the
output marks each method as available at design time or not, and `rule_sam`,
`two_head_pro_sam` and `uip_js` are flagged because they consult the query
outcome.

## Data handling

Endpoint values are converted with `../pipeline/fix_endpoint_units.py`, which
dispatches on the reported unit. This runner therefore does **not** inherit the
defect described in `../docs/KNOWN_ISSUE_endpoint_units.md`; candidates whose
unit cannot be interpreted are dropped and counted in the metadata rather than
silently converted. Expect a substantial drop rate: many oncology records report
no usable arm-level response endpoint at all, and the metadata file records how
many candidates were lost for each reason.

## Status

The runner is new in this revision. The two ORR case studies quoted in the
manuscript predate it and were produced by the retrospective pipeline directly;
the external-control case study is deferred until the endpoint-unit defect is
fixed end to end.
