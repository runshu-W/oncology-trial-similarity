# SECRET-Style Retrospective-Calibrated Mixture Prior Design

Date: 2026-06-03

## Objective

Implement the next four items from the update report's "Current limitations and next steps":

1. Make `secret` a runnable Stage-1 backend instead of a future unsupported backend.
2. Expand retrospective lambda training from the current compact feature vector to the full `x_i` design.
3. Add retrospective evaluation with pseudo-query leakage controls.
4. Promote mixture prior from a sensitivity-only artifact to an optional retrospective-calibrated mode.

The user does not have expert review labels. Therefore this design uses retrospective predictive performance as the training and calibration signal. It must not claim expert-validated borrowing decisions.

## Scope Boundary

This is an engineering MVP, not a full reproduction of the SECRET paper.

The implemented `secret` backend will be called **SECRET-style**:

- It creates deterministic Q/A-style section summaries from the structured ClinicalTrials.gov JSON and optional protocol/SAP excerpts already available in the pipeline.
- It retrieves candidates using section-level similarity and exposes section evidence.
- It does not train SECRET's original contrastive representation model.
- It does not require expert labels or external LLM calls.

This boundary must be visible in code, CLI help, and documentation so the pipeline does not overclaim.

## Existing Baseline

Current feature branch already has:

- `clinicalbert`, `hashing`, and `trial2vec` Stage-1 paths.
- `secret` listed in `RETRIEVAL_BACKENDS`, but `ensure_supported_retrieval_backend("secret")` raises `NotImplementedError`.
- Stage-2 reranker with:
  - `overall_similarity_score`
  - `dimension_scores`
  - `red_flags`
  - `suggested_borrowing_discount`
  - `borrowable_quantities`
- Rule-based mixture prior:
  - `lambda_0 = 0.2`
  - candidate `lambda_rule` values normalized over top10.
- Retrospective training script:
  - can read `examples.jsonl` or `pipeline-results.jsonl`.
  - uses compact features:

```text
[
  overall_similarity_score / 100,
  observed_rate,
  discount,
  log(1 + n),
  gate
]
```

## Proposed Architecture

```text
Historical trial JSON/PDF
  -> structured summary
  -> SECRET-style Q/A sections
  -> secret section embeddings index

Query trial JSON/PDF
  -> structured query summary
  -> query SECRET-style Q/A sections
  -> section-weighted retrieval
  -> candidate section evidence
  -> existing Stage-2 prior-borrowing reranker
  -> full x_i lambda features
  -> rule lambda or learned lambda model
  -> mixture prior
  -> retrospective calibration/evaluation report
```

The important separation remains:

- Stage 1: high-recall retrieval.
- Stage 2: structured prior-borrowing suitability.
- Mixture prior: statistical prior construction.
- Retrospective calibration: no-expert substitute signal based on held-out outcomes.

## 1. SECRET-Style Backend

### Data Model

Create a small focused module, tentatively:

```text
docs/secret_retrieval.py
```

It should expose:

```python
SECRET_SECTIONS = (
    "disease_population",
    "intervention",
    "eligibility",
    "endpoint",
    "design",
    "results",
    "safety_followup",
)

SECRET_SECTION_WEIGHTS = {
    "disease_population": 0.22,
    "intervention": 0.20,
    "eligibility": 0.16,
    "endpoint": 0.18,
    "design": 0.10,
    "results": 0.10,
    "safety_followup": 0.04,
}
```

For each structured trial summary, build deterministic Q/A section strings:

```text
Q: What disease and population does the trial study?
A: ...

Q: What intervention or regimen is tested?
A: ...

Q: What eligibility criteria define the target population?
A: ...

Q: What primary endpoint and estimand are measured?
A: ...

Q: What design, phase, arm structure, and randomization are used?
A: ...

Q: What result quantities are available for borrowing?
A: ...

Q: What safety and follow-up information is available?
A: ...
```

The section builder should use existing normalized fields first and optional protocol/SAP excerpts second. If the excerpt exists, it should be included only as bounded evidence text, not as a giant full-document blob.

### Index Format

Add a secret index file:

```text
secret_embeddings.npz
```

Required arrays:

```text
nct_ids
retrieval_backend = ["secret"]
embedding_backend = ["clinicalbert" or "hashing"]
embedding_model
<one array per SECRET section>
```

Each section array has shape:

```text
num_trials x embedding_dim
```

### CLI

Add a build command:

```bash
python3 docs/oncology_trial_similarity_pipeline.py build-secret-index \
  --index-dir artifacts/oncology_trial_similarity_clinicalbert \
  --output-path artifacts/oncology_trial_similarity_clinicalbert/secret_embeddings.npz \
  --embedding-backend clinicalbert
```

Add search support:

```bash
python3 docs/oncology_trial_similarity_pipeline.py search \
  --retrieval-backend secret \
  --secret-index-path artifacts/oncology_trial_similarity_clinicalbert/secret_embeddings.npz \
  --index-dir artifacts/oncology_trial_similarity_clinicalbert \
  --query-json /path/to/query.json \
  --rerank
```

### Search Output

Candidate rows returned by Stage 1 should include:

```json
{
  "nct_id": "NCT...",
  "score": 0.0,
  "score_0_100": 0.0,
  "retrieval_backend": "secret",
  "secret_section_scores": {
    "disease_population": 0.0,
    "intervention": 0.0,
    "eligibility": 0.0,
    "endpoint": 0.0,
    "design": 0.0,
    "results": 0.0,
    "safety_followup": 0.0
  },
  "secret_evidence": {
    "disease_population": "...",
    "intervention": "...",
    "eligibility": "...",
    "endpoint": "...",
    "design": "...",
    "results": "...",
    "safety_followup": "..."
  }
}
```

Stage 2 should continue to consume the same enriched candidate summary schema.

## 2. Full `x_i` Feature Vector

The lambda training feature vector should become:

```text
x_i = [
  s_i,
  disease_match_i,
  regimen_match_i,
  endpoint_match_i,
  followup_match_i,
  eligibility_match_i,
  result_quality_i,
  -redflag_severity_i,
  log(n_i)
]
```

Definitions:

| Feature | Source | Normalization |
| --- | --- | --- |
| `s_i` | `overall_similarity_score` | divide by 100 |
| `disease_match_i` | `dimension_scores.disease_population_match` | divide by 5 |
| `regimen_match_i` | `dimension_scores.treatment_regimen_match` | divide by 5 |
| `endpoint_match_i` | `dimension_scores.endpoint_estimand_match` | divide by 5 |
| `followup_match_i` | `dimension_scores.safety_and_followup_relevance` | divide by 5 |
| `eligibility_match_i` | new `dimension_scores.eligibility_criteria_overlap` | divide by 5 |
| `result_quality_i` | `dimension_scores.result_usability` | divide by 5 |
| `-redflag_severity_i` | rule-based severity from `red_flags` | negative normalized value |
| `log(n_i)` | selected borrowable endpoint denominator | `log1p(n_i)` |

### Eligibility Score

Add a deterministic eligibility score to Stage 2:

```text
eligibility_criteria_overlap in [0, 5]
```

MVP rule:

- Extract normalized tokens from query/candidate `population.key_inclusion` and `population.key_exclusion`.
- Remove generic tokens such as `adult`, `participant`, `patient`, `study`, `adequate`, `available`.
- Compute Jaccard overlap of informative tokens.
- Convert to 0-5:

```text
eligibility_score = round(5 * jaccard_overlap, 2)
```

If both sides lack eligibility text, use `0.0` and add a red flag:

```text
"Eligibility overlap unavailable."
```

### Red Flag Severity

Add a pure helper:

```python
redflag_severity(red_flags: list[str]) -> float
```

MVP scoring:

- `Low disease` or `Low endpoint`: +1.0 each.
- `No primary endpoint-family overlap`: +1.0.
- `No normalized regimen-backbone overlap`: +0.8.
- `No posted results` or `No arm-level count/denominator`: +0.8.
- other flags: +0.25 each.

Normalize:

```text
severity = min(raw / 3.0, 1.0)
```

Then feature value is `-severity`.

## 3. Retrospective Evaluation Without Expert Labels

Create a dedicated evaluation script:

```text
scripts/evaluate_retrospective_lambda_model.py
```

Inputs:

```bash
python3 scripts/evaluate_retrospective_lambda_model.py \
  --examples-jsonl artifacts/retrospective_lambda_examples.jsonl \
  --output-json artifacts/retrospective_lambda_evaluation.json \
  --train-fraction 0.7 \
  --seed 20260603 \
  --epochs 100 \
  --hidden-dim 16
```

It should also support:

```bash
--pipeline-results-jsonl artifacts/pseudo_query_results.jsonl
```

### Leakage Rule

The script must write this assumption into the output JSON:

```text
Input examples must have been generated with query outcomes hidden from retrieval, reranking, feature construction, lambda selection, and model selection. The held-out outcome is used only for predictive loss and evaluation.
```

The evaluation script cannot prove upstream leakage is absent, but it can:

- validate required fields.
- refuse empty examples.
- record the source file and split seed.
- split examples deterministically.
- keep train/eval examples disjoint by row index.

### Metrics

Report at least:

```json
{
  "example_count": 0,
  "train_count": 0,
  "eval_count": 0,
  "metrics": {
    "weak_only_mean_nll": 0.0,
    "rule_lambda_mean_nll": 0.0,
    "learned_lambda_mean_nll": 0.0,
    "learned_minus_rule_mean_nll": 0.0
  }
}
```

Interpretation:

- Lower NLL is better.
- If `learned_lambda_mean_nll < rule_lambda_mean_nll`, retrospective calibration improved predictive fit on held-out pseudo-queries.
- This does not replace expert review.

## 4. Retrospective-Calibrated Mixture Prior Mode

The search output should keep rule-based mixture prior by default.

Add optional calibrated mode:

```bash
python3 docs/oncology_trial_similarity_pipeline.py search \
  --query-json /path/to/query.json \
  --index-dir artifacts/oncology_trial_similarity_clinicalbert \
  --rerank \
  --lambda-model-path artifacts/lambda_model.pt \
  --mixture-prior-mode retrospective_calibrated
```

Mode behavior:

| Mode | Behavior |
| --- | --- |
| `rule` | Current rule-based `lambda_rule` values remain active. |
| `retrospective_calibrated` | Load lambda scorer and compute `lambda_model`; active component weights come from the model. |

Output should include both rule and model values when the model is used:

```json
{
  "mixture_prior": {
    "mode": "retrospective_calibrated",
    "lambda_0": 0.2,
    "components": [
      {
        "lambda_rule": 0.12,
        "lambda_model": 0.10,
        "lambda_active": 0.10
      }
    ],
    "calibration_note": "No expert labels were used; lambda_model was trained by retrospective predictive loss."
  }
}
```

If `retrospective_calibrated` is requested without a valid model path, search should raise a clear `ValueError`.

## Model Artifact

Update `scripts/train_retrospective_lambda_model.py` to optionally save a model artifact:

```bash
python3 scripts/train_retrospective_lambda_model.py \
  --examples-jsonl artifacts/retrospective_lambda_examples.jsonl \
  --output-json artifacts/lambda_training_summary.json \
  --model-output artifacts/lambda_model.pt
```

The artifact should include:

```python
{
  "state_dict": model.state_dict(),
  "input_dim": 9,
  "hidden_dim": hidden_dim,
  "feature_names": [...],
  "lambda0": 0.2,
}
```

The pipeline should validate:

- `input_dim == len(feature_names) == 9`.
- expected feature names match the current full `x_i` order.
- model file exists.

## Testing Strategy

Use TDD for each behavior.

Required tests:

1. SECRET-style summary construction:
   - deterministic sections exist.
   - eligibility section uses inclusion/exclusion text.
   - protocol/SAP excerpt is bounded.

2. SECRET index search:
   - synthetic section embeddings produce expected ranking.
   - result includes `secret_section_scores` and `secret_evidence`.
   - `ensure_supported_retrieval_backend("secret")` no longer raises unsupported error.

3. Full lambda feature vector:
   - `build_training_example_from_pipeline_result` emits 9 features.
   - feature order and values match expected normalized dimension scores.
   - red flag severity is negative in the feature vector.

4. Retrospective evaluation:
   - deterministic train/eval split.
   - outputs weak-only, rule, and learned NLL metrics.
   - rejects empty or invalid examples.

5. Retrospective-calibrated mixture prior mode:
   - search refuses calibrated mode without model.
   - with a tiny deterministic model artifact, output includes `lambda_model` and `lambda_active`.
   - rule mode remains default and backward compatible.

6. Full verification:
   - all unit tests pass.
   - py_compile passes for changed Python files.

## Documentation Updates

Update the report:

```text
docs/trial2vec_secret_mixture_prior_update_report_2026-06-03.md
```

with:

- SECRET-style backend status changed from future to runnable MVP.
- full `x_i` now implemented.
- retrospective evaluation workflow.
- calibrated mixture-prior mode.
- caveat that no expert review labels were used.

## Success Criteria

Implementation is complete when:

1. `--retrieval-backend secret` runs against a built secret index.
2. Lambda training examples use the 9-feature `x_i`.
3. A retrospective evaluation JSON can be generated without expert labels.
4. A trained lambda model artifact can be used in search with `--mixture-prior-mode retrospective_calibrated`.
5. Rule mode remains the default and existing outputs/tests remain compatible.
6. Full test suite and py_compile pass.

## Confidence Check

Pre-implementation confidence estimate: 0.92.

- No duplicate runnable SECRET backend exists; current code only has an unsupported guardrail.
- Architecture matches the existing pattern: pipeline orchestration stays in `docs/oncology_trial_similarity_pipeline.py`, focused math/features live in small modules, CLI scripts handle offline workflows.
- No new network or external model dependency is required for the SECRET-style MVP.
- The root cause of each limitation is clear from the current implementation.
- The only deliberate tradeoff is that SECRET-style retrieval is a deterministic Q/A-section implementation, not a full contrastive SECRET reproduction.
