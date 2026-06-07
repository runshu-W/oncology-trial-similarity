# Trial2Vec/SECRET Retrieval and Retrospective Mixture Prior Design

Date: 2026-06-02

## Objective

Modify the current oncology trial similarity and Bayesian prior-borrowing prototype in two focused ways:

1. Replace the current Stage-1 Bio_ClinicalBERT mean-pooling retriever with a stronger trial-level retrieval backend based on Trial2Vec or SECRET.
2. Replace the current single weighted beta-binomial power-prior approximation with a mixture prior whose top-candidate component weights `lambda_i` are learned by retrospective prediction, because expert borrowability labels are not yet available.

The intended system remains a feasibility and methods prototype. It should not claim validated clinical borrowing decisions until oncology/statistical expert adjudication is added.

## Current Baseline

The existing pipeline has four main layers:

1. Parse ClinicalTrials.gov-style JSON into a rule-based structured oncology summary.
2. Use multi-aspect Bio_ClinicalBERT embeddings for Stage-1 retrieval.
3. Use the deterministic Stage-2 prior-borrowing reranker to score disease biology, population, eligibility, regimen, endpoint, follow-up, design, and result usability.
4. For supported binary endpoints, build a weighted beta-binomial prior:

```text
alpha_prior = 1 + sum_i w_i y_i
beta_prior  = 1 + sum_i w_i (n_i - y_i)
```

where `w_i` is currently the reranker-derived borrowing discount.

The limitation is that `w_i` is used as a power-prior discount, not as a formal mixture component weight. The new design separates these two concepts.

## Proposed Architecture

```text
Query trial JSON/protocol
  -> structured query summary
  -> Stage-1 retrieval backend
       clinicalbert | trial2vec | secret
  -> top-K candidate summaries
  -> Stage-2 prior-borrowing reranker
  -> Stage-2 top10 with endpoint-matched quantities
  -> lambda model for mixture-prior component weights
  -> mixture prior
  -> posterior and retrospective predictive evaluation
```

Stage 1 is responsible for high-recall candidate discovery. Stage 2 remains responsible for clinical/statistical borrowability. The mixture-prior layer is responsible for converting reranked candidates into statistical prior components.

## Stage-1 Retrieval Modification

The retrieval layer should expose a backend-neutral interface:

```json
{
  "candidate_nct_id": "NCT...",
  "retrieval_score": 0.0,
  "retrieval_rank": 1,
  "retrieval_backend": "trial2vec",
  "candidate_summary": {}
}
```

Supported backends:

| Backend | Role | Implementation priority |
| --- | --- | --- |
| `clinicalbert` | Existing local baseline and fallback. | Keep for reproducibility. |
| `trial2vec` | Preferred first replacement. Uses trial-level embeddings trained for trial-to-trial similarity. | Implement first. |
| `secret` | Future stronger protocol-summary retriever. Useful when protocol/SAP summarization is available. | Implement after Trial2Vec. |

Stage 2 should consume the same candidate summary schema regardless of Stage-1 backend. This avoids coupling prior-borrowing logic to one embedding method.

## Stage-2 Reranker Retention

The current Stage-2 reranker should remain in place. It computes:

```text
overall_score_i =
  0.75 * structured_clinical_score_i
  + 0.25 * retrieval_score_i
```

It also outputs:

```text
dimension_scores_i
red_flags_i
suitability_i
discount_i
borrowable_quantities_i
```

The value `discount_i`, also written as `a_i` below, remains the information discount inside each historical component. It is not the same as the mixture component weight `lambda_i`.

## Mixture Prior Definition

For each candidate `i` in the Stage-2 top10, define:

```text
y_i = endpoint-specific event count
n_i = endpoint-specific denominator
a_i = borrowing discount from Stage 2, in [0, 1]
lambda_i = learned mixture-prior component weight
lambda_0 = weak-prior component weight
```

Each historical component is:

```text
p_i(p) = Beta(alpha_i, beta_i)

alpha_i = 1 + a_i y_i
beta_i  = 1 + a_i (n_i - y_i)
```

The weak component is:

```text
p_0(p) = Beta(alpha_0, beta_0)
```

with default:

```text
alpha_0 = 1
beta_0  = 1
```

The final prior is:

```text
p_prior(p) =
  lambda_0 * p_0(p)
  + sum_{i=1}^{10} lambda_i * p_i(p)
```

Subject to:

```text
lambda_0 >= 0
lambda_i >= 0
lambda_0 + sum_i lambda_i = 1
```

Recommended first implementation:

```text
lambda_0 = fixed weak-prior weight, e.g. 0.20
sum_i lambda_i = 0.80
```

The fixed `lambda_0` gives the mixture prior a robust weak component and makes the first version easier to interpret.

## Lambda Model

For candidate `i`, construct a feature vector:

```text
x_i = [
  retrieval_score_i,
  overall_score_i,
  disease_match_i,
  regimen_match_i,
  endpoint_match_i,
  followup_match_i,
  eligibility_match_i,
  result_quality_i,
  redflag_severity_i,
  log(1 + n_i),
  observed_rate_i
]
```

where:

```text
disease_match_i     = disease_biology_match_i / 5
regimen_match_i     = treatment_regimen_match_i / 5
endpoint_match_i    = endpoint_estimand_match_i / 5
followup_match_i    = outcome_assessment_followup_i / 5
eligibility_match_i = eligibility_criteria_overlap_i / 5
result_quality_i    = result_usability_i / 5
observed_rate_i     = y_i / n_i
```

The lambda model produces a raw utility:

```text
u_i = f_theta(x_i)
```

The usable raw weight is:

```text
r_i = I_i * G_i * exp(u_i / tau)
```

where:

```text
I_i = 1 if endpoint-matched count/denominator are usable, otherwise 0
G_i = product of conservative gates for endpoint, result, disease, and red flags
tau = softmax temperature
```

Normalize to mixture weights:

```text
lambda_i =
  (1 - lambda_0) * r_i / sum_k r_k
```

If no candidate has usable endpoint-matched data, the system should set:

```text
lambda_0 = 1
lambda_i = 0 for all i
```

and report that the prior is weak-only.

## Retrospective Prediction Training

Because expert labels are unavailable, train `f_theta` with retrospective prediction.

For each completed trial `q` that has a supported endpoint with count/denominator:

1. Treat `q` as a pseudo-query.
2. Hide `q`'s endpoint result during retrieval, Stage-2 scoring, and lambda feature construction.
3. Retrieve candidates with Trial2Vec or SECRET.
4. Rerank candidates with the existing Stage-2 prior-borrowing reranker.
5. Build mixture-prior components from the Stage-2 top10.
6. Predict the held-out result `(y_q, n_q)` using the mixture prior.
7. Optimize the lambda model to maximize held-out predictive probability.

For component `i`, the beta-binomial predictive probability for the held-out query result is:

```text
P_i(y_q | n_q) =
  choose(n_q, y_q)
  * B(y_q + alpha_i, n_q - y_q + beta_i)
  / B(alpha_i, beta_i)
```

For the weak component:

```text
P_0(y_q | n_q) =
  choose(n_q, y_q)
  * B(y_q + alpha_0, n_q - y_q + beta_0)
  / B(alpha_0, beta_0)
```

The mixture predictive probability is:

```text
P(y_q | n_q, mixture) =
  lambda_0 * P_0(y_q | n_q)
  + sum_i lambda_i * P_i(y_q | n_q)
```

Training loss:

```text
Loss_predictive =
  -log P(y_q | n_q, mixture)
```

## Rule-Based Anchor and ESS Protection

To avoid unstable black-box weighting with limited retrospective data, define a rule-based anchor:

```text
r_i_rule =
  I_i
  * G_i
  * a_i
  * overall_score_i / 100
  * log(1 + n_i)
```

```text
lambda_i_rule =
  (1 - lambda_0) * r_i_rule / sum_k r_k_rule
```

Use it as a regularizer:

```text
Loss_total =
  Loss_predictive
  + rho * KL(lambda_rule || lambda_model)
  + eta * ESS_penalty
```

where:

```text
ESS = sum_i lambda_i * a_i * n_i
ESS_penalty = max(0, ESS - ESS_cap)^2
```

Recommended first values:

```text
lambda_0 = 0.20
ESS_cap = 100
rho > 0
eta > 0
```

These constants should be treated as sensitivity parameters, not validated defaults.

## Posterior Updating

After observing query data `(y_0, n_0)`, each component updates conjugately:

```text
p_i(p | y_0, n_0) =
  Beta(alpha_i + y_0, beta_i + n_0 - y_0)
```

The posterior component weights update by Bayes rule:

```text
lambda_i_post =
  lambda_i * P_i(y_0 | n_0)
  / [lambda_0 * P_0(y_0 | n_0) + sum_k lambda_k * P_k(y_0 | n_0)]
```

Important distinction:

```text
lambda_i      = prior mixture weight, fixed before seeing query outcome
lambda_i_post = posterior mixture weight, updated after seeing query outcome
```

The system must not use query outcome information to compute `lambda_i`.

## Data Leakage Controls

The retrospective training path must enforce:

1. The pseudo-query result `(y_q, n_q)` is used only in the loss.
2. The pseudo-query itself is excluded from the candidate pool.
3. Any duplicate, companion, sponsor-related, or overlapping-arm trials should be flagged and excluded when automated detection exists; otherwise they should be reported as a limitation.
4. A temporal split is preferred: when evaluating pseudo-query `q`, only use historical trials whose results were available before `q`'s result posting date.
5. Hyperparameters must be selected on validation pseudo-queries, not on the final test set.

## Evaluation Plan

Primary retrospective metrics:

```text
negative log predictive density
Brier score for selected probability thresholds
calibration of Pr(p >= threshold)
coverage-like behavior of credible intervals
ESS distribution
prior-data conflict frequency
```

Baselines:

| Baseline | Purpose |
| --- | --- |
| Weak-only prior | Shows value over no borrowing. |
| Current weighted beta-binomial power prior | Compares against current implementation. |
| Rule-based mixture weights | Checks whether learning improves over transparent rules. |
| Uniform top10 mixture | Checks whether ranking/weighting adds value. |
| Stage-1-only weighting | Shows whether Stage 2 adds prior-borrowing value. |

The model should be reported as better only if it improves retrospective prediction and does not create unacceptable ESS inflation or prior-data conflict behavior.

## Implementation Scope

First implementation should include:

1. Add retrieval backend abstraction.
2. Add Trial2Vec retrieval backend.
3. Keep ClinicalBERT backend as fallback.
4. Preserve existing Stage-2 reranker.
5. Add rule-based mixture prior baseline.
6. Add retrospective training dataset builder.
7. Add small lambda model, starting with logistic/MLP scoring on structured features.
8. Add predictive-likelihood training and validation.
9. Add reports for `lambda_i`, `lambda_i_rule`, `lambda_i_post`, ESS, and predictive metrics.

SECRET should be treated as a second-phase backend after the Trial2Vec path works, because SECRET requires a reliable protocol summarization pipeline.

## Non-Goals

This design does not:

1. Claim expert-validated borrowing suitability.
2. Replace oncology/statistical expert review.
3. Implement a full robust MAP or commensurate prior in the first revision.
4. Train a new Trial2Vec or SECRET model from scratch.
5. Use query outcomes to determine prior weights at deployment time.

## Recommended Manuscript Framing

The revised method can be described as:

```text
We replace the first-stage ClinicalBERT retriever with a trial-level retrieval
backend based on Trial2Vec or SECRET. Retrieved candidates are passed to the
existing prior-borrowing-oriented clinical-statistical reranker. For Bayesian
synthesis, the top reranked candidates are represented as mixture-prior
components. Component weights are learned without expert labels by retrospective
prediction: completed trials are treated as pseudo-queries, their outcomes are
hidden during retrieval and weighting, and the mixture weights are optimized to
maximize the beta-binomial predictive likelihood of the held-out query outcome.
```

## References

- Wang Z, Sun J. Trial2Vec: Zero-Shot Clinical Trial Document Similarity Search using Self-Supervision. Findings of EMNLP 2022.
- Das et al. SECRET: Semi-supervised Clinical Trial Document Similarity Search. ACL 2025.
- Ibrahim JG, Chen MH. Power prior distributions for regression models. Statistical Science. 2000.
- Hobbs BP, Carlin BP, Mandrekar SJ, Sargent DJ. Hierarchical commensurate and power prior models for adaptive incorporation of historical information in clinical trials. Biometrics. 2011.
- Schmidli H, Gsteiger S, Roychoudhury S, O'Hagan A, Spiegelhalter D, Neuenschwander B. Robust meta-analytic-predictive priors in clinical trials with historical control information. Biometrics. 2014.
