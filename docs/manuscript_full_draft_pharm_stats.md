# Full Manuscript Draft for Pharmaceutical Statistics / Statistics in Medicine

## Terminology Ledger

| Canonical term | Definition in this manuscript | Notes |
|---|---|---|
| Historical borrowing | Use of historical trial evidence to construct or inform a prior distribution for a new trial. | Avoid using "trial matching" as a synonym. |
| Borrowability | Suitability of a historical trial to contribute prior evidence after considering clinical, endpoint, result, and conflict dimensions. | Distinct from text similarity. |
| Stage 1 retrieval | High-recall retrieval of candidate historical trials. | Backends include hashing, ClinicalBERT, Trial2Vec-style, and SECRET-style retrieval. |
| SECRET pool | Section-weighted candidate pool intended to improve borrowability-relevant retrieval. | Current implementation is deterministic and section based. |
| Stage 2 reranker | Explainable candidate reranking using borrowability-relevant features. | Not an expert decision. |
| Endpoint observation | Extracted pair `(y_i, n_i)` from a candidate trial endpoint. | Used to construct beta-binomial components. |
| Effective sample-size discount | Candidate-specific discount `a_i` applied inside the beta component. | Distinct from mixture weight. |
| Mixture weight | Candidate-specific component weight `lambda_i`. | Normalized with weak-prior weight `lambda_0`. |
| SAM adapter | Prior-data conflict adapter used as a sensitivity analysis. | Not claimed as regulatory qualification. |
| Retrospective predictive calibration | Evaluation using held-out pseudo-query outcomes. | Not expert validation or prospective validation. |

## Title Candidates

1. Retrospective predictive calibration of oncology trial similarity for Bayesian historical borrowing
2. From trial similarity to borrowability: an explainable mixture-prior framework for oncology historical evidence
3. Explainable retrieval and mixture-prior calibration for Bayesian historical borrowing in oncology
4. True-date retrospective validation of oncology trial retrieval for Bayesian historical borrowing
5. A no-expert-label evidence package for oncology trial similarity and Bayesian historical borrowing

Recommended title: **Retrospective predictive calibration of oncology trial similarity for Bayesian historical borrowing**.

## Structured Abstract

### Background

Bayesian historical borrowing can improve the efficiency of oncology trials, but the choice of historical evidence is a central methodological and regulatory concern. Textual similarity between trial records is insufficient for borrowing because historical evidence must also be compatible in disease setting, patient population, regimen, endpoint definition, follow-up, eligibility criteria, result usability, information size, and prior-data conflict risk.

### Methods

We developed an explainable pipeline for oncology trial retrieval and Bayesian mixture-prior construction from ClinicalTrials.gov records. Trial records were converted into structured summaries, retrieved through Stage 1 backends including hashing and SECRET-style section retrieval, and reranked using borrowing-relevant clinical and statistical features. Candidate endpoint observations were converted into beta-binomial historical components, with effective sample-size discounts and mixture weights represented as distinct quantities. A two-head DeepSets model was used where trained retrospective predictions were available, and a SAM-style conflict adapter was evaluated as a transparent sensitivity approach. Because expert borrowability labels were unavailable, validation used leakage-controlled pseudo-queries, held-out beta-binomial predictive negative log-likelihood, true-date temporal subset validation, simulation operating characteristics, paired backend benchmarking, baseline comparisons, and feature ablation.

### Results

In objective response rate pseudo-queries, SECRET pool retrieval improved reranked component-readiness over hashing by 0.0912 with a paired bootstrap confidence interval of 0.0799 to 0.1018, and improved endpoint-match score by 1.1047 with a confidence interval of 1.0600 to 1.1503. In borrowing baseline comparisons, the trained two-head mixture model achieved mean negative log-likelihood 3.0181 compared with 3.1620 for the rule-based mixture; the conflict-adapted rule baseline achieved 2.9721. True-date temporal subset analyses using ClinicalTrials.gov primary completion dates showed consistent improvements for calibrated and conflict-adapted methods across date-based and rolling-origin subsets. Simulation studies across exchangeable, conflict, mixture-conflict, and heterogeneous historical scenarios quantified type I error, power, bias, mean squared error, coverage, and SAM trigger behavior.

### Conclusions

The framework provides a reproducible methodology prototype and retrospective predictive calibration evidence package for oncology historical borrowing. It supports transparent method development and sensitivity analysis, but does not replace expert clinical and statistical review of historical evidence.

## One-Sentence Argument

In oncology historical borrowing, we show that trial retrieval can be reframed as borrowability-oriented evidence selection using structured trial summaries, explainable candidate features, beta-binomial mixture components, and retrospective calibration evidence, with the boundary that the current evidence does not include expert borrowability labels or prospective validation.

## 1. Introduction

Bayesian borrowing from historical trials is attractive in oncology because many studies are small, single-arm, disease-specific, or constrained by limited eligible populations. A well-calibrated historical prior can improve precision and reduce unnecessary uncertainty, but inappropriate borrowing can bias estimates, distort operating characteristics, and weaken interpretability. The practical question is therefore not only how to construct a prior, but how to identify historical trials that are suitable to contribute prior evidence. [Citation needed: Bayesian historical borrowing in clinical trials.] [Citation needed: regulatory guidance on external controls and historical borrowing.]

Existing trial similarity methods can retrieve records that share terms, interventions, or broad disease labels, but borrowing requires a narrower and more structured notion of comparability. Two oncology trials can be textually similar while differing in treatment line, endpoint definition, response assessment window, follow-up duration, eligibility restrictions, availability of result tables, or red flags such as incompatible arms or immature data. Conversely, a trial with less obvious surface similarity may contain precisely the endpoint observation needed for a calibrated historical component. This distinction motivates a pipeline that separates retrievability from borrowability.

We developed a methodology prototype that links oncology trial retrieval to Bayesian mixture-prior construction. ClinicalTrials.gov records are converted into structured summaries, candidate historical trials are retrieved through Stage 1 backends, SECRET-style section retrieval forms a high-recall candidate pool, and an explainable Stage 2 reranker scores candidates on borrowing-relevant dimensions. Extractable endpoint observations are converted into beta-binomial historical components. Candidate mixture weights and effective sample-size discounts are treated as distinct quantities, allowing the method to separate component allocation from within-component information borrowing.

The present study evaluates this framework without expert borrowability labels. This constraint is important. We do not claim that the pipeline can adjudicate clinical borrowing suitability. Instead, we assemble a reproducible retrospective evidence package using leakage-controlled pseudo-queries, held-out predictive negative log-likelihood (NLL), true-date temporal subset analyses, simulation operating characteristics, paired Stage 1 retrieval benchmarking, baseline prior comparisons, and feature ablation. This design supports method development and transparent calibration while preserving the boundary between automated retrospective evidence and expert clinical-statistical review.

## 2. Methods

### 2.1 Data source and structured trial summaries

The pipeline uses local ClinicalTrials.gov oncology trial records. Each record is parsed into a structured summary that includes disease context, patient population, regimen, endpoint text, follow-up or time-frame text, eligibility criteria, result availability, extractable endpoint observations, and red flags. The structured summary is the shared representation used by retrieval, reranking, candidate feature construction, and temporal validation.

True date metadata are extracted separately to support temporal validation. The date extraction module reads newer ClinicalTrials.gov API-style records and local legacy exported JSON structures. It extracts primary completion date, completion date, results first posted date, and start date. Dates are normalized with precision labels, including day, month, year, missing, and unparseable. Month-only dates are mapped to deterministic mid-month anchors for sorting while retaining the precision label. Primary completion date is preferred for temporal sorting, followed by completion date, results first posted date, start date, and NCT numeric proxy only when true date metadata are unavailable.

### 2.2 Leakage-controlled pseudo-query construction

Completed trials are used retrospectively as pseudo-queries. For each pseudo-query, the query outcome is hidden from retrieval and reranking. Candidate selection uses only structured trial information and historical candidate records. The held-out pseudo-query endpoint observation is used only for retrospective predictive evaluation. Pipeline-result rows used for training or evaluation require explicit leakage-control metadata and held-out query outcomes; rows lacking those fields are rejected by the training and evaluation code.

The current empirical evidence package focuses on objective response rate (ORR) pseudo-queries because ORR endpoint observations are commonly extractable from ClinicalTrials.gov result tables in this prototype. Extension to other endpoints is treated as a pre-submission enhancement and future validation target.

### 2.3 Stage 1 retrieval and SECRET pool construction

Stage 1 retrieval identifies high-recall candidate historical trials. The implemented backends include hashing, ClinicalBERT-style embeddings, Trial2Vec-style retrieval, and SECRET-style section retrieval. The SECRET-style path uses fixed section-level representations and weights to emphasize borrowing-relevant trial sections rather than title similarity alone. Candidate pools produced by Stage 1 are then available for Stage 2 reranking and endpoint observation extraction.

The paired Stage 1 benchmark compares retrieval outputs on common query IDs using the same endpoint key and candidate budget. This paired design avoids comparing backend results across different query sets and supports query-level bootstrap confidence intervals for metric differences.

### 2.4 Explainable Stage 2 reranking

Stage 2 reranking scores candidate pairs using borrowing-relevant features. The current feature schema includes overall similarity, disease match, regimen match, endpoint match, follow-up match, eligibility match, result quality, negative red flag severity, and information size. These features are intended to make the basis for candidate prioritization auditable. They are not expert labels and are not interpreted as final clinical borrowing decisions.

### 2.5 Endpoint observations and beta-binomial components

For candidate `i`, the pipeline extracts an endpoint observation when a compatible result table is available:

```text
y_i = candidate endpoint response or event count
n_i = candidate endpoint denominator
```

Candidates without compatible endpoint observations can still be retrieved and reranked, but they do not contribute a historical beta-binomial component. When an endpoint observation is usable, the candidate contributes a beta component:

```text
alpha_i = 1 + a_i y_i
beta_i  = 1 + a_i (n_i - y_i)
```

Here `a_i` is an effective sample-size discount. It controls the amount of information borrowed inside the candidate beta component.

### 2.6 Mixture prior, mixture weights, and effective sample-size discounts

The prior for the response probability is represented as a weak component plus historical components:

```text
p(theta) = lambda_0 Beta(theta | 1, 1)
         + sum_i lambda_i Beta(theta | alpha_i, beta_i)
```

The current prototype uses `lambda_0 = 0.2` by default unless otherwise specified. The component weight `lambda_i` and the discount `a_i` are conceptually distinct. The former controls allocation across mixture components, whereas the latter controls the effective sample size inside a component. The two-head DeepSets model is designed to preserve this distinction by learning mixture allocation and sample-size discounting as separate outputs.

### 2.7 SAM-style prior-data conflict adaptation

An optional SAM-style conflict adapter is evaluated as a sensitivity analysis. The adapter downweights historical components when the pseudo-query outcome conflicts with the prior predictive distribution. In this manuscript, SAM adaptation is interpreted as a transparent conflict-sensitivity mechanism, not as a regulatory qualification procedure.

### 2.8 Validation without expert borrowability labels

Expert borrowability labels were unavailable. Validation therefore used internal retrospective and simulation evidence:

1. held-out beta-binomial predictive NLL;
2. true-date temporal subset NLL;
3. simulation operating characteristics;
4. paired Stage 1 backend benchmarking;
5. borrowing baseline head-to-head comparisons;
6. feature ablation and section-weight sensitivity.

All results are interpreted as retrospective predictive calibration or simulation evidence. They do not establish expert-level borrowability of individual candidates.

## 3. Results

### 3.1 True date metadata supported temporal evaluation

The date extraction module produced true ClinicalTrials.gov temporal metadata for 7,173 oncology trials. Primary completion date was available for all 7,173 trials in the local data extract; 6,631 had day-level precision and 542 had month-level precision. Completion date was missing for 9 trials, while results first posted date and start date were available for all records in the current extract. This coverage enabled temporal analyses based on primary completion dates rather than NCT numeric proxy ordering.

### 3.2 SECRET pool improved paired Stage 1 retrieval metrics

The paired Stage 1 benchmark compared hashing and SECRET pool retrieval on 1,470 common ORR pseudo-queries using the same candidate budget. SECRET pool increased reranked component-readiness from 0.5450 to 0.6361. The paired delta was 0.0912 with a bootstrap confidence interval of 0.0799 to 0.1018. SECRET pool also increased reranked endpoint-match score from 2.2281 to 3.3327, with a paired delta of 1.1047 and confidence interval 1.0600 to 1.1503. Endpoint-and-result readiness at the topK level was unchanged in this comparison.

### 3.3 Borrowing baseline comparisons showed improved predictive calibration for calibrated and conflict-adapted priors

The borrowing baseline comparison used 1,414 ORR pseudo-query examples. The weak-only prior had mean NLL 3.1877. The rule mixture baseline had mean NLL 3.1620. The trained two-head mixture model achieved mean NLL 3.0181, corresponding to a delta of -0.1439 relative to the rule baseline. The conflict-adapted rule baseline achieved mean NLL 2.9721, corresponding to a delta of -0.1900. MAP-like and power-prior-like baselines had higher NLLs in the current evaluation.

These results suggest that calibrated mixture allocation and explicit prior-data conflict adaptation can improve retrospective predictive performance. They do not show that individual historical candidates are expert-approved for borrowing.

### 3.4 True-date temporal subset validation supported the calibration signal

True-date temporal subset validation summarized predictive NLL across date-based and rolling-origin subsets using primary completion date metadata. In date-based subsets, the trained two-head model improved mean NLL relative to the rule baseline across cutoffs from 2019-12-31 through 2022-12-31. The corresponding deltas ranged from -0.1459 to -0.1556. The rule-SAM baseline also improved NLL across the same cutoffs, with deltas ranging from -0.1736 to -0.1807.

Rolling-origin subsets showed similar patterns. The trained two-head model improved over the rule baseline in all four windows, with deltas from -0.1263 to -0.1741. The rule-SAM baseline improved over the rule baseline with deltas from -0.1661 to -0.2051. These analyses are true-date temporal subset summaries, not full temporal retraining experiments. Full retraining with date-based and rolling-origin splits is listed as a pre-submission enhancement because the current local environment did not include `torch`.

### 3.5 Simulation operating characteristics quantified exchangeability and conflict behavior

Simulation operating characteristics were estimated under six scenarios: exchangeable historical evidence, mild optimistic conflict, strong optimistic conflict, mild pessimistic conflict, mixture historical conflict, and heterogeneous historical evidence. The current evidence package used 500 Monte Carlo iterations and 400 deterministic template examples for computational tractability.

Under exchangeability, weak-only, rule, rule-SAM, and fixed-discount methods showed low empirical type I error in the current decision setup, with coverage ranging from 0.9424 to 0.9693. Under strong optimistic conflict, rule and fixed-discount methods showed higher power but also increased bias and reduced coverage. The rule-SAM method showed a SAM trigger rate of 0.4545 in strong optimistic conflict, indicating that the conflict adapter responded more often under stronger prior-data disagreement. These results provide stress-test evidence for borrowing behavior under simplified data-generating mechanisms.

### 3.6 Feature ablation provided a transparent sensitivity check

Feature ablation used a deterministic feature-weight proxy to examine the nine-feature borrowability schema. Dropping endpoint, follow-up, or result-quality features increased mean NLL modestly relative to the full proxy. Dropping information size reduced mean NLL in this proxy analysis, indicating that the current hand-weighted proxy does not fully represent the learned model behavior. This ablation should be interpreted as a transparency and sensitivity analysis, not as a full retraining ablation of the two-head model.

## 4. Discussion

This study presents an explainable methodology prototype for moving from oncology trial similarity to Bayesian historical borrowability. The central contribution is not a new text retrieval model alone, but a pipeline that links retrieval, candidate-level borrowability features, endpoint observation extraction, beta-binomial mixture components, retrospective calibration, prior-data conflict adaptation, and simulation operating-characteristics evaluation. This design directly addresses a common gap in historical borrowing workflows: retrieved historical trials are often treated as similar before the statistical consequences of borrowing are assessed.

The results support three practical observations. First, retrieval backend choice matters for downstream borrowing readiness. SECRET pool retrieval improved component-readiness and endpoint-match metrics under a paired benchmark. Second, mixture-prior calibration and conflict adaptation can improve held-out predictive NLL compared with weak-only or rule-based borrowing. Third, true-date temporal subset analyses provide a more credible retrospective validation structure than NCT numeric proxy ordering, although they do not replace full temporal retraining.

The simulation study complements retrospective calibration by examining operating characteristics under controlled exchangeability and conflict scenarios. Optimistic historical conflict increased apparent power for borrowing methods but also increased bias and reduced coverage. The SAM-style adapter showed higher trigger rates under stronger conflict, supporting its role as a transparent sensitivity mechanism. These findings are consistent with the broader principle that historical borrowing should be evaluated not only for predictive fit but also for conflict behavior and decision operating characteristics. [Citation needed: prior-data conflict and robust borrowing.]

### 4.1 Limitations

This study has several important limitations. First, no expert borrowability labels were available. Consequently, the evaluation cannot establish that any individual retrieved historical trial is clinically appropriate for borrowing in a real oncology trial. The results should be interpreted as retrospective predictive calibration, simulation operating-characteristics evidence, and internal validation rather than expert adjudication or regulatory qualification. Second, the current true-date temporal analyses summarize predictive performance over temporal subsets using existing learned NLL rows; full date-based and rolling-origin temporal retraining should be completed in a `torch`-enabled environment before stronger claims about forward validation are made. Third, automated extraction from ClinicalTrials.gov may introduce errors in endpoint mapping, arm identification, eligibility interpretation, date parsing, and result usability assessment. Fourth, the current empirical analyses focus primarily on ORR pseudo-queries, and generalization to progression-free survival, disease control rate, complete response, partial response, or time-to-event endpoints remains to be demonstrated. Finally, simulation scenarios simplify the complexities of oncology trial design and should be viewed as stress tests of borrowing behavior rather than substitutes for protocol-specific operating-characteristics studies.

### 4.2 Regulatory and practical implications

Historical borrowing for decision-informing trials requires pre-specification of evidence selection, prior construction, conflict assessment, and operating-characteristics evaluation. The present framework supports several of these elements retrospectively, including transparent candidate features, explicit beta-binomial components, mixture weights, effective sample-size discounts, and SAM-style conflict adaptation. However, any use in a real trial would require disease-specific expert review, endpoint-specific validation, protocol-level pre-specification, and formal operating-characteristics evaluation tailored to the intended decision. The repository should therefore be used as a reproducible methodology prototype and calibration evidence package, not as a clinical borrowing recommendation system.

### 4.3 Pre-submission enhancements

Two extensions would materially strengthen a submission. First, full temporal retraining should be run for date-based and rolling-origin splits in a `torch`-enabled environment, so that future pseudo-queries are evaluated by models trained only on earlier pseudo-queries. Second, the validation should be extended beyond ORR to at least one additional endpoint family, such as disease control rate, complete or partial response, or a clinically meaningful time-to-event proxy. A full retraining ablation of the two-head model would also improve the mechanistic interpretation of the nine-feature schema.

## 5. Conclusion

We developed a reproducible methodology prototype for oncology trial retrieval and Bayesian historical borrowing that distinguishes retrievability from borrowability. The current evidence package combines paired retrieval benchmarking, held-out predictive NLL, true-date temporal subset summaries, simulation operating characteristics, baseline comparisons, and feature ablation without expert borrowability labels. The framework supports transparent method development and retrospective calibration, but individual borrowing decisions remain dependent on expert clinical and statistical review.

## Suggested Supplementary Material

1. Full simulation scenario definitions and operating-characteristics tables.
2. True date extraction precision and missingness report.
3. Full paired Stage 1 query-level metrics and bootstrap procedure.
4. Full borrowing baseline NLL rows.
5. Feature ablation and SECRET section-weight sensitivity results.
6. Leakage-control schema and pseudo-query construction details.
7. Reproducibility commands for all no-expert-label validation workflows.
8. Torch-enabled full temporal retraining command templates.
9. Data availability and raw ClinicalTrials.gov extraction limitations.

## Claim-Evidence Map

| Claim | Evidence | Status |
|---|---|---|
| SECRET pool improves borrowability-relevant Stage 1 retrieval metrics over hashing. | Paired benchmark over 1,470 common ORR pseudo-queries with bootstrap CIs. | Supported for current ORR setup. |
| Trained two-head mixture calibration improves held-out predictive NLL over rule baseline. | Baseline table: 3.0181 vs 3.1620 mean NLL. | Supported for current retrospective ORR examples. |
| SAM-style conflict adaptation improves retrospective NLL and responds to conflict. | Baseline NLL and simulation SAM trigger rates. | Supported as sensitivity evidence. |
| True-date temporal ordering strengthens retrospective validation. | Primary completion date coverage and temporal subset NLL tables. | Supported for subset calibration; full retraining pending. |
| The framework identifies clinically borrowable historical evidence. | No expert labels. | Not supported; explicitly out of scope. |

## Assumptions and Missing Inputs

- Add verified citations for Bayesian historical borrowing, power priors, robust mixture priors, MAP priors, commensurate priors, SAM or prior-data conflict methods, Trial2Vec, SECRET, and regulatory guidance.
- Complete full temporal retraining in a `torch` environment before making forward-validation claims.
- Add multi-endpoint validation before claiming endpoint generality.
- Decide whether the submission target uses a structured abstract and adjust formatting accordingly.
