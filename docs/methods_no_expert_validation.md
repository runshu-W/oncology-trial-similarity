# Methods Without Expert Borrowability Labels

This document summarizes the current methodology package for a Pharmaceutical Statistics or Statistics in Medicine style manuscript. The project is framed as a retrospective predictive calibration and simulation study, not as a clinically validated borrowing recommendation system.

## Study Objective

The objective is to identify historical oncology trials that can form candidate evidence for Bayesian mixture-prior borrowing. The pipeline is designed to distinguish text retrieval from borrowability. A candidate is useful only if it is close enough in disease, population, regimen, endpoint, follow-up, eligibility, result usability, information size, and red flags to support borrowing-sensitive prior construction.

## Pseudo-Query Design and Leakage Control

Completed ClinicalTrials.gov oncology trials are treated retrospectively as pseudo-queries. For each pseudo-query, the query outcome is hidden from retrieval and candidate selection. Candidate retrieval and reranking use structured trial fields and historical candidate records. The held-out pseudo-query endpoint observation is used only for retrospective evaluation of predictive performance.

Pipeline results intended for lambda-model training require leakage-control metadata. Rows without hidden query outcomes and explicit held-out endpoint observations are rejected by the retrospective training/evaluation code path.

## Structured Summary Extraction

Each ClinicalTrials.gov JSON record is converted into a structured oncology trial summary. The summary includes:

- disease and tumor context;
- patient population and treatment line when available;
- intervention and regimen descriptors;
- primary and secondary endpoint descriptors;
- follow-up and time-frame text;
- eligibility criteria;
- result availability and extractable endpoint observations;
- red flags that can reduce borrowing suitability;
- true date metadata when available.

The extraction is automated and therefore may contain errors. This is a manuscript limitation and a reproducibility consideration.

## Stage 1 Retrieval and SECRET Pool

Stage 1 retrieves high-recall historical candidates. Available backends include hashing, ClinicalBERT, Trial2Vec-style embeddings, and SECRET-style section retrieval. The SECRET-style path uses fixed section-level representations and weights to form a candidate pool that is more aligned with borrowing-relevant trial sections than title-level similarity alone.

Paired Stage 1 benchmarking compares backends on common query IDs with the same endpoint key and candidate budget. The current benchmark compares hashing and SECRET pool outputs over ORR pseudo-queries.

## Stage 2 Explainable Reranking

Stage 2 reranking assigns candidate-level borrowing features. The current feature schema includes overall similarity, disease match, regimen match, endpoint match, follow-up match, eligibility match, result quality, negative red flag severity, and information size.

The goal of this stage is not to produce a final clinical decision. It creates a transparent feature vector for downstream mixture-prior construction and retrospective calibration.

## Endpoint Observation Extraction

For each candidate with usable outcome results, the pipeline extracts an endpoint observation:

```text
y_i = candidate endpoint response or event count
n_i = candidate endpoint denominator
```

Candidates without compatible endpoint results may still be retrieved and reranked, but they cannot contribute a beta-binomial historical component.

## Beta Component Construction

For candidate `i`, the historical beta component is constructed as:

```text
alpha_i = 1 + a_i y_i
beta_i  = 1 + a_i (n_i - y_i)
```

Here `a_i` is the effective sample-size discount. It is distinct from `lambda_i`, the mixture weight assigned to candidate component `i`.

The mixture prior combines a weak component and historical components:

```text
p(theta) = lambda_0 Beta(theta | 1, 1)
         + sum_i lambda_i Beta(theta | alpha_i, beta_i)
```

The current prototype uses `lambda_0 = 0.2` by default unless otherwise specified.

## Two-Head DeepSets Lambda Model

The two-head DeepSets model is designed to separate two borrowing roles:

- `lambda_i`: mixture allocation across historical components;
- `a_i`: within-component sample-size discount.

This separation is important because a trial can receive a low mixture weight because it is not central to the candidate set, or receive a low discount because its endpoint evidence is uncertain or partially transportable. The two quantities should not be collapsed into a single score.

## SAM Conflict Adapter

The optional SAM conflict adapter downweights prior components when observed pseudo-query data conflict with the prior predictive distribution. In this project it is used as a sensitivity analysis and a transparent conflict-adaptation baseline.

## True Date Metadata

The date extraction module reads ClinicalTrials.gov date fields from both newer API-style records and local legacy exported JSON structures. It extracts:

- primary completion date;
- completion date;
- results first posted date;
- start date.

Dates retain precision labels: day, month, year, missing, or unparseable. Month-only dates are mapped to deterministic mid-month anchors for temporal sorting while retaining their precision label. The temporal validation utilities prefer primary completion date, then completion date, results first posted date, start date, and finally NCT numeric proxy only when date metadata are unavailable.

## Validation Without Expert Labels

Because expert borrowability labels are unavailable, validation uses internal evidence:

- held-out beta-binomial predictive NLL;
- true-date temporal NLL summaries;
- simulation operating characteristics;
- paired Stage 1 backend benchmarks;
- traditional borrowing baseline comparisons;
- feature ablation and sensitivity analysis.

These evaluations support method development and calibration assessment. They do not establish clinical correctness of individual borrowing recommendations.
