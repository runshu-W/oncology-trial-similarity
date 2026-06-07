# Manuscript Positioning

Target journals: Pharmaceutical Statistics and Statistics in Medicine.

## Recommended Framing

This manuscript should be framed as a statistical methodology and retrospective calibration study for oncology historical borrowing. It should not be framed as a clinical decision support system or an expert-validated trial matching tool.

Recommended one-sentence argument:

> We develop and retrospectively evaluate an explainable oncology trial retrieval and Bayesian mixture-prior borrowing framework that separates trial retrievability from borrowability and quantifies predictive calibration without relying on expert borrowability labels.

## Title Candidates

1. Explainable oncology trial retrieval for Bayesian historical borrowing without expert borrowability labels
2. Retrospective predictive calibration of trial similarity for Bayesian historical borrowing in oncology
3. From trial similarity to borrowability: a mixture-prior framework for oncology historical evidence
4. A true-date retrospective validation framework for oncology trial retrieval and Bayesian historical borrowing
5. Simulation and temporal calibration of automated historical-borrowing evidence selection in oncology

## Abstract Framework

### Background

Bayesian historical borrowing can improve efficiency in oncology trials, but selecting historical evidence requires more than text similarity. Historical trials must be comparable in disease, population, regimen, endpoint, follow-up, eligibility, result usability, and conflict risk.

### Methods

Describe the two-stage pipeline: structured ClinicalTrials.gov extraction, Stage 1 retrieval, SECRET-style candidate pooling, explainable Stage 2 reranking, endpoint observation extraction, beta component construction, two-head DeepSets mixture-prior calibration, and optional SAM prior-data conflict adaptation. Emphasize leakage-controlled retrospective pseudo-queries.

### Results

Report paired Stage 1 benchmark results, baseline head-to-head NLL, true-date temporal NLL, simulation operating characteristics, and feature ablation. State that results use ORR pseudo-queries and no expert borrowability labels.

### Conclusions

The framework provides a reproducible retrospective calibration package for historical borrowing candidate selection. It supports method development and sensitivity analysis, but individual borrowing decisions require clinical and statistical review.

## Core Limitation Paragraph

This study does not include expert borrowability labels. Consequently, the evaluation cannot establish that any retrieved historical trial is clinically appropriate for borrowing in a real oncology trial. Instead, the evidence is based on retrospective predictive calibration, true-date temporal summaries, simulation operating characteristics, paired retrieval benchmarks, and feature ablation. Automated ClinicalTrials.gov extraction may introduce errors in endpoint mapping, arm identification, eligibility interpretation, date parsing, and result usability. The current results focus primarily on ORR pseudo-queries and should not be interpreted as prospective validation, regulatory qualification, or a substitute for pre-specified clinical and statistical review of historical evidence.

## Regulatory Caution Paragraph

Historical borrowing for confirmatory or decision-informing trial analyses requires pre-specification of the borrowing model, transparent external evidence selection, prior-data conflict assessment, operating-characteristics evaluation, and sensitivity analysis. The present framework addresses several of these elements retrospectively, including candidate transparency, mixture-prior construction, prior-data conflict adaptation, and simulation operating characteristics. However, it has not been qualified by regulators, has not been prospectively deployed, and has not been adjudicated by clinical or statistical experts. Any use in a real trial design would require protocol-level pre-specification, disease-specific expert review, endpoint-specific validation, and formal operating-characteristics assessment tailored to the intended decision.

## Recommended Submission Strategy

Submit first as a preprint and then as a full methods article to Pharmaceutical Statistics or Statistics in Medicine after manuscript-ready tables and figures are assembled. A short communication is possible, but the breadth of the pipeline, simulation study, temporal validation, and baseline comparisons is better suited to a full methods article. Clinical Cancer Research would require stronger oncology case studies and expert review; Nature Methods would require broader methodological novelty, external validation, and larger-scale endpoint generalization.
