# Random 5 ClinicalBERT Top3 Similarity Test

## Test Setup

- Random seed: `20260515`
- Sampled query trials: `NCT03180307, NCT02058706, NCT03056339, NCT04434937, NCT05287113`
- Index: `artifacts/oncology_trial_similarity_clinicalbert`
- Embedding backend: `clinicalbert`
- Embedding model: `emilyalsentzer/Bio_ClinicalBERT`
- Retrieval: top100 first-stage multi-aspect ClinicalBERT cosine search
- Rerank: deterministic prior-borrowing rerank over top100
- Reported below: reranked top3 for each query

## Pipeline Interpretation

For each query, the pipeline first converts the trial JSON into the same structured summary schema used by the historical index. It then embeds five aspects separately: disease/population, intervention, endpoint, design, and results/safety. The first-stage retrieval uses weighted cosine similarity across these five ClinicalBERT aspect embeddings. The second-stage reranker then reviews the top100 candidates using structured clinical/statistical rules for disease match, regimen match, endpoint/estimand match, design match, result usability, and safety/follow-up relevance. The final top3 below are therefore not simply nearest text neighbors; they are candidates that survived a prior-borrowing-oriented rerank.

## Query NCT03180307

- Title: OTL38 for Intra-operative Imaging of Folate Receptor Positive Ovarian Cancer
- Phase: Phase 3
- Cancer summary: `{'primary_site': ['Ovary'], 'histology': ['Ovarian cancer'], 'molecular_marker': ['Not reported'], 'stage_or_risk': ['Not reported'], 'line_of_therapy': 'Not reported', 'prior_treatment': 'Not reported'}`
- Intervention summary: `{'experimental_regimen': 'Drug: OTL38; Device: near infrared camera imaging system; Procedure: laparotomy', 'control_or_comparator': 'Not reported', 'drug_classes': [], 'backbone_regimen': ['Not normalized'], 'dose_schedule_summary': 'See source record', 'treatment_duration': 'Not reported'}`

| Rank | Candidate | Score | Retrieval | Suitability | Discount | Main rationale | Red flags |
|---:|---|---:|---:|---|---:|---|---|
| 1 | NCT03951077 - Study of the Safety and Efficacy of Elagolix in Women With Polycystic  | 69.33 | 98.5 | medium | 0.4 | Shared histology: ovarian cancer. Shared primary site/category: ovary. Shared primary endpoint families: other. Same arm structure: Multi-arm. Same randomization status: Yes. | Line of therapy was not normalized from available text.; Low treatment-regimen match. |
| 2 | NCT03382574 - Pilot Study of Denosumab in BRCA1/2 Mutation Carriers Scheduled for Ri | 65.16 | 98.65 | medium | 0.4 | Shared histology: ovarian cancer. Shared primary site/category: ovary. Shared primary endpoint families: other. Same arm structure: Multi-arm. Same randomization status: Yes. | Line of therapy was not normalized from available text.; Low treatment-regimen match.; No arm-level count/denominator pair found for primary borrowable quantities. |
| 3 | NCT04598321 - BrUOG 390: Neoadjuvant Treatment With Talazoparib | 63.61 | 98.45 | medium | 0.4 | Shared histology: ovarian cancer. Shared primary site/category: ovary. Shared primary endpoint families: other. | Low treatment-regimen match. |

### Top3 Dimension Scores

- `NCT03951077`: `{'disease_population_match': 3.5, 'treatment_regimen_match': 0.0, 'endpoint_estimand_match': 5.0, 'design_phase_match': 4.3, 'result_usability': 5.0, 'safety_and_followup_relevance': 0.0}`
- `NCT03382574`: `{'disease_population_match': 3.5, 'treatment_regimen_match': 0.0, 'endpoint_estimand_match': 5.0, 'design_phase_match': 3.5, 'result_usability': 3.0, 'safety_and_followup_relevance': 0.0}`
- `NCT04598321`: `{'disease_population_match': 3.5, 'treatment_regimen_match': 0.0, 'endpoint_estimand_match': 5.0, 'design_phase_match': 0.5, 'result_usability': 5.0, 'safety_and_followup_relevance': 0.0}`

## Query NCT02058706

- Title: LHRH Analogue Therapy With Enzalutamide or Bicalutamide in Treating Patients With Hormone Sensitive Prostate Cancer
- Phase: Phase 2
- Cancer summary: `{'primary_site': ['Prostate'], 'histology': ['Prostate cancer'], 'molecular_marker': ['Not reported'], 'stage_or_risk': ['Not reported'], 'line_of_therapy': 'Not reported', 'prior_treatment': 'Not reported'}`
- Intervention summary: `{'experimental_regimen': 'Drug: enzalutamide; Drug: bicalutamide; Procedure: orchiectomy; Drug: leuprolide acetate; Drug: goserelin acetate; Other: laboratory biomarker analysis', 'control_or_comparator': 'Not reported', 'drug_classes': [], 'backbone_regimen': ['Not normalized'], 'dose_schedule_summary': 'See source record', 'treatment_duration': 'Not reported'}`

| Rank | Candidate | Score | Retrieval | Suitability | Discount | Main rationale | Red flags |
|---:|---|---:|---:|---|---:|---|---|
| 1 | NCT03279250 - Apalutamide and Gonadotropin-Releasing Hormone Analog With or Without  | 70.54 | 99.17 | medium | 0.4 | Shared histology: prostate cancer. Shared primary site/category: prostate. Shared primary endpoint families: other. Same arm structure: Multi-arm. Same randomization status: Yes. | Line of therapy was not normalized from available text.; Low treatment-regimen match. |
| 2 | NCT02059213 - A Phase II Study of Androgen Deprivation Therapy With or Without Palbo | 70.52 | 99.08 | medium | 0.4 | Shared histology: prostate cancer. Shared primary site/category: prostate. Shared primary endpoint families: other. Same arm structure: Multi-arm. Same randomization status: Yes. | Line of therapy was not normalized from available text.; Low treatment-regimen match. |
| 3 | NCT01946165 - Abiraterone Acetate Plus LHRH Agonist and Abiraterone Acetate Plus LHR | 70.52 | 99.07 | medium | 0.4 | Shared histology: prostate cancer. Shared primary site/category: prostate. Shared primary endpoint families: other. Same arm structure: Multi-arm. Same randomization status: Yes. | Line of therapy was not normalized from available text.; Low treatment-regimen match. |

### Top3 Dimension Scores

- `NCT03279250`: `{'disease_population_match': 3.5, 'treatment_regimen_match': 0.0, 'endpoint_estimand_match': 5.0, 'design_phase_match': 5.0, 'result_usability': 5.0, 'safety_and_followup_relevance': 0.0}`
- `NCT02059213`: `{'disease_population_match': 3.5, 'treatment_regimen_match': 0.0, 'endpoint_estimand_match': 5.0, 'design_phase_match': 5.0, 'result_usability': 5.0, 'safety_and_followup_relevance': 0.0}`
- `NCT01946165`: `{'disease_population_match': 3.5, 'treatment_regimen_match': 0.0, 'endpoint_estimand_match': 5.0, 'design_phase_match': 5.0, 'result_usability': 5.0, 'safety_and_followup_relevance': 0.0}`

## Query NCT03056339

- Title: Umbilical & Cord Blood (CB) Derived CAR-Engineered NK Cells for B Lymphoid Malignancies
- Phase: Phase 1/Phase 2
- Cancer summary: `{'primary_site': ['Hematologic malignancy'], 'histology': ['Not normalized'], 'molecular_marker': ['Not reported'], 'stage_or_risk': ['Not reported'], 'line_of_therapy': 'Relapsed/refractory or previously treated', 'prior_treatment': 'Not reported'}`
- Intervention summary: `{'experimental_regimen': 'Drug: Fludarabine; Drug: Cyclophosphamide; Drug: Mesna; Biological: iC9/CAR.19/IL15-Transduced CB-NK Cells; Drug: AP1903', 'control_or_comparator': 'Not reported', 'drug_classes': ['Chemotherapy'], 'backbone_regimen': ['Not normalized'], 'dose_schedule_summary': 'See source record', 'treatment_duration': 'Not reported'}`

| Rank | Candidate | Score | Retrieval | Suitability | Discount | Main rationale | Red flags |
|---:|---|---:|---:|---|---:|---|---|
| 1 | NCT04432506 - Anakinra for the Reduction of CAR-T Toxicity in Patients With Relapsed | 68.23 | 98.62 | medium | 0.4 | Shared primary site/category: hematologic malignancy. Shared drug classes: chemotherapy. Shared primary endpoint families: other. Same arm structure: Single-arm. Same randomization status: No. | Cancer histology was not normalized from available text. |
| 2 | NCT03323034 - Pevonedistat, Irinotecan, and Temozolomide in Treating Patients With R | 68.19 | 98.44 | medium | 0.4 | Shared primary site/category: hematologic malignancy. Shared drug classes: chemotherapy. Shared primary endpoint families: other. Same arm structure: Single-arm. Same randomization status: No. | Cancer histology was not normalized from available text. |
| 3 | NCT02227199 - Brentuximab Vedotin, Ifosfamide, Carboplatin, and Etoposide in Treatin | 66.98 | 98.43 | medium | 0.4 | Shared primary site/category: hematologic malignancy. Shared drug classes: chemotherapy. Shared primary endpoint families: other. Same randomization status: No. |  |

### Top3 Dimension Scores

- `NCT04432506`: `{'disease_population_match': 2.0, 'treatment_regimen_match': 1.5, 'endpoint_estimand_match': 5.0, 'design_phase_match': 4.3, 'result_usability': 5.0, 'safety_and_followup_relevance': 0.0}`
- `NCT03323034`: `{'disease_population_match': 2.0, 'treatment_regimen_match': 1.5, 'endpoint_estimand_match': 5.0, 'design_phase_match': 4.3, 'result_usability': 5.0, 'safety_and_followup_relevance': 0.0}`
- `NCT02227199`: `{'disease_population_match': 2.0, 'treatment_regimen_match': 1.5, 'endpoint_estimand_match': 5.0, 'design_phase_match': 3.5, 'result_usability': 5.0, 'safety_and_followup_relevance': 0.0}`

## Query NCT04434937

- Title: Open-Label Study of Parsaclisib, in Japanese Participants With Relapsed or Refractory Follicular Lymphoma (CITADEL-213)
- Phase: Phase 2
- Cancer summary: `{'primary_site': ['Hematologic malignancy'], 'histology': ['Not normalized'], 'molecular_marker': ['Not reported'], 'stage_or_risk': ['Not reported'], 'line_of_therapy': 'Relapsed/refractory or previously treated', 'prior_treatment': 'Not reported'}`
- Intervention summary: `{'experimental_regimen': 'Drug: parsaclisib', 'control_or_comparator': 'Not reported', 'drug_classes': [], 'backbone_regimen': ['Not normalized'], 'dose_schedule_summary': 'See source record', 'treatment_duration': 'Not reported'}`

| Rank | Candidate | Score | Retrieval | Suitability | Discount | Main rationale | Red flags |
|---:|---|---:|---:|---|---:|---|---|
| 1 | NCT02953652 - Efficacy and Safety of Oral HBI-8000 in Patients With Relapsed or Refr | 63.9 | 99.59 | medium | 0.4 | Shared primary site/category: hematologic malignancy. Shared primary endpoint families: orr/cr/pr. Same arm structure: Single-arm. Same randomization status: No. | Cancer histology was not normalized from available text.; Low treatment-regimen match. |
| 2 | NCT02927925 - A Study to Assess the Clinical Efficacy and Safety of Daratumumab in P | 63.83 | 99.32 | medium | 0.4 | Shared primary site/category: hematologic malignancy. Shared primary endpoint families: orr/cr/pr. Same arm structure: Single-arm. Same randomization status: No. | Cancer histology was not normalized from available text.; Low treatment-regimen match. |
| 3 | NCT02038946 - Study of Nivolumab in Subjects With Relapsed or Refractory Follicular  | 63.8 | 99.21 | medium | 0.4 | Shared primary site/category: hematologic malignancy. Shared primary endpoint families: orr/cr/pr. Same arm structure: Single-arm. Same randomization status: No. | Cancer histology was not normalized from available text.; Low treatment-regimen match. |

### Top3 Dimension Scores

- `NCT02953652`: `{'disease_population_match': 2.0, 'treatment_regimen_match': 0.0, 'endpoint_estimand_match': 5.0, 'design_phase_match': 5.0, 'result_usability': 5.0, 'safety_and_followup_relevance': 0.0}`
- `NCT02927925`: `{'disease_population_match': 2.0, 'treatment_regimen_match': 0.0, 'endpoint_estimand_match': 5.0, 'design_phase_match': 5.0, 'result_usability': 5.0, 'safety_and_followup_relevance': 0.0}`
- `NCT02038946`: `{'disease_population_match': 2.0, 'treatment_regimen_match': 0.0, 'endpoint_estimand_match': 5.0, 'design_phase_match': 5.0, 'result_usability': 5.0, 'safety_and_followup_relevance': 0.0}`

## Query NCT05287113

- Title: Study of Retinfanlimab in Combination With INCAGN02385 and INCAGN02390 as First-Line Treatment in Participants With PD-L1-Positive (CPS ≥ 1) Recurrent/Metastatic Squamous Cell Carcinoma of the Head and Neck
- Phase: Phase 2
- Cancer summary: `{'primary_site': ['Not normalized'], 'histology': ['Not normalized'], 'molecular_marker': ['PD-L1'], 'stage_or_risk': ['Not reported'], 'line_of_therapy': 'Frontline / previously untreated', 'prior_treatment': 'Not reported'}`
- Intervention summary: `{'experimental_regimen': 'Drug: Retifanlimab; Drug: INCAGN02385; Drug: INCAGN02390; Drug: Placebo', 'control_or_comparator': 'Not reported', 'drug_classes': [], 'backbone_regimen': ['Not normalized'], 'dose_schedule_summary': 'See source record', 'treatment_duration': 'Not reported'}`

| Rank | Candidate | Score | Retrieval | Suitability | Discount | Main rationale | Red flags |
|---:|---|---:|---:|---|---:|---|---|
| 1 | NCT02432846 - Intratumoral Vaccination With Intuvax Pre-nephrectomy Followed by Suni | 58.07 | 98.78 | low | 0.15 | Shared primary endpoint families: os. Same arm structure: Multi-arm. Same randomization status: Yes. | Cancer histology was not normalized from available text.; Low disease/population match.; Low treatment-regimen match. |
| 2 | NCT05079230 - Study of Magrolimab Versus Placebo in Combination With Venetoclax and  | 57.05 | 98.88 | low | 0.15 | Shared primary endpoint families: os. Same arm structure: Multi-arm. Same randomization status: Yes. | Low disease/population match.; Low treatment-regimen match. |
| 3 | NCT02369874 - Study of MEDI4736 Monotherapy and in Combination With Tremelimumab Ver | 57.04 | 98.87 | low | 0.15 | Shared marker: pd-l1. Shared primary endpoint families: os. Same arm structure: Multi-arm. Same randomization status: Yes. | Cancer histology was not normalized from available text.; Line of therapy was not normalized from available text.; Low disease/population match.; Low treatment-regimen match. |

### Top3 Dimension Scores

- `NCT02432846`: `{'disease_population_match': 0.75, 'treatment_regimen_match': 0.0, 'endpoint_estimand_match': 5.0, 'design_phase_match': 5.0, 'result_usability': 5.0, 'safety_and_followup_relevance': 0.0}`
- `NCT05079230`: `{'disease_population_match': 0.75, 'treatment_regimen_match': 0.0, 'endpoint_estimand_match': 5.0, 'design_phase_match': 4.3, 'result_usability': 5.0, 'safety_and_followup_relevance': 0.0}`
- `NCT02369874`: `{'disease_population_match': 0.75, 'treatment_regimen_match': 0.0, 'endpoint_estimand_match': 5.0, 'design_phase_match': 4.3, 'result_usability': 5.0, 'safety_and_followup_relevance': 0.0}`

## How to Read These Results

- `retrieval_score` is the first-stage ClinicalBERT multi-aspect similarity score. It is useful for recall but not sufficient for borrowing decisions.
- `score` is the stage-2 overall prior-borrowing score, combining structured rerank dimensions with retrieval score.
- `suitability` is a practical category: high, medium, low, or do_not_borrow.
- `discount` is an initial borrowing-weight suggestion for sensitivity analysis, not a final statistical conclusion.
- `red_flags` should be reviewed before including any historical trial in a Bayesian prior.

## Important Caveats

- The first-stage retrieval now uses Bio_ClinicalBERT, but Bio_ClinicalBERT is not specifically trained for trial-to-trial similarity; it is a biomedical language model used here with mean-pooling.
- The second-stage reranker is deterministic and rule-based. It is more interpretable than pure embedding search, but it is not a substitute for expert clinical/statistical review.
- Protocol/SAP text is still not fully parsed in this environment, so eligibility, estimand, censoring, and analysis-population details remain incomplete.
- For binary endpoints such as ORR, extracted count/denominator can inform beta-binomial priors. For PFS/OS, additional survival information is usually needed.
