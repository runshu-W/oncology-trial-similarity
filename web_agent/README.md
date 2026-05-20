# Web Agent

The web agent is a local FastAPI interface for the oncology trial similarity pipeline. It accepts one ClinicalTrials.gov-style trial JSON by upload or paste and runs the existing ClinicalBERT retrieval plus prior-borrowing rerank flow.

## Run Locally

From the `oncology-trial-similarity` directory:

```bash
../.venv/bin/python -m pip install -r requirements.txt
../.venv/bin/uvicorn web_agent.app:app --reload --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## Defaults

- Index: `../artifacts/oncology_trial_similarity_clinicalbert`
- `top_k`: `10`
- `rerank_top_n`: `100`
- Input: uploaded or pasted JSON object

## Output

The page displays:

- Query trial title and phase.
- Top reranked historical trials.
- Overall score, retrieval score, prior-borrowing suitability, and suggested discount.
- Red flags, dimension scores, and borrowable quantities.
- Bayesian prior-borrowing analysis when ORR or PFS6 historical count/denominator data are available.
- Prior/posterior density charts, weight-sensitivity summaries, posterior probability grids, effective sample size, and tipping points.
- Chart/table pairs for Bayesian outputs, including borrowing-weight forest plots and tipping-point curves.
- Two-arm ORR decision support under path A when treatment and control arm data are available.
- Download buttons for raw JSON and Markdown report.

Suggested discounts are not final prior weights. Bayesian results use a lightweight weighted beta-binomial power-prior approximation, not a full robust MAP mixture posterior. They do not define `p_target`, `OR_target`, or `gamma`; those decision thresholds remain user-defined. Any candidate historical trial still needs clinical and statistical expert review before use in a primary Bayesian analysis.
