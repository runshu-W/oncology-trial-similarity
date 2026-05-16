# Oncology Trial Similarity Pipeline for Prior Borrowing

目标：输入一个新的 oncology trial JSON，输出本地 7000 个 ClinicalTrials.gov oncology trials 中最适合用于 prior borrowing 的 top10 historical trials，并给出相似理由、可借用证据、以及不建议借用的风险点。

## 方法选择

推荐采用 Trial2Vec + SECRET hybrid：

1. Trial2Vec 思路：保留 trial 的 meta-structure，不把全文混成一坨文本；分别编码 disease、intervention、eligibility、outcome、phase/design 等 clinical aspects，再聚合为 trial embedding。
2. SECRET 思路：先把 protocol 或 JSON 长文本总结成面向相似检索的 structured trial summary，再基于 summary 做检索；这能缓解 protocol/SAP 太长、字段噪声太多的问题。
3. Prior borrowing 约束：相似不等于可借用。最终 rerank 必须强调 population、tumor type、line of therapy、intervention/regimen、endpoint definition、follow-up、arm structure、standard-of-care era、result availability。

参考依据：

- Trial2Vec 使用 trial meta-structure 和 clinical knowledge 通过 self-supervision 生成 contrastive samples，并编码多方面 trial 信息用于 zero-shot trial similarity search。
- SECRET 针对 clinical trial document similarity，先总结 protocol，再基于 query protocol 搜索相似 historical trials；论文报告其在 overall 和 partial similarity search 上优于 baseline。

## 输入文件假设

数据库路径：

```text
/Users/wang/PHD/clinic.gov/Oncology_All_Trials/Oncology_All_Trials/
  NCT00001337/
    *.json
    Study_Protocol.pdf
    Statistical_Analysis_Plan.pdf
  NCT...
```

每个 trial 的 JSON 至少包含：

- `Study details`
- `Study details -> 5. Study Overview`
- `Study details -> 7. Phase`
- `Results Posted`
- `Results Posted -> 2. Study Design`
- `Results Posted -> 5. Outcome measures`

## 总体流程

1. Ingest
   - 遍历所有 NCT 子文件夹。
   - 找到每个文件夹内的 JSON、protocol PDF、SAP PDF。
   - 先从 JSON 抽取结构化字段；若 JSON 缺 eligibility 或 arms 等重要信息，再从 protocol/SAP 补充。

2. Normalize
   - 统一 cancer type、histology、stage、line of therapy、biomarker、treatment class、endpoint。
   - 将 intervention 拆成 backbone regimen、experimental add-on、drug class、dose/schedule。
   - 将 outcome 拆成 endpoint family，例如 ORR、CR、PFS、OS、DOR、AE discontinuation。

3. SECRET-style summary
   - 对每个 historical trial 生成固定 schema 的 concise summary。
   - 对 query trial 也生成同一 schema summary。
   - 该 summary 是 embedding 和 rerank 的主输入。

4. Multi-aspect embedding
   - 为每个 trial 生成多个 aspect embedding：
     - disease_population
     - intervention
     - design
     - endpoint
     - results_safety
     - full_secret_summary
   - 先用 ANN/cosine 找 top100 candidates。

5. Prior-borrowing rerank
   - 用加权分数 rerank top100：
     - disease/population: 0.30
     - intervention/regimen: 0.25
     - endpoint/statistical estimand: 0.20
     - design/phase/arm/randomization: 0.15
     - temporal/result/safety usability: 0.10
   - 对明显不可借用情况加 penalty：
     - 不同 tumor lineage 或 disease biology
     - 不同 treatment line
     - pediatric vs adult 不一致
     - endpoint 定义不同且无法映射
     - pre/post immunotherapy 或 targeted-therapy era 差异过大
     - no posted results 或 result denominator 不清楚

6. Output
   - 输出 top10：
     - NCT number
     - title
     - phase/design
     - disease/population match
     - intervention match
     - endpoint match
     - reusable prior fields
     - caution
     - final borrowing suitability score

## Prompt 1: SECRET-style Trial Summary

用于每个 historical trial 和 query trial。输入可以是 JSON 抽取文本 + protocol/SAP chunks。

```text
You are a clinical trial similarity extraction system for oncology trials.

Task:
Convert the provided ClinicalTrials.gov trial record and optional protocol/SAP text into a compact, structured summary optimized for clinical trial similarity search and Bayesian prior borrowing.

Important:
- Do not invent facts.
- If a field is missing, write "Not reported".
- Normalize oncology terminology where possible.
- Preserve exact endpoint names and outcome definitions.
- Prefer clinically meaningful concepts over long copied text.
- Focus on whether another trial can borrow information from this trial.

Return valid JSON only with this schema:
{
  "nct_id": "",
  "brief_title": "",
  "phase": "",
  "status": "",
  "cancer_type": {
    "primary_site": "",
    "histology": "",
    "molecular_marker": "",
    "stage_or_risk": "",
    "line_of_therapy": "",
    "prior_treatment": ""
  },
  "population": {
    "age": "",
    "key_inclusion": [],
    "key_exclusion": [],
    "performance_status": "",
    "subgroups": []
  },
  "intervention": {
    "experimental_regimen": "",
    "control_or_comparator": "",
    "drug_classes": [],
    "backbone_regimen": "",
    "dose_schedule_summary": "",
    "treatment_duration": ""
  },
  "design": {
    "allocation": "",
    "interventional_model": "",
    "masking": "",
    "primary_purpose": "",
    "single_or_multi_arm": "",
    "randomized": "",
    "sample_size": "",
    "number_of_arms": "",
    "follow_up": ""
  },
  "endpoints": {
    "primary": [
      {
        "name": "",
        "endpoint_family": "",
        "definition": "",
        "time_frame": "",
        "unit": "",
        "estimand_or_measure": ""
      }
    ],
    "secondary_or_other": []
  },
  "results": {
    "has_posted_results": "",
    "primary_results": [],
    "safety_results": [],
    "denominators": [],
    "follow_up_duration": ""
  },
  "borrowing_relevance": {
    "borrowable_quantities": [],
    "major_similarity_drivers": [],
    "major_nonborrowability_risks": [],
    "notes": ""
  },
  "one_paragraph_summary_for_embedding": ""
}
```

Input trial record:
```json
{{TRIAL_JSON}}
```

Optional protocol/SAP excerpts:
```text
{{PROTOCOL_AND_SAP_EXCERPTS}}
```
```

## Prompt 2: Pairwise Prior Borrowing Rerank

用于 top100 candidates 的精排。它不替代 embedding，而是判断“能不能借 prior”。

```text
You are evaluating whether a historical oncology clinical trial is suitable for Bayesian prior borrowing for a new query trial.

Use only the two structured summaries below.
Do not reward superficial title similarity if disease biology, treatment line, regimen, endpoint, or design are mismatched.

Score each dimension from 0 to 5:
- disease_population_match
- treatment_regimen_match
- endpoint_estimand_match
- design_phase_match
- result_usability
- safety_and_followup_relevance

Then produce:
- overall_similarity_score from 0 to 100
- prior_borrowing_suitability: "high", "medium", "low", or "do_not_borrow"
- borrowable_quantities: specific outcome/result quantities that may inform a prior
- required_adjustments: discounting, commensurate prior, robust mixture prior, subgroup-only borrowing, endpoint transformation, or no borrowing
- explanation: concise clinical rationale
- red_flags: reasons to avoid or strongly down-weight borrowing

Return valid JSON only:
{
  "candidate_nct_id": "",
  "dimension_scores": {
    "disease_population_match": 0,
    "treatment_regimen_match": 0,
    "endpoint_estimand_match": 0,
    "design_phase_match": 0,
    "result_usability": 0,
    "safety_and_followup_relevance": 0
  },
  "overall_similarity_score": 0,
  "prior_borrowing_suitability": "",
  "borrowable_quantities": [],
  "required_adjustments": [],
  "explanation": "",
  "red_flags": []
}

Query trial summary:
```json
{{QUERY_SUMMARY_JSON}}
```

Historical candidate summary:
```json
{{CANDIDATE_SUMMARY_JSON}}
```
```

## Prompt 3: Top10 Final Report

```text
You are preparing a clinical evidence retrieval report for oncology prior borrowing.

Given the query trial summary and reranked candidate trials, produce a top10 report.
Prioritize candidates that match disease biology, line of therapy, regimen backbone, endpoint definition, and available result denominators.

Return Markdown with a compact table and short notes.
For each candidate include:
- rank
- NCT ID
- title
- score
- borrowing suitability
- why similar
- what can be borrowed
- caution/red flags

Query trial summary:
```json
{{QUERY_SUMMARY_JSON}}
```

Reranked candidates:
```json
{{RERANKED_CANDIDATES_JSON}}
```
```

## 推荐输出格式

```json
{
  "query_nct_id": "NEW_OR_NCT_ID",
  "top10": [
    {
      "rank": 1,
      "nct_id": "NCT...",
      "score": 92.4,
      "prior_borrowing_suitability": "high",
      "title": "",
      "match_summary": "",
      "borrowable_quantities": ["ORR denominator and response count", "PFS event count"],
      "red_flags": []
    }
  ]
}
```

## Stage 2: Prior-borrowing Rerank

第一阶段 embedding search 只负责召回候选 trial。第二阶段会对 topN candidates 做 deterministic pairwise prior-borrowing rerank，输出更接近最终分析决策的字段：

- `dimension_scores`
- `overall_similarity_score`
- `prior_borrowing_suitability`
- `suggested_borrowing_discount`
- `required_adjustments`
- `red_flags`
- `borrowable_quantities`

运行示例：

```bash
python3 oncology_trial_similarity_pipeline.py search \
  --query-json /path/to/new_trial.json \
  --index-dir artifacts/oncology_trial_similarity \
  --top-k 100 \
  --rerank \
  --rerank-top-n 100 \
  --output artifacts/oncology_trial_similarity/new_trial_stage2_reranked.json \
  --report-output artifacts/oncology_trial_similarity/new_trial_stage2_report.md
```

`suggested_borrowing_discount` 不是最终统计结论，而是 power prior、commensurate prior 或 robust mixture prior 的初始敏感性分析建议。

## Biomedical Embedding Backend

当前 pipeline 支持两种 embedding backend：

- `hashing`: 默认后端，不依赖模型，适合快速调试。
- `clinicalbert`: 使用本机 HuggingFace cache 中的 `emilyalsentzer/Bio_ClinicalBERT`，通过 `transformers` 做 mean-pooling embedding，输出 768 维向量。

本机已经有 `torch`、`transformers` 和 Bio_ClinicalBERT cache，因此不需要额外安装 `sentence-transformers` 也能运行 biomedical embedding。注意使用 `clinicalbert` 时应调用项目虚拟环境中的 Python：

```bash
.venv/bin/python oncology_trial_similarity_pipeline.py build-index \
  --db-root /Users/wang/PHD/clinic.gov/Oncology_All_Trials/Oncology_All_Trials \
  --output-dir artifacts/oncology_trial_similarity_clinicalbert \
  --embedding-backend clinicalbert \
  --embedding-batch-size 16 \
  --embedding-max-length 256
```

用 ClinicalBERT index 查询时：

```bash
.venv/bin/python oncology_trial_similarity_pipeline.py search \
  --query-json /path/to/new_trial.json \
  --index-dir artifacts/oncology_trial_similarity_clinicalbert \
  --top-k 100 \
  --rerank \
  --rerank-top-n 100 \
  --output artifacts/oncology_trial_similarity_clinicalbert/new_trial_stage2_reranked.json \
  --report-output artifacts/oncology_trial_similarity_clinicalbert/new_trial_stage2_report.md
```

## 实操建议

- 如果 query trial 只有 JSON，没有 protocol/SAP，也能运行；但对 eligibility、dose、endpoint estimand 的判断会弱一些。
- 第一次跑 7000 个 historical trials 时应缓存 summary 和 embedding。之后每个新 query 只需要生成 query summary 和 query embedding。
- top10 不建议只看 cosine similarity；prior borrowing 要做 LLM rerank 或规则 rerank。
- 最终用于 Bayesian prior 时，建议把 score 转换成 discount factor 或 commensurability prior，而不是直接把 historical data 全量合并。
