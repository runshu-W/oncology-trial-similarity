# Oncology Trial Similarity and Prior Borrowing Pipeline 完整解析

## 1. 项目整体目标

本 pipeline 的目标是为一个新的 oncology clinical trial 自动寻找历史 ClinicalTrials.gov oncology database 中最相似、且最有可能用于 Bayesian prior borrowing 的 historical trials。

你的数据库中每个 trial 都以 NCT number 为文件夹名保存，文件夹里包含：

```text
NCTxxxx/
  *.json
  Study_Protocol.pdf
  Statistical_Analysis_Plan.pdf
```

当前 pipeline 的输入是一个新的 oncology trial JSON，输出是：

1. 第一阶段相似 trial 召回结果。
2. 第二阶段 prior-borrowing rerank 结果。
3. 一份 Markdown 可读报告。
4. 每个 candidate 的可借用 endpoint/result quantities、red flags、borrowing suitability 和建议 discount。

换句话说，这个系统不是普通的“找标题相似 trial”，而是一个面向 Bayesian historical borrowing 的 trial selection pipeline。

## 2. Pipeline 总览

整个 pipeline 分成两条主线：

```text
Offline indexing:
  historical trial folders
    -> JSON/PDF discovery
    -> structured extraction
    -> oncology normalization
    -> outcome/result parsing
    -> trial summary generation
    -> multi-aspect embedding
    -> index artifacts

Online query:
  new trial JSON
    -> query summary
    -> query multi-aspect embedding
    -> first-stage retrieval topK
    -> stage-2 prior-borrowing rerank
    -> JSON + Markdown report
```

离线建库只需要对 7173 个 historical trials 跑一次。之后每个新 query trial 只需要进行 query processing 和 search。

## 3. 输入数据

### 3.1 Historical Database Input

默认 historical database 路径写在代码中：

```python
DEFAULT_DB_ROOT = Path(
    "/Users/wang/PHD/clinic.gov/Oncology_All_Trials/Oncology_All_Trials"
)
```

每个 NCT 文件夹包含一个 JSON 文件，以及 protocol/SAP PDF。例如：

```text
/Users/wang/PHD/clinic.gov/Oncology_All_Trials/Oncology_All_Trials/NCT00001337/
  NCT00001337_data.json
  Study_Protocol.pdf
  Statistical_Analysis_Plan.pdf
```

当前实际构建索引时识别到 7173 个 NCT trial folders。

### 3.2 Query Input

online query 输入是一个新的 trial JSON：

```bash
--query-json /path/to/new_trial.json
```

要求格式尽量与 historical trial JSON 一致，包含：

- `Study details`
- `Study details -> 1. NCT number`
- `Study details -> 5. Study Overview`
- `Study details -> 7. Phase`
- `Results Posted`
- `Results Posted -> 1. Intervention/Treatment`
- `Results Posted -> 2. Study Design`
- `Results Posted -> 5. Outcome measures`

如果 query folder 中也有 protocol/SAP PDF，pipeline 会自动发现它们的路径。

## 4. 第一部分：文件发现与读取

### 4.1 发现 JSON 文件

函数：

```python
find_trial_json(folder)
```

逻辑：

1. 在每个 NCT folder 中寻找 `*.json`。
2. 如果存在 `NCTxxxx.json`，优先使用。
3. 否则使用排序后的第一个 JSON 文件。

这个设计兼容你当前的命名方式，例如：

```text
NCT00001337_data.json
```

### 4.2 发现 Protocol 和 SAP

函数：

```python
find_supporting_pdfs(folder)
```

它在每个 NCT folder 中查找 PDF：

- 文件名包含 `protocol` 的 PDF 被识别为 protocol。
- 文件名包含 `statistical_analysis`、`statistical analysis` 或 `sap` 的 PDF 被识别为 SAP。

输出：

```python
{
  "protocol": Path(...) or None,
  "sap": Path(...) or None
}
```

### 4.3 PDF 文本读取

函数：

```python
read_pdf_excerpt(path, max_chars=12000)
```

当前实现是可选的。如果系统中安装了 `pdftotext`，则从 PDF 中读取 excerpt；如果没有安装，则返回空字符串，但仍保留 PDF 路径。

这意味着当前 pipeline 不会因为缺少 PDF parser 而失败。它可以先基于 JSON 跑通，同时为后续 protocol/SAP parsing 留好接口。

输出 summary 中会记录：

```json
"source_documents": {
  "json_path": "...",
  "protocol_pdf": "...",
  "sap_pdf": "...",
  "protocol_text_available": false,
  "sap_text_available": false
}
```

## 5. 第二部分：原始 JSON 结构化抽取

### 5.1 TrialRecord 对象

pipeline 用 `TrialRecord` 保存单个 trial 的信息：

```python
@dataclass
class TrialRecord:
    nct_id: str
    folder: Path
    json_path: Path
    protocol_path: Path | None
    sap_path: Path | None
    raw_json: dict[str, Any]
    extracted: dict[str, Any]
```

它同时保存：

- NCT ID
- folder path
- JSON path
- protocol/SAP path
- raw JSON
- extracted structured fields

这样做的好处是：后续所有 summary、embedding 和 rerank 结果都能追溯回原始文件。

### 5.2 JSON 读取

函数：

```python
read_json(path)
```

使用 Python 标准库 `json.load` 读取 UTF-8 JSON。

### 5.3 安全读取嵌套字段

函数：

```python
get_nested(obj, *keys, default="")
```

ClinicalTrials.gov JSON 可能缺字段，因此不能直接硬访问嵌套字段。`get_nested` 会安全地从 dict 中取值，缺失时返回 default。

例如：

```python
overview = get_nested(details, "5. Study Overview", default={})
```

### 5.4 文本清洗

函数：

```python
clean_text(value)
```

作用：

- 将 list 拼接成 `;` 分隔文本。
- 将 dict 转成 `key: value` 文本。
- 合并多余空白和换行。
- 对所有字段统一格式。

这是很关键的一步，因为 clinical trial JSON 中经常有长文本、换行、嵌套 dict、长 arm label 等。

## 6. 第三部分：Outcome 和 Result 解析

这是当前 pipeline 中与 prior borrowing 最直接相关的部分。

### 6.1 输入

输入字段：

```text
Results Posted -> 5. Outcome measures
```

每个 outcome 通常包含：

- Type
- Title
- Description
- Time Frame
- Population Description
- Unit of Measure
- Param Type
- Data Table

### 6.2 提取 outcome metadata

函数：

```python
extract_outcomes(results_posted)
```

每个 outcome 会被转成：

```json
{
  "type": "PRIMARY",
  "title": "Overall Response ...",
  "description": "...",
  "time_frame": "...",
  "population_description": "...",
  "unit": "Participants",
  "param_type": "COUNT_OF_PARTICIPANTS",
  "denominators": [...],
  "measurements": [...],
  "arm_results": [...]
}
```

### 6.3 解析 denominator

如果 `Data Table` 中有：

```json
{
  "Category": "Denominator (Participants)",
  "Arm A": "235",
  "Arm B": "112"
}
```

pipeline 会解析成：

```json
[
  {
    "arm": "Arm A",
    "denominator": 235.0,
    "raw": "235"
  },
  {
    "arm": "Arm B",
    "denominator": 112.0,
    "raw": "112"
  }
]
```

### 6.4 解析 measurement

如果 `Data Table` 中有：

```json
{
  "Category": "Measurement",
  "Arm A": "216 (91.9%)",
  "Arm B": "94 (83.9%)"
}
```

函数：

```python
parse_count_percent(text)
```

会用正则表达式提取：

```json
{
  "raw": "216 (91.9%)",
  "count": 216.0,
  "percent": 91.9
}
```

### 6.5 合并 arm-level result

pipeline 会把同一个 arm 的 denominator 和 measurement 合并：

```json
{
  "arm": "With EPOCH-R ...",
  "count": 216.0,
  "denominator": 235.0,
  "percent": 91.9,
  "proportion": 0.919149,
  "raw": "216 (91.9%)"
}
```

这个结构就是后续 Bayesian prior borrowing 最重要的候选数据。

例如 ORR 可以转化为 beta-binomial prior：

```text
r = response count
n = denominator
p ~ Beta(alpha0 + w*r, beta0 + w*(n-r))
```

其中 `w` 是后续 rerank 给出的 borrowing discount。

## 7. 第四部分：Endpoint Family 标准化

函数：

```python
infer_endpoint_family(title)
```

它将 endpoint title 归类为 endpoint family：

- PFS
- OS
- EFS
- DFS
- RFS
- DOR
- DCR
- ORR/CR/PR
- CR
- PR
- MRD
- DLT
- Safety/AE
- Treatment discontinuation
- Other

例如：

```text
Progression Free Survival (PFS) -> PFS
Overall Response -> ORR/CR/PR
Adverse Events -> Safety/AE
```

为什么要做这一步：

1. 不同 trial 对同一 endpoint 的写法可能不同。
2. prior borrowing 需要判断 endpoint 是否可比。
3. rerank 阶段会用 endpoint family overlap 判断 candidate 是否适合借用。

## 8. 第五部分：Oncology 概念标准化

函数：

```python
infer_oncology_concepts(text)
```

输入是 trial title、summary、description、intervention、protocol/SAP excerpt 拼接后的文本。

它用 rule-based regex 抽取以下信息。

### 8.1 Primary Site

例如：

- Hematologic malignancy
- Lung
- Breast
- Ovary
- Colorectal
- Pancreas
- Prostate
- Skin
- Brain/CNS

### 8.2 Histology

例如：

- DLBCL
- PMBL
- Burkitt lymphoma
- Mantle cell lymphoma
- Non-Hodgkin lymphoma
- Hodgkin lymphoma
- Multiple myeloma
- ALL
- AML
- CLL
- NSCLC
- SCLC

### 8.3 Molecular Marker

例如：

- CD20
- HER2
- EGFR
- ALK
- BRAF
- KRAS
- BCL2
- PD-L1
- MSI
- MMR

### 8.4 Line of Therapy

规则识别：

- `relapsed` / `refractory` / `R/R` / `previously treated`
  -> `Relapsed/refractory or previously treated`

- `previously untreated` / `untreated` / `frontline` / `first-line` / `newly diagnosed`
  -> `Frontline / previously untreated`

- `maintenance`
  -> `Maintenance`

- `neoadjuvant`
  -> `Neoadjuvant`

- `adjuvant`
  -> `Adjuvant`

Line of therapy 是 prior borrowing 中非常重要的变量。相同 cancer type 下，frontline 和 relapsed/refractory 的 historical response rate 往往完全不同。

### 8.5 Age Group

规则识别：

- Pediatric
- Adult
- Pediatric and adult
- Not reported

## 9. 第六部分：Intervention 和 Regimen 标准化

函数：

```python
infer_intervention_concepts(interventions, text)
```

它抽取两类信息：

### 9.1 Drug Class

例如：

- Anti-CD20 antibody
- Chemotherapy
- Corticosteroid
- PD-1/PD-L1 inhibitor
- BTK inhibitor
- Proteasome inhibitor
- Immunomodulatory drug
- PI3K inhibitor
- Anti-VEGF antibody

### 9.2 Backbone Regimen

例如：

- DA-EPOCH-R
- EPOCH-R
- EPOCH
- R-CHOP
- CHOP
- R-ICE
- ICE
- Gemcitabine-based
- Platinum-based

为什么要抽 regimen backbone：

因为对于 oncology prior borrowing，药物 class 相似还不够。一个 anti-CD20 antibody trial 和另一个 anti-CD20 antibody trial 未必可比；但如果都是 DA-EPOCH-R，临床可比性会更强。

当前规则中还处理了层级去重。例如识别到 `DA-EPOCH-R` 时，会移除较泛化的 `EPOCH-R` 和 `EPOCH`，避免一个 trial 同时被标成多个重复 backbone。

## 10. 第七部分：Trial Design 标准化

函数：

```python
infer_design_concepts(design, outcomes)
```

它抽取：

- randomized: Yes / No / Not reported
- single_or_multi_arm: Single-arm / Multi-arm / Not reported
- number_of_arms

设计信息来自：

```text
Results Posted -> 2. Study Design
```

例如：

```json
{
  "Allocation": "Non_randomized",
  "Interventional Model": "Single_group",
  "Masking": "None",
  "Primary Purpose": "Treatment"
}
```

会被标准化为：

```json
{
  "single_or_multi_arm": "Single-arm",
  "randomized": "No",
  "number_of_arms": "2"
}
```

Design 对 prior borrowing 很重要，因为 randomized controlled trial、single-arm phase 2、dose-finding phase 1 的结果可借用方式不同。

## 11. 第八部分：Rule-based Trial Summary

函数：

```python
make_rule_based_summary(extracted)
```

这是整个 pipeline 的中心步骤。它把 messy raw JSON 转成统一 summary schema。

### 11.1 Summary 顶层字段

输出包括：

```json
{
  "nct_id": "...",
  "brief_title": "...",
  "phase": "...",
  "status": "...",
  "cancer_type": {...},
  "population": {...},
  "intervention": {...},
  "design": {...},
  "endpoints": {...},
  "results": {...},
  "borrowing_relevance": {...},
  "source_documents": {...},
  "one_paragraph_summary_for_embedding": "..."
}
```

### 11.2 Cancer Type

来自 `infer_oncology_concepts`：

```json
"cancer_type": {
  "primary_site": ["Hematologic malignancy"],
  "histology": ["Non-Hodgkin lymphoma", "PMBL"],
  "molecular_marker": ["CD20"],
  "stage_or_risk": ["Not reported"],
  "line_of_therapy": "Frontline / previously untreated",
  "prior_treatment": "Not reported"
}
```

### 11.3 Intervention

来自 intervention field 和文本规则：

```json
"intervention": {
  "experimental_regimen": "...",
  "control_or_comparator": "Not reported",
  "drug_classes": ["Anti-CD20 antibody", "Chemotherapy", "Corticosteroid"],
  "backbone_regimen": ["DA-EPOCH-R"],
  "dose_schedule_summary": "See source record",
  "treatment_duration": "Not reported"
}
```

### 11.4 Endpoints

Primary endpoints 会保留完整结构：

```json
"endpoints": {
  "primary": [
    {
      "type": "PRIMARY",
      "title": "Overall Response ...",
      "endpoint_family": "ORR/CR/PR",
      "arm_results": [...]
    }
  ],
  "secondary_or_other": [...]
}
```

### 11.5 Results

结果部分包括：

```json
"results": {
  "has_posted_results": true,
  "primary_results": [...],
  "safety_results": [...],
  "denominators": [...],
  "follow_up_duration": "Not normalized"
}
```

### 11.6 Borrowing Relevance

这是为 prior borrowing 专门设计的字段：

```json
"borrowing_relevance": {
  "borrowable_quantities": [...],
  "major_similarity_drivers": [...],
  "major_nonborrowability_risks": [...],
  "notes": "Rule-based fallback summary. Use LLM prompt for stronger normalization."
}
```

其中 `borrowable_quantities` 是最关键输出之一。它保存 primary endpoint 的 arm-level count、denominator、percent、proportion。

例如对 `NCT00001337`：

```json
{
  "endpoint": "Overall Response (Complete Response + Partial Response)",
  "endpoint_family": "ORR/CR/PR",
  "unit": "Participants",
  "param_type": "COUNT_OF_PARTICIPANTS",
  "arm_results": [
    {
      "arm": "With EPOCH-R ...",
      "count": 216.0,
      "denominator": 235.0,
      "percent": 91.9,
      "proportion": 0.919149
    },
    {
      "arm": "EPOCH Alone ...",
      "count": 94.0,
      "denominator": 112.0,
      "percent": 83.9,
      "proportion": 0.839286
    }
  ]
}
```

## 12. 第九部分：Multi-aspect Embedding

pipeline 借鉴 Trial2Vec 的思想：clinical trial 不应该被压成一段普通文本，而应该拆成不同 aspect 分别比较。

### 12.1 Aspect 权重

代码中定义：

```python
ASPECT_WEIGHTS = {
    "disease_population": 0.30,
    "intervention": 0.25,
    "endpoint": 0.20,
    "design": 0.15,
    "results_safety": 0.10,
}
```

含义：

- disease/population 最重要，权重 0.30。
- intervention/regimen 次重要，权重 0.25。
- endpoint 权重 0.20。
- design 权重 0.15。
- results/safety 权重 0.10。

### 12.2 Aspect Text

函数：

```python
aspect_text(summary, aspect)
```

不同 aspect 使用 summary 的不同部分：

```text
disease_population -> cancer_type + population
intervention       -> intervention
endpoint           -> endpoints
design             -> design
results_safety     -> results
```

### 12.3 Retrieval Embedding Backend

函数：

```python
hashing_embedding(text, dim=2048)
```

当前实现的 intended/default retrieval path 是 ClinicalBERT-compatible multi-aspect embedding：当 ML dependencies 和模型可用时，first-stage retrieval 应使用 biomedical/clinical encoder 表示不同 aspect 的语义相似度。

hashing embedding 仍保留为 lightweight fallback/smoke-test backend。它是本地 deterministic hashing embedding：

1. 正则分词。
2. 对 token 做 blake2b hash。
3. 映射到 2048 维向量。
4. 根据 hash 值加正负号。
5. L2 normalization。

优点：

- 不需要外部 API。
- 不需要下载模型。
- 快速、稳定、可复现。
- 适合验证 pipeline。

限制：

- 只能捕捉词面相似。
- 不能理解医学同义词和深层语义。
- 主要用于缺少 ML runtime/model 时的 fallback 或 smoke test。

Trial2Vec 是可选 Stage-1 retrieval backend：在已经构建 Trial2Vec index 后，可以用 Trial2Vec 表示替代 ClinicalBERT-compatible embedding。SECRET-style backend 现在也可作为可选 Stage-1 retrieval path：它把 structured summary 转成 deterministic Q/A sections，并按 section-level cosine similarity 检索候选 trial。

## 13. 第十部分：Offline Index Construction

命令：

```bash
python3 oncology_trial_similarity_pipeline.py build-index \
  --db-root /Users/wang/PHD/clinic.gov/Oncology_All_Trials/Oncology_All_Trials \
  --output-dir artifacts/oncology_trial_similarity
```

### 13.1 处理过程

对每个 NCT folder：

1. 找 JSON。
2. 找 protocol/SAP PDF。
3. 读取 JSON。
4. 抽取结构化字段。
5. 解析 outcomes 和 arm-level results。
6. 做 oncology/intervention/design normalization。
7. 生成 trial summary。
8. 生成 five-aspect embeddings。
9. 写入索引文件。

### 13.2 输出 artifacts

当前输出：

```text
artifacts/oncology_trial_similarity/trial_summaries.jsonl
artifacts/oncology_trial_similarity/trial_embeddings.npz
```

实际文件大小：

```text
trial_summaries.jsonl: 约 219MB
trial_embeddings.npz: 约 9.4MB
```

### 13.3 trial_summaries.jsonl

每一行是一个 trial 的 structured summary。

它包含：

- NCT ID
- title
- phase/status
- cancer type
- population
- intervention
- design
- endpoints
- results
- borrowable quantities
- source documents

### 13.4 trial_embeddings.npz

这是 numpy 压缩文件，包含：

```text
nct_ids
disease_population
intervention
endpoint
design
results_safety
```

每个 aspect 是一个矩阵：

```text
number_of_trials x 2048
```

## 14. 第十一部分：Online Query Search

查询命令：

```bash
python3 oncology_trial_similarity_pipeline.py search \
  --query-json /path/to/new_trial.json \
  --index-dir artifacts/oncology_trial_similarity \
  --top-k 100 \
  --output artifacts/oncology_trial_similarity/new_trial_top100.json
```

### 14.1 Query Processing

query trial 会经过与 historical trial 相同的处理：

1. 读取 query JSON。
2. 找 query folder 下的 protocol/SAP。
3. 生成 query summary。
4. 生成 query five-aspect embeddings。

### 14.2 First-stage Retrieval

对每个 historical trial，计算五个 aspect 的 cosine similarity：

```python
sim = cosine(query_emb[aspect], embeddings[aspect][idx])
```

然后加权求和：

```python
score += weight * sim
```

输出：

```json
{
  "nct_id": "NCT...",
  "score": 0.7145,
  "score_0_100": 71.45,
  "aspect_scores": {
    "disease_population": 0.72,
    "intervention": 0.66,
    "endpoint": 0.80,
    "design": 0.55,
    "results_safety": 0.60
  }
}
```

### 14.3 为什么 first-stage retrieval 还不够

第一阶段主要解决 recall 问题：从 7173 个 trials 中找出可能相关的一批 candidates。

如果运行环境只能使用 hashing fallback，第一阶段仍容易受到词面重叠影响。例如两个 trial 都包含 rituximab 或 lymphoma，就可能排得比较靠前，但它们的 line of therapy、histology、endpoint 或 design 未必适合 prior borrowing。即使使用 ClinicalBERT-compatible 或 optional Trial2Vec retrieval，第一阶段也主要负责 recall，而不是最终 prior-borrowing suitability judgment。

所以还需要第二阶段 rerank。

## 15. 第十二部分：Stage 2 Prior-borrowing Rerank

第二阶段是当前最终版 pipeline 的关键增强。

运行命令：

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

### 15.1 Rerank 输入

输入是：

- query summary
- first-stage top100 candidates

每个 candidate 已经包含：

- retrieval score
- cancer type
- intervention
- design
- results
- borrowable quantities
- nonborrowability risks

### 15.2 Rerank 方法

函数：

```python
score_prior_borrowing_pair(query_summary, candidate)
```

它用 deterministic structured rules 对每个 query-candidate pair 评分。

该 reranker 是 LLM reranker 的本地可运行替代版本。它不需要外部 API，但输出结构与未来 LLM reranker 兼容。

### 15.3 六个维度分数

每个 candidate 会得到：

```json
"dimension_scores": {
  "disease_population_match": 0,
  "treatment_regimen_match": 0,
  "endpoint_estimand_match": 0,
  "design_phase_match": 0,
  "result_usability": 0,
  "safety_and_followup_relevance": 0
}
```

每个维度范围是 0 到 5。

## 16. Rerank 维度详解

### 16.1 disease_population_match

这个维度比较：

- histology overlap
- primary site overlap
- molecular marker overlap
- line of therapy match

内部权重：

```text
histology: 45%
primary site: 25%
molecular marker: 15%
line of therapy: 15%
```

如果 query 是 frontline PMBL/NHL，而 candidate 是 relapsed/refractory DLBCL，则会产生 red flag：

```text
Treatment line mismatch
Low disease/population match
```

### 16.2 treatment_regimen_match

这个维度比较：

- backbone regimen overlap
- drug class overlap

内部权重：

```text
backbone regimen: 70%
drug class: 30%
```

这体现了 oncology prior borrowing 的实际逻辑：同样是 chemotherapy 不够，最好 regimen backbone 也相近，例如 DA-EPOCH-R vs DA-EPOCH-R。

### 16.3 endpoint_estimand_match

这个维度比较：

- primary endpoint family overlap
- denominator availability

内部权重：

```text
endpoint family overlap: 75%
denominator available: 25%
```

如果 query 的 primary endpoint 是 ORR/PFS，而 candidate 没有相同 endpoint family，会产生：

```text
No primary endpoint-family overlap
Low endpoint/estimand match
```

### 16.4 design_phase_match

这个维度比较：

- phase similarity
- single-arm vs multi-arm 是否一致
- randomized status 是否一致

内部权重：

```text
phase: 40%
arm structure: 30%
randomization: 30%
```

phase 完全一致得分最高；phase 差距不超过 1 也会给部分分数。

### 16.5 result_usability

这个维度判断 candidate 的结果能不能真正用于 prior：

- 是否有 posted results
- 是否有 arm-level count/denominator
- 是否有 denominator 信息

内部权重：

```text
has posted results: 40%
usable arm-level results: 40%
denominators available: 20%
```

如果没有 arm-level count/denominator，会产生：

```text
No arm-level count/denominator pair found for primary borrowable quantities
```

### 16.6 safety_and_followup_relevance

这个维度目前比较轻量，判断：

- 是否有 safety endpoint
- follow-up duration 是否已标准化

当前 follow-up 仍多为 `Not normalized`，所以该维度通常较低。后续接入 protocol/SAP 和 LLM summary 后，这个维度会更有价值。

## 17. Overall Score 与 Suitability

### 17.1 Clinical Score

六个维度会加权成 clinical score：

```text
disease_population_match: 30%
treatment_regimen_match: 25%
endpoint_estimand_match: 20%
design_phase_match: 10%
result_usability: 10%
safety_and_followup_relevance: 5%
```

### 17.2 与 Retrieval Score 融合

最终 score 结合：

```text
75% structured clinical rerank score
25% first-stage retrieval score
```

也就是说，最终排序主要看 prior-borrowing 结构化匹配，而不是纯 embedding similarity。

### 17.3 Suitability 分层

根据 overall score 和 red flags，candidate 被分为：

```text
high
medium
low
do_not_borrow
```

当前规则：

- `overall >= 80` 且没有低匹配 red flag -> high
- `overall >= 60` -> medium
- `overall >= 40` -> low
- 其他 -> do_not_borrow

### 17.4 Borrowing Discount

每个 suitability 对应一个建议 discount：

```text
high: 0.75
medium: 0.40
low: 0.15
do_not_borrow: 0.00
```

这个 discount 不是最终统计结论，而是 power prior、commensurate prior 或 robust mixture prior 的初始敏感性分析建议。

## 18. Stage 2 输出 JSON

当运行：

```bash
--rerank --rerank-top-n 100
```

输出 JSON 包含：

```json
{
  "query_summary": {...},
  "top_matches": [...],
  "top10": [...],
  "reranked_top_matches": [...],
  "reranked_top10": [...]
}
```

### 18.1 query_summary

表示系统如何理解 query trial。

例如 `NCT00001337` 被识别为：

```json
{
  "histology": ["Non-Hodgkin lymphoma", "PMBL"],
  "molecular_marker": ["CD20"],
  "line_of_therapy": "Frontline / previously untreated",
  "backbone_regimen": ["DA-EPOCH-R"],
  "primary endpoints": ["ORR/CR/PR", "PFS"]
}
```

### 18.2 top_matches

这是第一阶段 retrieval 排序结果。

字段包括：

- nct_id
- score
- score_0_100
- aspect_scores
- title
- phase
- status
- cancer_type
- intervention
- design
- results
- result_usability
- similarity_drivers
- nonborrowability_risks
- borrowable_quantities

### 18.3 reranked_top_matches

这是第二阶段 prior-borrowing rerank 后的结果。

每个 candidate 包括：

```json
{
  "rank": 1,
  "candidate_nct_id": "NCT...",
  "title": "...",
  "retrieval_score": 71.45,
  "overall_similarity_score": 73.2,
  "prior_borrowing_suitability": "medium",
  "suggested_borrowing_discount": 0.4,
  "dimension_scores": {...},
  "borrowable_quantities": [...],
  "required_adjustments": [...],
  "explanation": "...",
  "red_flags": [...],
  "candidate_snapshot": {...}
}
```

## 19. Markdown Report 输出

如果指定：

```bash
--report-output artifacts/oncology_trial_similarity/new_trial_stage2_report.md
```

会生成可读报告。

报告包括：

1. Query trial title。
2. Query cancer/intervention/design summary。
3. Reranked top matches table。
4. 每个 candidate 的：
   - rank
   - NCT ID
   - score
   - suitability
   - discount
   - key rationale
   - red flags

这个报告适合人工 review 和汇报展示。

## 20. NCT00001337 实际运行示例

已经运行：

```bash
python3 oncology_trial_similarity_pipeline.py search \
  --query-json /Users/wang/PHD/clinic.gov/Oncology_All_Trials/Oncology_All_Trials/NCT00001337/NCT00001337_data.json \
  --index-dir artifacts/oncology_trial_similarity \
  --top-k 100 \
  --rerank \
  --rerank-top-n 100 \
  --output artifacts/oncology_trial_similarity/NCT00001337_stage2_reranked.json \
  --report-output artifacts/oncology_trial_similarity/NCT00001337_stage2_report.md
```

输出文件：

```text
artifacts/oncology_trial_similarity/NCT00001337_stage2_reranked.json
artifacts/oncology_trial_similarity/NCT00001337_stage2_report.md
```

实际结果：

```text
query: NCT00001337
first-stage top_matches: 100
reranked candidates: 100
```

reranked top examples：

```text
1. NCT02481310 - score 73.2 - medium - discount 0.4
2. NCT00001379 - score 73.12 - medium - discount 0.4
3. NCT00006436 - score 62.76 - medium - discount 0.4
4. NCT01092182 - score 61.96 - medium - discount 0.4
5. NCT01445535 - score 61.57 - medium - discount 0.4
```

注意：这些结果仍需要人工 review。当前 reranker 是 deterministic heuristic reranker，不是最终医学结论。

## 21. Pipeline 如何服务 Bayesian Prior Borrowing

### 21.1 二分类 endpoint

对于 ORR、CR、AE discontinuation 这类 endpoint，pipeline 能抽取：

```text
count
denominator
proportion
```

可以用于 beta-binomial prior：

```text
p ~ Beta(alpha0 + w*r, beta0 + w*(n-r))
```

其中：

- `r` 是 historical response count
- `n` 是 denominator
- `w` 是 `suggested_borrowing_discount`

### 21.2 Mixture-prior extension

The weighted beta-binomial approximation can be extended into an exploratory robust mixture prior. For each candidate historical trial `i`, the candidate-specific beta component is:

```text
Beta(1 + a_i y_i, 1 + a_i(n_i - y_i))
```

where `y_i` is the historical response count, `n_i` is the historical denominator, and `a_i` is the effective sample-size discount for that trial. The full prior can then be written as:

```text
p_prior(p) = lambda_0 Beta(1, 1) + sum_i lambda_i Beta(1 + a_i y_i, 1 + a_i(n_i - y_i))
```

The mixture weights satisfy `lambda_0 + sum_i lambda_i = 1`, with `lambda_0 >= 0` and every `lambda_i >= 0`.

Here `lambda_0` is the weak-prior component weight and `lambda_i` is the prior mixture/component weight, or prior mass, for candidate trial `i`. This deliberately separates `a_i`, which controls how much information a candidate contributes inside its beta component, from `lambda_i`, which controls how much prior mass is assigned to that candidate component in the mixture. If the observed query data are later used to update component responsibilities, those posterior component weights should be denoted `lambda_i_post`.

The `lambda_i` values may be trained retrospectively by treating completed trials as pseudo-queries, retrieving candidate historical trials, and fitting weights to predict held-out query outcomes. The pseudo-query outcome must be held out from retrieval, reranking, feature construction, tuning, and model selection to avoid leakage; when generating pipeline-result JSONL, use `--hide-query-outcomes-for-retrieval`, which records leakage-control metadata and stores query outcomes separately under `heldout_query_outcomes`. This retrospective training is a calibration and sensitivity-analysis tool; it does not replace clinical/statistical expert adjudication before borrowing historical evidence in a primary analysis.

### 21.3 Time-to-event endpoint

对于 PFS/OS，当前 JSON 中通常只有 count/percent 或 participant-level summaries，不一定足够构建严格 survival prior。

更理想的信息包括：

- hazard ratio
- event count
- median survival
- Kaplan-Meier curve
- reconstructed IPD
- censoring rule
- follow-up duration

这些信息通常需要 protocol/SAP 或 publication 补充。

### 21.4 Borrowing 策略

pipeline 输出的 suitability 可以映射为：

```text
high:
  commensurate prior 或 power prior，较高 borrowing weight

medium:
  robust mixture prior 或 moderate discount

low:
  sensitivity analysis only

do_not_borrow:
  不进入 primary analysis，只作为背景证据
```

## 22. 当前 Pipeline 的优势

### 22.1 端到端可运行

已经可以从本地 7173 个 trials 构建索引，并对一个 query trial 输出 top100、reranked top10 和 Markdown report。

### 22.2 结构化可解释

输出不是一个黑盒分数，而是包含：

- disease match
- intervention match
- endpoint match
- design match
- result usability
- red flags

### 22.3 与 Trial2Vec 思路一致

pipeline 使用 multi-aspect representation，不把 trial 当作一整段文本。当前默认语义召回路径是 ClinicalBERT-compatible embedding；Trial2Vec 在本 revision 中是可选 Stage-1 retrieval backend，需要先构建 Trial2Vec index。

### 22.4 与 SECRET 思路兼容

当前 rule-based summary 已经可以生成 SECRET-style deterministic Q/A sections，用于 section-level retrieval 和 `secret_section_scores` 解释。需要注意的是，这仍是 SECRET-style MVP，不是完整 LLM protocol summarization + reviewer-facing explanation workflow。

### 22.5 可直接连接 Bayesian prior

对于二分类 endpoint，已经抽取了 count、denominator 和 proportion，并给出 suggested borrowing discount。

## 23. 当前限制

### 23.1 Retrieval backend status

当前 first-stage retrieval 的 intended/default path 是 ClinicalBERT-compatible multi-aspect embedding，前提是 ML dependencies 和模型可用。

backend 状态如下：

- ClinicalBERT-compatible embedding：默认语义 retrieval path。
- Hashing embedding：lightweight fallback/smoke-test backend，只能捕捉 token overlap。
- Trial2Vec：可选 Stage-1 retrieval backend，需要先构建 Trial2Vec index。
- SECRET-style：可选 Stage-1 retrieval backend，使用 deterministic Q/A sections 和 section-weighted cosine search；不是完整 SECRET 论文 workflow。

### 23.2 Rule-based normalization 有边界

当前 cancer type、histology、line of therapy 和 regimen backbone 都是 regex rules。

优点是可解释、可离线运行；缺点是覆盖有限，容易漏掉复杂表达。

### 23.3 Protocol/SAP 尚未充分使用

pipeline 已能发现 protocol/SAP path，也预留了 `read_pdf_excerpt`。但当前环境缺少 `pdftotext` 或 Python PDF parser，所以实际 summary 主要依赖 JSON。

后续需要安装 PDF parser，并抽取：

- eligibility
- estimand
- analysis population
- censoring rule
- dose modification
- missing data handling
- interim analysis

### 23.4 Follow-up 和 estimand 还不充分

当前 `follow_up_duration` 多为 `Not normalized`。Endpoint family 不等于完整 estimand，PFS/OS 尤其需要更多统计定义。

### 23.5 Reranker 是 deterministic heuristic

当前 stage 2 reranker 是本地规则版。它已经能产生结构化审查结果，但还不是 expert-level LLM/clinical-statistical reviewer。

后续建议在相同输出 schema 下接入 LLM reranker。

## 24. 后续推荐升级路线

### 24.1 加入 PDF parser

安装 `pdftotext`、`pypdf` 或 `PyPDF2`，让 protocol/SAP excerpt 真正进入 summary generation。

### 24.2 用 LLM 生成 SECRET-style summary

用现有 prompt 将 JSON + protocol/SAP excerpt 统一转换为 structured summary。

重点补充：

- eligibility
- line of therapy
- prior treatment
- comparator
- endpoint definition
- estimand
- analysis population
- follow-up
- borrowing risks

### 24.3 升级 embedding / retrieval backend

保持 multi-aspect 架构，默认使用 ClinicalBERT-compatible retrieval；也可以继续评估 PubMedBERT、SapBERT、OpenAI embedding、biomedical sentence-transformer，或在构建 index 后使用 optional Trial2Vec backend。hashing embedding 只作为 lightweight fallback/smoke-test backend 保留。

### 24.4 LLM pairwise rerank

保留当前 `dimension_scores` schema，用 LLM 对 top100 做更细致 pairwise judgment。

### 24.5 构建 evaluation benchmark

人工标注一批 query-candidate pairs：

- high
- medium
- low
- do_not_borrow

评价：

- Precision@10
- nDCG@10
- MRR
- expert agreement
- red flag detection rate

## 25. 当前命令总结

### 25.1 构建索引

```bash
python3 oncology_trial_similarity_pipeline.py build-index \
  --db-root /Users/wang/PHD/clinic.gov/Oncology_All_Trials/Oncology_All_Trials \
  --output-dir artifacts/oncology_trial_similarity
```

### 25.2 第一阶段检索

```bash
python3 oncology_trial_similarity_pipeline.py search \
  --query-json /path/to/new_trial.json \
  --index-dir artifacts/oncology_trial_similarity \
  --top-k 100 \
  --output artifacts/oncology_trial_similarity/new_trial_top100.json
```

### 25.3 第二阶段 rerank + report

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

## 26. 一句话总结

当前 pipeline 已经形成完整闭环：它能从本地 ClinicalTrials.gov oncology trial JSON/PDF 文件夹中构建 structured trial index，用 ClinicalBERT-compatible multi-aspect retrieval 做第一阶段候选召回，并可在构建 Trial2Vec index 后切换到 optional Trial2Vec backend，再用 prior-borrowing-oriented structured reranker 对 top100 candidates 进行可解释排序，最终输出 JSON 和 Markdown 报告，为 Bayesian historical prior selection 提供可审查、可追溯、可继续升级的技术基础。
