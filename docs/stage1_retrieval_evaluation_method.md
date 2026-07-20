# Stage 1 Retrieval Evaluation 方法说明

日期：2026-06-04  
对应实现：

- `pipeline/secret_retrieval.py`
- `pipeline/evaluate_stage1_retrieval.py`
- `tests/test_stage1_retrieval_evaluation.py`

## 1. 为什么需要单独评估 Stage 1

Stage 1 的任务不是直接决定能不能借用历史数据，而是尽量把可能相关的 historical trials 召回给 Stage 2。

因此 Stage 1 的评价不能只看最终 lambda NLL，也不能只看 retrieval cosine score。更合理的方式是看两类 proxy：

1. **recall proxy**：topK 里有没有召回 endpoint/result-ready 的候选。
2. **Stage2-ready quality proxy**：rerank 后 top10 的 disease、regimen、endpoint、result usability 是否更好。

如果一个 retrieval backend 找到很多有结果的 trial，但 disease/regimen 不相关，那么 Stage 2 会被迫从差候选里选。反过来，如果 retrieval backend 找到语义很接近的 trial，但没有可用 ORR count/denominator，也无法形成 mixture prior component。

## 2. 输入格式

评估脚本读取一个或多个 pipeline result JSONL：

```bash
python3 pipeline/evaluate_stage1_retrieval.py \
  --results hashing=artifacts/.../hashing_pipeline_results.jsonl \
  --results secret=artifacts/.../secret_pipeline_results.jsonl \
  --baseline-label hashing \
  --endpoint-key ORR \
  --top-k-eval 30 \
  --output-dir artifacts/stage1_retrieval_evaluation
```

每条 result 应包含：

```text
query_summary
top_matches
reranked_top_matches 或 reranked_top10
retrieval_backend
embedding_backend
```

## 3. Query-level 指标

对每个 query `q` 和 backend `b`，令 Stage 1 返回：

```text
C_q^b = {c_1, ..., c_K}
```

如果使用 `--top-k-eval K_eval`，则只评价前 `K_eval` 个候选：

```text
C_q^b(K_eval) = top K_eval candidates
```

### 3.1 Endpoint hit

令 query 的 endpoint family 集合为：

```text
E_q = endpoint families in query primary endpoints
```

候选 trial 的 endpoint family 集合为：

```text
E_i = endpoint families in candidate endpoints and borrowable_quantities
```

如果：

```text
E_q intersection E_i != empty
```

或 endpoint family 与 `endpoint_key` 匹配，例如 `ORR` 匹配 `ORR/CR/PR`，则：

```text
endpoint_hit_i = 1
```

否则：

```text
endpoint_hit_i = 0
```

topK endpoint hit rate：

```text
endpoint_hit_rate_q
= (1 / K_eval) * sum_i endpoint_hit_i
```

### 3.2 Result-ready

如果候选有 posted result 且 denominator 可用，或 endpoint/borrowable quantity 中存在 arm-level result：

```text
result_ready_i = 1
```

否则：

```text
result_ready_i = 0
```

topK result-ready rate：

```text
result_ready_rate_q
= (1 / K_eval) * sum_i result_ready_i
```

### 3.3 Endpoint and result-ready

这是 Stage 1 能不能把“可能形成 prior component 的候选”召回来的核心 proxy：

```text
endpoint_result_ready_i = endpoint_hit_i * result_ready_i
```

```text
endpoint_result_ready_rate_q
= (1 / K_eval) * sum_i endpoint_result_ready_i
```

## 4. Rerank top10 quality 指标

Stage 2 reranker 输出 top10：

```text
R_q^b = {r_1, ..., r_10}
```

对于 reranked candidate，计算：

```text
component_ready_i = endpoint_hit_i * result_ready_i
```

rerank component-ready rate：

```text
component_ready_rate_q
= (1 / |R_q^b|) * sum_i component_ready_i
```

同时报告 Stage 2 dimension scores 的均值：

```text
mean_disease_match
mean_regimen_match
mean_endpoint_match
mean_eligibility_match
mean_result_usability
mean_overall_similarity
mean_redflag_count
```

这些指标回答的问题是：Stage 1 给 Stage 2 的候选，是否更接近临床/统计 borrowing 需要。

## 5. Backend overlap

当比较两个 backend，例如 hashing 和 SECRET，脚本计算同一 query 下 topK candidate 集合的 Jaccard overlap：

```text
J(A, B) = |A intersection B| / |A union B|
```

低 overlap 不一定坏。它表示新 backend 找到了不同候选。是否更好，需要结合 endpoint/result-ready、rerank quality 和最终 downstream NLL 判断。

## 6. 当前 full ORR hashing baseline

对 full ORR retrospective pipeline results：

```text
artifacts/retrospective_lambda_oncology_orr_all/pipeline_results.jsonl
```

生成了：

```text
artifacts/stage1_retrieval_evaluation_orr_all/stage1_retrieval_report.md
```

核心结果：

| Backend | Queries | TopK endpoint+result-ready | Rerank component-ready | Rerank overall | Endpoint match | Disease match | Regimen match |
|---|---:|---:|---:|---:|---:|---:|---:|
| hashing | 1470 | 0.9789 | 0.5450 | 44.2768 | 2.2281 | 1.2480 | 0.5407 |

解释：

- hashing top100 几乎总能召回有 endpoint/result 的 trial。
- 但 reranked top10 中真正 endpoint/result-ready 的比例只有约 0.545。
- disease/regimen match 偏低，说明 Stage 1 仍可能给 Stage 2 很多词面相关但临床 borrowing 不够接近的候选。

## 7. SECRET-style smoke 结果

在 8 个 smoke pseudo-query 上，使用同一 `top-k-eval=30` 比较 hashing 和 SECRET-style direct scoring：

```text
artifacts/stage1_retrieval_evaluation_secret_smoke_top30/stage1_retrieval_report.md
```

| Backend | Queries | Top30 endpoint+result-ready | Rerank component-ready | Rerank overall | Endpoint match | Disease match | Regimen match |
|---|---:|---:|---:|---:|---:|---:|---:|
| hashing | 8 | 0.9792 | 0.6500 | 43.5805 | 2.4604 | 1.0311 | 0.6035 |
| SECRET-style | 8 | 0.9708 | 0.4125 | 46.7133 | 3.3985 | 1.4203 | 0.5500 |

Candidate overlap:

```text
mean top30 Jaccard(secret vs hashing) = 0.0967
mean shared top30 candidates = 5.125
```

解释：

- SECRET-style 找到的 candidate 集合和 hashing 很不一样。
- SECRET-style 的 overall、endpoint、disease match 更高，说明语义检索方向有效。
- 但 smoke 中 SECRET-style reranked top10 的 component-ready rate 更低，说明它会找出更语义相关但不一定有可借用 ORR component 的 trial。
- 因此当前不应盲目用 SECRET-style 替代 hashing/ClinicalBERT，而应作为 Stage 1 candidate generator，并与 Stage 2 component-readiness rerank、lambda NLL downstream evaluation 联合判断。

## 8. 本轮 Stage 1 优化

`pipeline/secret_retrieval.py` 的 `score_secret_index()` 已从 Python candidate loop 改为 NumPy 向量化 cosine scoring。

原逻辑：

```text
for candidate:
  for section:
    cosine(query_section, candidate_section)
```

优化后：

```text
for section:
  similarities = matrix_section @ query_section / norms
weighted_score = sum_section weight_section * similarities_section
```

这个改动把 SECRET-style smoke search 从长时间无输出降到约 3 秒完成 8 个 query 的 direct scoring。

## 9. 下一步

推荐的 Stage 1 优化路线：

1. 用同一批 ORR pseudo-query 跑 full SECRET-style pipeline results。
2. 用本 evaluator 比较 hashing/ClinicalBERT/SECRET/Trial2Vec 的 recall proxy 和 rerank quality。
3. 对每个 Stage 1 backend 重新训练 `two_head_deepsets` lambda model。
4. 用 held-out beta-binomial NLL 判断哪个 Stage 1 backend 真正改善 downstream borrowing。
5. 如果 SECRET 提升 semantic match 但降低 component-ready rate，可以实现 hybrid retrieval：`ClinicalBERT or hashing recall pool + SECRET section rerank + Stage2 borrowing rerank`。

## 10. Full ORR SECRET pool rerank 结果

根据 smoke 结果，直接全库 SECRET-style retrieval 会找到不同候选，但 component-ready rate 可能下降。因此本轮实现了更稳健的 hybrid：

```text
hashing/ClinicalBERT top100 recall pool
-> SECRET-style section similarity pool rerank
-> Stage2 prior-borrowing rerank top10
-> two_head_deepsets lambda training
```

对应脚本：

```text
pipeline/apply_secret_pool_rerank.py
```

生成 full ORR hybrid pipeline results：

```text
artifacts/stage1_secret_pool_rerank_orr_all/pipeline_results.jsonl
```

### 10.1 Full Stage1 proxy 对比

评估报告：

```text
artifacts/stage1_retrieval_evaluation_secret_pool_orr_all/stage1_retrieval_report.md
```

| Backend | Queries | Top100 endpoint+result-ready | Rerank component-ready | Rerank overall | Endpoint match | Disease match | Result usability | Red flags |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hashing | 1470 | 0.9789 | 0.5450 | 44.2768 | 2.2281 | 1.2480 | 3.6450 | 3.7387 |
| SECRET pool rerank | 1470 | 0.9789 | 0.6361 | 48.0341 | 3.3327 | 1.3900 | 3.8905 | 3.1828 |

解释：

- SECRET pool rerank 保留同一个 top100 recall pool，所以 top100 endpoint/result-ready rate 不下降。
- Stage2 top10 component-ready rate 从 `0.5450` 提升到 `0.6361`。
- Endpoint match 从 `2.2281` 提升到 `3.3327`。
- Overall similarity 从 `44.2768` 提升到 `48.0341`。
- 平均 red flag count 从 `3.7387` 降到 `3.1828`。

这说明 hybrid 比“直接 SECRET 全库替换”更适合当前任务：它保留 recall pool，同时用 SECRET-style section semantics 改善 Stage2 候选质量。

### 10.2 Downstream lambda NLL 对比

用 SECRET pool rerank results 重新训练 `two_head_deepsets` lambda model：

```text
artifacts/retrospective_lambda_secret_pool_orr_all
```

与原 hashing two-head 对比：

```text
artifacts/stage1_secret_pool_downstream_comparison/comparison_report.md
```

| Run | Examples | Components | Eval learned NLL | Learned - rule NLL | MAE | RMSE | Corr | Temporal learned - rule |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hashing + two_head_deepsets | 1407 | 4427 | 3.225659 | -0.043240 | 0.286604 | 0.329059 | 0.028905 | -0.079173 |
| SECRET pool + two_head_deepsets | 1414 | 6039 | 2.973668 | -0.129279 | 0.295364 | 0.337528 | 0.053002 | -0.092257 |

保守 paired common-query check：

| Scope | Common queries | Hashing learned NLL | SECRET pool learned NLL | SECRET - hashing |
|---|---:|---:|---:|---:|
| eval | 66 | 3.400884 | 3.263035 | -0.137848 |
| all | 1362 | 3.053042 | 3.019367 | -0.033675 |

注意：

- Run-level NLL 不是完全 paired comparison，因为 SECRET pool 产生了 `1414` 个 examples，而 hashing baseline 是 `1407` 个。
- 但是 common-query sanity check 仍然支持 SECRET pool 方向：在 66 个共同 eval queries 上，SECRET pool learned NLL 平均低 `0.137848`。
- 因此当前 Stage1 推荐从“单纯 hashing/ClinicalBERT recall”升级为 **recall pool + SECRET-style pool rerank**，再接 Stage2 rerank 和 two-head lambda。
