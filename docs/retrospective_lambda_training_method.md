# Retrospective Lambda Training 方法说明

日期：2026-06-04  
适用分支：`codex/trial2vec-secret-mixture-prior`  
对应实现：

- `pipeline/train_retrospective_lambda_model.py`
- `pipeline/evaluate_retrospective_lambda_model.py`
- `pipeline/mixture_prior.py`
- `pipeline/oncology_trial_similarity_pipeline.py`

## 0. 本次全量 ORR 训练结果先说结论

本次已经不再是 116 个 example 的 pilot run，而是对全部 ORR eligible pseudo-query 做了 full retrospective run。

数据规模：

| 项目 | 数值 |
|---|---:|
| 扫描 trial JSON | 7,173 |
| ORR eligible pseudo-query | 1,470 |
| 成功构造 mixture example | 1,407 |
| 被排除 pseudo-query | 63 |
| Train examples | 1,126 |
| Eval examples | 281 |
| Mixture components | 4,427 |
| 平均 components/query | 3.146 |
| Epochs | 100 |
| Bootstrap iterations | 1,000 |

核心 held-out eval 结果如下。NLL 是 negative log likelihood，越低越好：

| Method | Eval mean NLL | 解释 |
|---|---:|---|
| Weak-only prior | 3.279826 | 完全不借历史信息，只用弱先验 |
| Rule lambda | 3.268898 | 使用人工规则分配 lambda |
| Learned lambda | 3.230277 | 使用训练出的 lambda model 分配 lambda |

learned lambda 相对 rule lambda 的差值：

```text
learned_minus_rule_mean_nll
= learned_NLL - rule_NLL
= 3.230277 - 3.268898
= -0.038621
```

因为 NLL 越低越好，所以 `-0.038621` 表示 learned lambda 在 held-out eval set 上优于 rule lambda。换成 predictive likelihood ratio：

```text
likelihood_ratio_learned_vs_rule
= exp(rule_NLL - learned_NLL)
= exp(3.268898 - 3.230277)
= 1.039377
```

也就是说，learned lambda 对 held-out ORR count 的平均 predictive likelihood 约为 rule lambda 的 1.039 倍。这是一个真实、稳定、但幅度不大的提升。

Bootstrap 95% CI：

| Metric | Point | 2.5% | 97.5% |
|---|---:|---:|---:|
| learned_minus_rule_nll | -0.038621 | -0.062998 | -0.013601 |
| learned_mean_nll | 3.230277 | 3.089978 | 3.395883 |
| rule_mean_nll | 3.268898 | 3.124611 | 3.436093 |
| learned_mae | 0.287448 | 0.268567 | 0.305696 |
| learned_rmse | 0.328913 | 0.308709 | 0.347049 |
| learned_correlation | 0.026756 | -0.095814 | 0.165517 |

ORR 点预测指标：

| Split | Model | MAE | RMSE | Correlation | Mean predicted | Mean observed |
|---|---|---:|---:|---:|---:|---:|
| train | weak | 0.327906 | 0.367163 | NA | 0.500000 | 0.296147 |
| train | rule | 0.287076 | 0.334646 | 0.086132 | 0.401419 | 0.296147 |
| train | learned | 0.287783 | 0.332133 | 0.087197 | 0.408384 | 0.296147 |
| eval | weak | 0.316971 | 0.357204 | NA | 0.500000 | 0.311297 |
| eval | rule | 0.287104 | 0.331629 | 0.030216 | 0.402109 | 0.311297 |
| eval | learned | 0.287448 | 0.328913 | 0.026756 | 0.408753 | 0.311297 |

客观评价：

- 从 NLL 看，learned lambda 比 rule lambda 更好，而且 bootstrap CI 对 `learned_minus_rule_nll` 完全小于 0，说明这个 NLL 改善不是单次 split 的偶然噪声。
- 从 ORR 点预测看，learned lambda 的 RMSE 比 rule lambda 略好：0.328913 vs 0.331629。
- learned lambda 的 MAE 比 rule lambda 略差：0.287448 vs 0.287104，差异非常小。
- learned predicted ORR 与 observed ORR 的相关系数只有 0.026756，bootstrap CI 包含 0。这说明模型还不能可靠地区分“哪个 trial ORR 更高、哪个更低”。
- 因此，本次结果应解释为：lambda model 学到了一点有用的 mixture weighting calibration，能稳定降低 predictive NLL；但它还不是一个强 ORR regression/prediction model，更不能替代专家判断或正式 statistical borrowing justification。

为什么这里没有 AUC、precision、accuracy：

- AUC、precision、accuracy 是二分类问题的常见指标，例如判断 `response=yes/no` 或 `trial success/failure`。
- 本项目的 retrospective lambda training 不是在判断一个 trial 属于正类还是负类，而是在预测 held-out ORR count：`y_q responders out of n_q patients`。
- 因此正确主指标是 beta-binomial predictive NLL，因为它直接衡量 mixture prior 对完整 count/denominator outcome 的概率预测质量。
- ORR 点预测可以辅助看 MAE、RMSE、correlation、scatter plot、calibration plot。
- 如果以后定义一个二分类标签，例如 `ORR >= 30%` 或 `trial met endpoint = yes/no`，那时才适合增加 AUC、precision、recall、accuracy。

结果图表和表格位置：

- Train/eval NLL curve：`artifacts/retrospective_lambda_oncology_orr_all/figures/train_eval_nll_curve.svg`
- Training objective curve：`artifacts/retrospective_lambda_oncology_orr_all/figures/training_objective_curve.svg`
- Weak vs rule vs learned NLL bar chart：`artifacts/retrospective_lambda_oncology_orr_all/figures/evaluation_nll_comparison.svg`
- Predicted ORR vs observed ORR scatter：`artifacts/retrospective_lambda_oncology_orr_all/figures/predicted_vs_observed_orr_scatter.svg`
- Calibration plot：`artifacts/retrospective_lambda_oncology_orr_all/figures/calibration_plot.svg`
- MAE/RMSE/correlation table：`artifacts/retrospective_lambda_oncology_orr_all/lambda_rate_metrics.csv`
- Bootstrap CI table：`artifacts/retrospective_lambda_oncology_orr_all/lambda_bootstrap_ci.csv`
- Full evaluation JSON：`artifacts/retrospective_lambda_oncology_orr_all/lambda_evaluation.json`
- Markdown result report：`artifacts/retrospective_lambda_oncology_orr_all/retrospective_lambda_training_results.md`

## 1. 这个训练到底在学什么

本项目的 Bayesian historical borrowing 现在有一个 mixture prior：

```text
p(theta) = lambda_0 * p_0(theta) + sum_i lambda_i * p_i(theta)
```

其中：

- `p_0(theta)` 是弱先验，也就是不从历史 trial 借信息的 component。
- `p_i(theta)` 是第 `i` 个历史 trial 形成的 beta prior component。
- `lambda_0` 是弱先验 component 的 prior weight。
- `lambda_i` 是第 `i` 个历史 trial component 的 prior weight。

这里要训练的不是 endpoint rate 本身，而是一个函数：

```text
lambda_i = model(candidate_i, query_trial)
```

更具体地说，模型看到每个候选历史 trial 的 9 个特征：

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

然后输出一个 score：

```text
z_i = f_theta(x_i)
```

这个 score 再被转成 mixture weight `lambda_i`。训练目标是：如果把一个已经完成的 trial 当作 pseudo-query，并隐藏它的真实结果，那么模型给历史 trial 分配出来的 `lambda_i`，应该能让 mixture prior 更好地预测这个 pseudo-query 的 held-out outcome。

一句话概括：

```text
retrospective lambda training = 用历史 completed trials 做 pseudo-query，
训练一个 lambda_i scorer，使 mixture prior 对 held-out query outcome 的 predictive likelihood 更高。
```

## 2. 为什么可以不用专家标签

理想情况下，`lambda_i` 可以由专家给标签，例如：

```text
这个历史 trial 非常可借用：lambda_i 高
这个历史 trial 不可借用：lambda_i 低
```

但当前项目没有专家评审标签。因此训练信号来自 retrospective prediction：

1. 选一个已经完成的 trial `q`。
2. 把 `q` 的结果隐藏起来，只保留 disease、intervention、endpoint label、design、eligibility 等检索信息。
3. 用隐藏结果后的 `q` 去检索历史 trials。
4. Stage 2 rerank 得到 top candidates。
5. 每个 candidate 形成一个 beta prior component。
6. 模型预测每个 candidate 的 `lambda_i`。
7. 用这些 `lambda_i` 组成 mixture prior。
8. 看这个 mixture prior 对 `q` 的真实 held-out outcome 给出多高 predictive probability。

如果 probability 越高，negative log likelihood 越低，模型越好。

这是一种 no-expert predictive calibration，不是 clinical/statistical expert validation。它可以学习哪些相似性特征更有预测价值，但不能证明某个历史 trial 在真实注册分析中一定适合借用。

## 3. 数据生成总流程

### 3.1 completed trial 作为 pseudo-query

对每个 completed trial `q`，运行 search 时必须使用：

```bash
python3 pipeline/oncology_trial_similarity_pipeline.py search \
  --query-json /path/to/completed_trial.json \
  --index-dir artifacts/oncology_trial_similarity_clinicalbert \
  --top-k 100 \
  --rerank \
  --rerank-top-n 10 \
  --hide-query-outcomes-for-retrieval \
  --output artifacts/pseudo_query_results/NCTxxxx.json
```

关键参数是：

```text
--hide-query-outcomes-for-retrieval
```

它会做两件事：

1. `query_summary` 去掉 posted outcome values，防止检索、rerank、feature construction 看到真实结果。
2. 真实结果另存到 `heldout_query_outcomes`，只用于 post-retrieval predictive loss、evaluation 和 analysis。

输出 JSON 里必须有：

```json
{
  "retrospective_leakage_control": {
    "query_outcomes_hidden_from_retrieval": true,
    "heldout_query_outcomes_for_post_retrieval_analysis": true
  },
  "heldout_query_outcomes": {
    "...": "..."
  }
}
```

训练脚本默认会拒绝没有这些 leakage-control metadata 的 pipeline result。

### 3.2 为什么必须隐藏 query outcome

如果不隐藏结果，会发生 leakage：

```text
query 的真实 ORR/count/denominator 出现在 query text 或 endpoint fields 中
=> retrieval 可能直接根据结果相似性找到 candidate
=> rerank/features 可能间接使用真实 outcome
=> lambda model 在训练时偷看答案
=> retrospective evaluation 虚高
```

正确流程是：

```text
query clinical/design text -> retrieval/rerank/feature construction
query held-out outcome -> only post-retrieval loss/evaluation/analysis
```

## 4. 每个候选 trial 如何变成 beta component

假设 pseudo-query `q` 检索并 rerank 后得到 top 10 historical candidates。对每个 candidate `i`，pipeline 先找目标 endpoint，例如 `ORR`。

从 candidate 的 `borrowable_quantities` 中抽取：

```text
y_i = historical treatment-arm response count
n_i = historical treatment-arm denominator
r_i = y_i / n_i
```

如果 candidate 没有可用的 matched endpoint count/denominator，就不会进入 mixture prior components。

每个 candidate 还有一个 borrowing discount：

```text
a_i = suggested_borrowing_discount_i
```

代码中把它限制在：

```text
0 <= a_i <= 1
```

然后历史 trial `i` 形成一个 beta prior component：

```text
p_i(theta) = Beta(alpha_i, beta_i)
```

其中：

```text
alpha_i = 1 + a_i * y_i
beta_i  = 1 + a_i * (n_i - y_i)
```

解释：

- `Beta(1, 1)` 是均匀弱先验。
- `a_i * y_i` 是折扣后的成功数。
- `a_i * (n_i - y_i)` 是折扣后的失败数。
- `a_i` 越小，历史 trial 的 effective sample size 越小。

这一步对应 `pipeline/mixture_prior.py` 中的 `components_from_reranked_rows()`。

## 5. rule lambda 如何得到

训练模型之前，pipeline 已经有一套 rule-based `lambda_rule`。它不是最终必须使用的答案，而是给训练提供一个温和的 regularization target。

对每个 candidate，先算一个 raw rule weight：

```text
w_i^rule = gate_i * a_i * max(s_i, 0) * log(1 + n_i)
```

其中：

- `s_i = overall_similarity_score_i / 100`
- `a_i = suggested_borrowing_discount_i`
- `n_i = candidate endpoint denominator`
- `gate_i` 是 conservative borrowing gate

当前 gate 规则：

```text
if endpoint_match < 1.5 or result_quality <= 0:
    gate_i = 0
else:
    gate_i = 1
    if disease_match < 1.5:
        gate_i *= 0.2
    elif disease_match < 2.5:
        gate_i *= 0.6
    if any "Low ..." red flag exists:
        gate_i *= 0.5
```

然后保留一个弱先验预算 `lambda_0`，默认：

```text
lambda_0 = 0.2
```

候选历史 trials 总共只能分配：

```text
1 - lambda_0 = 0.8
```

如果至少一个 `w_i^rule > 0`：

```text
lambda_i^rule = (1 - lambda_0) * w_i^rule / sum_j w_j^rule
```

如果所有 rule weights 都是 0：

```text
lambda_0 = 1
lambda_i^rule = 0
```

## 6. 9-feature x_i 怎么得到

训练样本里每个 candidate `i` 有一行 feature vector：

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

### 6.1 `s_i`

```text
s_i = overall_similarity_score_i / 100
```

`overall_similarity_score_i` 是 Stage 2 reranker 输出的综合相似分数，范围通常是 0 到 100。除以 100 后变成 0 到 1 左右，方便神经网络训练。

### 6.2 `disease_match_i`

```text
disease_match_i = disease score / 5
```

优先读取：

```text
disease_population_match
```

如果没有，则读取旧字段：

```text
disease_biology_match
```

Stage 2 dimension score 是 0 到 5，所以除以 5 归一化。

### 6.3 `regimen_match_i`

```text
regimen_match_i = treatment_regimen_match / 5
```

表示 experimental regimen / backbone / drug class 是否接近。

### 6.4 `endpoint_match_i`

```text
endpoint_match_i = endpoint_estimand_match / 5
```

表示 endpoint family、estimand、时间窗、measurement 是否可比。

### 6.5 `followup_match_i`

```text
followup_match_i = followup score / 5
```

优先读取：

```text
safety_and_followup_relevance
```

如果没有，则读取：

```text
outcome_assessment_followup
```

### 6.6 `eligibility_match_i`

```text
eligibility_match_i = eligibility_criteria_overlap / 5
```

表示 inclusion/exclusion criteria、population restrictions 是否相近。

### 6.7 `result_quality_i`

```text
result_quality_i = result_usability / 5
```

表示 candidate 有没有 posted results、是否有 arm-level count/denominator、endpoint 数据是否可用。

### 6.8 `-redflag_severity_i`

先计算 red flag severity：

```text
severity_i = min(raw_i / 3, 1)
```

其中当前 raw score 规则：

```text
low disease or low endpoint                  -> +1.0
no primary endpoint-family overlap           -> +1.0
no normalized regimen-backbone overlap       -> +0.8
no posted results or no count/denominator    -> +0.8
other red flag                               -> +0.25
```

输入特征使用负数：

```text
negative_redflag_severity_i = -severity_i
```

为什么取负数：神经网络一般会把较大的输入理解成“更多证据”。red flag 越严重越不应该借用，所以让它变成负方向的 signal。

### 6.9 `log(n_i)`

当前代码实际使用：

```text
log_n_i = log(1 + n_i)
```

即 `math.log1p(n_i)`。

为什么不用原始 `n_i`：

- 样本量可能跨度很大。
- `n=20` 到 `n=200` 的差异很重要，但不应该让大样本完全支配模型。
- `log(1+n)` 把样本量变成递增但边际递减的证据强度。

## 7. Lambda scorer 结构

默认 lambda model 是一个很小的 MLP：

```text
z_i = f_theta(x_i)
```

具体结构：

```text
h_i = ReLU(W_1 x_i + b_1)
z_i = W_2 h_i + b_2
```

其中：

- `x_i` 是 9 维 feature vector。
- `hidden_dim` 默认 16。
- `z_i` 是 candidate `i` 的 raw score。

代码对应：

```text
Linear(input_dim=9, hidden_dim)
ReLU()
Linear(hidden_dim, 1)
```

这里 `z_i` 本身还不是 `lambda_i`。它只是未归一化分数。

现在训练代码支持四种 `--model-type`：

| model_type | 公式直觉 | 主要用途 |
|---|---|---|
| `mlp` | `z_i = MLP(x_i)` | 原始 baseline，逐个 candidate 独立打分 |
| `monotonic_softmax` | `z_i = b + sum_k softplus(w_k) x_{ik}` | 可解释 baseline，保证每个特征方向不反常 |
| `deepsets` | `z_i = rho(phi(x_i), mean_j phi(x_j))` | set-aware 模型，让每个 candidate score 看到整个 top-K candidate set |
| `two_head_deepsets` | `z_i` 用于 lambda，另一个 head 输出 `a_i` | 同时学习 mixture weight `lambda_i` 和 effective sample size discount `a_i` |

为什么加 `deepsets`：`lambda_i` 不是单独判断一个 trial 好不好，而是在一组 candidates 里分配总预算。DeepSets 用 `mean_j phi(x_j)` 表示整个候选集合的上下文，因此能学习“这个 candidate 相对其他候选是否更值得借”。

为什么加 `monotonic_softmax`：它是一个强可解释 baseline。因为 `negative_redflag_severity_i` 已经是负数，所以所有权重约束为非负时，red flag 越严重 score 越低；endpoint/result/disease 等特征越高 score 越高。

为什么加 `two_head_deepsets`：原来的 `a_i` 来自规则折扣，只训练 `lambda_i`。two-head 模型可以同时学习：

```text
lambda_i = component 在 mixture prior 里的 prior mass
a_i      = component 内部借入多少 effective sample size
```

训练时如果模型有 `predict_discount(x_i)`，就用模型预测的 `a_i` 重新构造：

```text
alpha_i = 1 + a_i * y_i
beta_i  = 1 + a_i * (n_i - y_i)
```

如果模型没有 discount head，则沿用 component 原本的 `alpha_i, beta_i`，保证旧 MLP artifact 向后兼容。

在 pipeline 推理阶段，如果 artifact 的 `model_type` 是 `two_head_deepsets`，输出的 mixture prior component 会额外包含：

```text
discount_rule    原 rule discount
discount_model   模型预测的 discount
discount_active  当前实际使用的 discount
alpha_rule       rule discount 对应的原 alpha
beta_rule        rule discount 对应的原 beta
alpha, beta      使用 discount_active 重新构造后的 active beta component
```

这样报告里可以同时审查“规则本来想借多少”和“模型最后决定借多少”。

## 8. score z_i 如何变成 lambda_i

模型输出每个 candidate 的 score：

```text
z_i = f_theta(x_i)
```

然后结合 gate：

```text
u_i = exp(z_i) * gate_i
```

等价地，在 log space 中：

```text
log u_i = z_i + log(gate_i)
```

如果 `gate_i = 0`，该 candidate 被 mask 掉，不能获得 lambda。

若至少一个 candidate 有正 gate：

```text
lambda_i(theta) = (1 - lambda_0) * u_i / sum_j u_j
```

也就是：

```text
lambda_i(theta)
= (1 - lambda_0) * softmax_i(z_i + log gate_i)
```

并且：

```text
sum_i lambda_i(theta) = 1 - lambda_0
lambda_0 + sum_i lambda_i(theta) = 1
```

如果所有 `gate_i = 0`：

```text
lambda_0 = 1
lambda_i = 0
```

这表示完全不从历史 trial 借用。

## 9. Beta-binomial predictive probability

对一个 pseudo-query，它的 held-out endpoint outcome 是：

```text
y_q = query treatment-arm response count
n_q = query treatment-arm denominator
```

如果 prior 是：

```text
theta ~ Beta(alpha, beta)
```

并且：

```text
y_q | theta ~ Binomial(n_q, theta)
```

把 `theta` 积分掉后，得到 beta-binomial predictive probability：

```text
P(y_q | n_q, alpha, beta)
= C(n_q, y_q) * B(y_q + alpha, n_q - y_q + beta) / B(alpha, beta)
```

其中：

```text
C(n, y) = n! / (y! (n-y)!)
B(a, b) = Gamma(a) Gamma(b) / Gamma(a+b)
```

代码使用 log space 来避免数值下溢：

```text
log P(y_q | n_q, alpha, beta)
= log C(n_q, y_q)
  + log B(y_q + alpha, n_q - y_q + beta)
  - log B(alpha, beta)
```

弱先验 component 使用：

```text
alpha_0 = 1
beta_0 = 1
```

所以：

```text
P_0 = P(y_q | n_q, 1, 1)
```

第 `i` 个历史 trial component 使用：

```text
P_i = P(y_q | n_q, alpha_i, beta_i)
```

## 10. Mixture predictive probability

给定模型预测出的 mixture weights：

```text
lambda_0, lambda_1, ..., lambda_K
```

整个 mixture prior 对 held-out outcome 的 predictive probability 是：

```text
P_mix(y_q | n_q)
= lambda_0 * P_0
  + sum_i lambda_i * P_i
```

展开就是：

```text
P_mix(y_q | n_q)
= lambda_0 * P(y_q | n_q, 1, 1)
  + sum_i lambda_i(theta) * P(y_q | n_q, alpha_i, beta_i)
```

训练的核心目标就是让这个 `P_mix` 尽量大。

## 10.1 Stage 3: SAM / robust prior-data conflict adapter

Stage 2 学到的是 pre-trial borrowing：只看 query 设计信息和历史 candidates，决定历史 prior 怎么组成。Stage 3 处理另一个问题：当当前 trial 结果出来后，如果它和历史 prior 明显冲突，就动态减少 historical borrowing。

当前实现采用保守的 SAM-style conflict brake。先计算 weak prior predictive：

```text
P_weak = P(y_q | n_q, 1, 1)
```

再把 historical candidates 的 active lambdas 归一化为 candidate-only mixture：

```text
tilde_lambda_i = lambda_i_active / sum_j lambda_j_active
```

得到 historical informative predictive：

```text
P_hist = sum_i tilde_lambda_i * P(y_q | n_q, alpha_i, beta_i)
```

然后计算 predictive probability ratio：

```text
R = P_hist / P_weak
```

如果 historical prior 对当前数据的解释能力不低于 weak prior：

```text
R >= 1  =>  borrowing_multiplier = 1
```

如果 historical prior 比 weak prior 更不支持当前数据：

```text
R < 1   =>  borrowing_multiplier = R^temperature
```

默认 `temperature = 1`。最终 active candidate mass 被下调：

```text
lambda_i_sam = lambda_i_pre_sam * borrowing_multiplier
lambda_0_sam = 1 - sum_i lambda_i_sam
```

直觉：

- 如果历史 prior 和当前数据一致，不额外增加 borrowing，只保留 Stage 2 的 borrowing。
- 如果历史 prior 和当前数据冲突，就把 candidate lambdas 乘以一个 0 到 1 之间的 multiplier。
- 减少掉的 historical mass 自动回到 weak prior `lambda_0_sam`。

pipeline 现在支持：

```text
--mixture-prior-mode sam
--mixture-prior-mode retrospective_calibrated_sam
```

`sam` 表示 rule lambda + SAM conflict adapter。`retrospective_calibrated_sam` 表示 learned lambda / learned discount + SAM conflict adapter。

## 11. 训练 loss

### 11.1 主 loss：negative log likelihood

因为概率越大越好，所以训练最小化：

```text
L_pred(theta)
= -log P_mix(y_q | n_q)
```

对多个 pseudo-query examples：

```text
L_pred_mean(theta)
= (1 / M) * sum_m L_pred_m(theta)
```

### 11.2 KL regularization toward rule lambda

当前训练还可以使用 rule lambda 作为弱监督 regularizer。

先把 rule lambda 归一到 candidate budget：

```text
lambda_i^rule_norm
= (1 - lambda_0) * lambda_i^rule / sum_j lambda_j^rule
```

如果 rule lambda 全部为 0，就跳过 KL。

KL 项：

```text
KL(lambda^rule || lambda^model)
= sum_i lambda_i^rule_norm
  * [log(lambda_i^rule_norm) - log(lambda_i^model)]
```

训练 loss 加上：

```text
rho * KL(lambda^rule || lambda^model)
```

当前默认：

```text
rho = 0.1
```

直觉：

- predictive likelihood 是主要训练信号。
- rule lambda 是稳定器，防止小数据时模型完全偏离临床规则。
- KL 权重较小，所以不是强迫复制 rule。

### 11.3 Listwise allocation auxiliary loss

优化后的训练代码新增了一个可选的 listwise allocation auxiliary loss。它解决的问题是：主 NLL 只告诉模型“整个 mixture prior 对 held-out outcome 预测得好不好”，但不直接告诉模型 top10 candidates 内部谁相对更值得借。listwise loss 会给候选集合内部提供一个更清楚的相对分配信号。

对每个 candidate，先计算它自己单独解释 query held-out outcome 的 beta-binomial log predictive：

```text
ell_i = log P(y_q | n_q, alpha_i, beta_i)
```

弱先验的 log predictive 是：

```text
ell_0 = log P(y_q | n_q, 1, 1)
```

candidate 的相对解释力定义为：

```text
r_i = ell_i - ell_0
```

如果 `r_i > 0`，说明 candidate `i` 单独作为 historical prior component 比 weak prior 更能解释这个 pseudo-query 的 held-out outcome；如果 `r_i < 0`，说明它比 weak prior 更差。

然后把这些相对解释力变成 soft target distribution：

```text
t_i = exp(r_i / T) / sum_j exp(r_j / T)
```

其中 `T` 是 temperature：

- `T` 小：target 更尖锐，最能解释 held-out outcome 的 candidate 权重更集中。
- `T` 大：target 更平滑，多个 candidate 可以共享 target mass。

模型自己的 candidate-only distribution 是：

```text
q_i(theta) = softmax_i(z_i + log gate_i)
```

注意这里不用乘 `(1 - lambda_0)`，因为 listwise loss 只关心候选之间的相对分配，不关心 weak prior 占多少总预算。

listwise cross-entropy 是：

```text
L_listwise(theta)
= - sum_i t_i log q_i(theta)
```

这个辅助项不是专家标签，也不是泄漏到 retrieval。它只在 retrospective training 阶段使用 held-out outcome 构造训练信号；正式 query inference 时不会使用 query outcome 来生成 `lambda_i`。

当前实现中，`--listwise-eta` 默认是 `0`，也就是不启用这个辅助项。它保留为可选敏感性实验项。本次 full ORR run 中：

| 配置 | Eval learned NLL | Learned - rule NLL |
|---|---:|---:|
| pure `two_head_deepsets` | 3.225659 | -0.043240 |
| `two_head_deepsets + listwise_eta=0.05` | 3.227351 | -0.041547 |

因此当前推荐主模型仍是 pure `two_head_deepsets`；listwise 暂不作为默认配置。

### 11.4 ESS penalty

模型如果把太多权重给大样本历史 trials，可能形成过大的 effective sample size。当前有一个轻量 penalty：

```text
ESS_model = sum_i lambda_i * a_i * n_i
```

如果超过 cap：

```text
L_ESS = 1e-4 * max(ESS_model - ESS_cap, 0)^2
```

当前默认：

```text
ESS_cap = 100
```

最终训练 loss：

```text
L(theta)
= L_pred(theta)
  + rho * KL(lambda^rule || lambda^model)
  + eta * L_listwise(theta)
  + L_ESS
```

注意：evaluation 时的 learned lambda NLL 不包含 KL 和 ESS penalty，只看纯 predictive NLL。
如果训练使用了 listwise auxiliary loss，evaluation 时同样不把 `eta * L_listwise` 加进 held-out NLL；评估仍然只比较 mixture prior 对 held-out count 的 pure beta-binomial predictive NLL。

## 12. 训练算法

训练使用 Adam optimizer。

对每个 epoch：

```text
for epoch in 1..E:
    optimizer.zero_grad()
    losses = []
    for each training example m:
        compute x_i for all candidates
        z_i = f_theta(x_i)
        lambda_i = softmax_with_gate(z_i, gate_i)
        compute beta-binomial predictive probability
        compute L_m(theta)
        append L_m
    mean_loss = mean(losses)
    backpropagate mean_loss
    optimizer.step()
```

命令示例：

```bash
python3 pipeline/train_retrospective_lambda_model.py \
  --pipeline-results-jsonl artifacts/retrospective/pipeline_results.jsonl \
  --output-json artifacts/retrospective/lambda_training_summary.json \
  --model-output artifacts/retrospective/lambda_model.pt \
  --epochs 100 \
  --learning-rate 0.01 \
  --hidden-dim 16 \
  --model-type two_head_deepsets
```

输出 summary 包含：

```json
{
  "epochs": 100,
  "final_loss": "...",
  "loss_history": ["..."],
  "input_dim": 9,
  "hidden_dim": 16,
  "model_type": "two_head_deepsets",
  "listwise_eta": 0.0,
  "listwise_temperature": 1.0,
  "model_output": "artifacts/retrospective/lambda_model.pt"
}
```

模型 artifact 保存：

```text
state_dict
input_dim
hidden_dim
feature_names
lambda0
model_type
```

加载时会检查：

```text
feature_names == LAMBDA_FEATURE_NAMES
input_dim == 9
```

这样可以避免以后 feature 顺序改了，却误用旧模型。

## 13. Retrospective evaluation

训练完成后，需要在 held-out pseudo-query examples 上评估，而不是只看训练 loss。

evaluation 脚本做 deterministic split：

```text
all examples -> train examples + eval examples
```

给定：

```text
train_fraction = 0.8
seed = 20260603
```

它会：

1. 用 train split 训练 lambda model。
2. 在 eval split 上计算三个 NLL：
   - weak-only
   - rule lambda
   - learned lambda

命令示例：

```bash
python3 pipeline/evaluate_retrospective_lambda_model.py \
  --pipeline-results-jsonl artifacts/retrospective/pipeline_results.jsonl \
  --output-json artifacts/retrospective/lambda_evaluation.json \
  --train-fraction 0.8 \
  --seed 20260603 \
  --epochs 100 \
  --learning-rate 0.01 \
  --hidden-dim 16
```

输出核心字段：

```json
{
  "evaluation_target": "retrospective_predictive_negative_log_likelihood",
  "outcome_usage": "held_out_query_outcomes_for_post_retrieval_predictive_evaluation_and_analysis",
  "metrics": {
    "weak_only_mean_nll": "...",
    "rule_lambda_mean_nll": "...",
    "learned_lambda_mean_nll": "...",
    "learned_minus_rule_mean_nll": "..."
  },
  "leakage_control_assumption": "Query outcomes must be hidden from retrieval/reranking/feature construction/model selection and reserved for post-retrieval predictive loss/evaluation/analysis."
}
```

解释：

- `weak_only_mean_nll`：完全不借历史信息。
- `rule_lambda_mean_nll`：使用 rule-based mixture weights。
- `learned_lambda_mean_nll`：使用神经网络学出来的 mixture weights。
- `learned_minus_rule_mean_nll < 0`：learned lambda 在 held-out pseudo-query 上比 rule lambda 更好。
- `learned_minus_rule_mean_nll > 0`：learned lambda 反而更差，不能说模型优于规则。

## 14. 训练好的模型如何用于新 query

搜索新 query 时：

```bash
python3 pipeline/oncology_trial_similarity_pipeline.py search \
  --query-json /path/to/new_query.json \
  --index-dir artifacts/oncology_trial_similarity_clinicalbert \
  --top-k 100 \
  --rerank \
  --rerank-top-n 10 \
  --mixture-prior-mode retrospective_calibrated \
  --lambda-model-path artifacts/retrospective/lambda_model.pt \
  --output artifacts/new_query_result.json
```

流程：

1. 新 query 没有可用 outcome，所以不会有 held-out outcome。
2. Stage 1 retrieval 找 candidates。
3. Stage 2 rerank 生成 top candidates、dimension scores、red flags、borrowable quantities。
4. `components_from_reranked_rows()` 生成 beta components 和 rule lambda。
5. lambda model 对每个 component 的 `x_i` 输出 raw score。
6. `apply_model_lambdas()` 把 raw scores normalize 成：

```text
lambda_model
lambda_active
```

同时保留：

```text
lambda_rule
```

这样报告中可以比较：

```text
rule-based prior mass vs retrospective-calibrated prior mass
```

## 15. 当前方法和 expert review 的关系

这个训练能回答的问题：

```text
在过去 completed trials 上，哪些 candidate features 更能预测 held-out endpoint outcome？
```

它不能单独回答：

```text
这个历史 trial 是否在真实 confirmatory analysis 中临床上可借用？
```

原因：

- retrospective prediction 仍可能受到 dataset shift 影响。
- top candidates 来自当前 retrieval/rerank pipeline，pipeline 偏差会传递到训练。
- 没有专家标签时，模型无法学习专家认为重要但 predictive loss 暂时没捕捉到的临床因素。
- endpoint outcome 的相似不等价于 trial exchangeability。

因此 manuscript 或正式分析中应表述为：

```text
retrospective predictive calibration / sensitivity analysis
```

不应表述为：

```text
expert-validated borrowing weights
```

## 16. 最小可运行清单

### Step 1: 生成 leakage-safe pseudo-query results

```bash
python3 pipeline/oncology_trial_similarity_pipeline.py search \
  --query-json completed_trial.json \
  --index-dir artifacts/oncology_trial_similarity_clinicalbert \
  --top-k 100 \
  --rerank \
  --rerank-top-n 10 \
  --hide-query-outcomes-for-retrieval \
  --output artifacts/retrospective/NCTxxxx.json
```

### Step 2: 合并 JSONL

每行一个 search result：

```text
artifacts/retrospective/pipeline_results.jsonl
```

### Step 3: 训练模型

```bash
python3 pipeline/train_retrospective_lambda_model.py \
  --pipeline-results-jsonl artifacts/retrospective/pipeline_results.jsonl \
  --output-json artifacts/retrospective/lambda_training_summary.json \
  --model-output artifacts/retrospective/lambda_model.pt \
  --epochs 100 \
  --learning-rate 0.01 \
  --hidden-dim 16
```

### Step 4: 做 retrospective evaluation

```bash
python3 pipeline/evaluate_retrospective_lambda_model.py \
  --pipeline-results-jsonl artifacts/retrospective/pipeline_results.jsonl \
  --output-json artifacts/retrospective/lambda_evaluation.json \
  --train-fraction 0.8 \
  --seed 20260603
```

### Step 5: 在新 query 中使用

```bash
python3 pipeline/oncology_trial_similarity_pipeline.py search \
  --query-json new_query.json \
  --index-dir artifacts/oncology_trial_similarity_clinicalbert \
  --top-k 100 \
  --rerank \
  --rerank-top-n 10 \
  --mixture-prior-mode retrospective_calibrated \
  --lambda-model-path artifacts/retrospective/lambda_model.pt \
  --output artifacts/new_query_result.json
```

## 17. 与 Trial2Vec / SECRET 的关系

Retrospective lambda training 不替代 Stage 1 retrieval。它发生在 Stage 1 和 Stage 2 之后。

```text
Trial2Vec / SECRET / ClinicalBERT
  -> 找候选历史 trials
  -> Stage 2 structured rerank
  -> mixture components
  -> retrospective lambda model 学每个 component 的 lambda_i
```

Trial2Vec 和 SECRET 影响的是：

```text
哪些 historical candidates 被找到
```

Retrospective lambda training 影响的是：

```text
找到之后，每个 candidate 在 mixture prior 里占多少 prior mass
```

所以二者是互补关系，不是替代关系。

## 18. 参考文献和方法来源

- Trial2Vec: Zero-Shot Clinical Trial Document Similarity Search using Self-Supervision. [arXiv:2206.14719](https://arxiv.org/abs/2206.14719), [ACL Anthology PDF](https://aclanthology.org/2022.findings-emnlp.476.pdf).
- SECRET: Semi-supervised Clinical Trial Document Similarity Search. [ACL Anthology PDF](https://aclanthology.org/2025.acl-long.264.pdf).
- Hobbs, Sargent, Carlin. Commensurate Priors for Incorporating Historical Information in Clinical Trials Using General and Generalized Linear Models. [PMC4007051](https://pmc.ncbi.nlm.nih.gov/articles/PMC4007051/).
- Hobbs, Carlin, Mandrekar. Hierarchical Commensurate and Power Prior Models for Adaptive Incorporation of Historical Information in Clinical Trials. [PMC3134568](https://pmc.ncbi.nlm.nih.gov/articles/PMC3134568/).
- Schmidli-style robust mixture prior and self-adapting mixture prior discussion: SAM: Self-adapting mixture prior to dynamically borrow information from historical data in clinical trials. [PMC10842647](https://pmc.ncbi.nlm.nih.gov/articles/PMC10842647/).
- Background on prior-data conflict and robust mixture priors: A decision-theoretic approach to Bayesian clinical trial design and evaluation of robustness to prior-data conflict. [PMC9118338](https://pmc.ncbi.nlm.nih.gov/articles/PMC9118338/).
