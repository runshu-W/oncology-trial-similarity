# 第十一轮逐条审稿回应

**稿件**：Retrospective predictive calibration of oncology trial similarity for Bayesian historical borrowing
**本文档**：对第十一轮意见（Minor Revision）的逐条回应。感谢您明确撤回上一轮基于确定性阈值分析的判断，并确认两项核心问题已实质解决；本轮按您的意见收窄推断措辞、澄清比较对象，并完成文档同步。

---

## 1. "Exact size matching" 的适用范围

**接受，三点全改。**

- **措辞**：全文（§3.1、§4.5、摘要、术语表、Table 8 caption、S1、S2、Figure S1 caption）改为您给出的两层表述：**训练半批经验 size 精确匹配（randomized boundary），评估半批平均 size 近似匹配**；"exactly size-matched / exact expected-size matching" 全部删除（grep = 0），统一改用 "size-standardized"。§4.5 明写：期望形式消除的是随机化变量的额外噪声，不消除临界值估计误差与有限模拟误差。
- **评估半批 size 差分布**（新计算，`round11_inference_scope.json`；同种子 20260908，六个 run 的估计与 round-10 artifact 逐项断言一致）：canonical@0.02513 的逐 split 评估半批 size 差（cap50 − no-borrowing）：均值 −7×10⁻⁶，central 95% range **[−0.000215, +0.000207]**，最大绝对值 **0.000577**；全部六个 run 的最大绝对差 ≤ 0.00075。逐 split 差值数组已存入 artifact；正文 §4.5 与 S2 报告分布而非仅两个均值。
- **比较对象**：Table 9 caption 与 Table 8 caption、§4.5 均明写——**Table 9 评估的是原先冻结的确定性决策规则；随机化分析是 capped design 的 size-standardized comparison，两者是不同 estimand，+0.0025 不归属于该冻结规则**。

## 2. "gain is real" 过强；不确定性来源需分别交代

**接受，全部条件化；不新增大规模模拟（按您的许可）。**

- "real" → **"positive in this conditional analysis"**；"robust under repetition" → **"stable across the examined splits"**（§4.5、讨论、结论、S1、S2 全部同步）。摘要与结论明写 "a conditional, batch-specific comparison, not a confirmed population claim" / "conditional on the simulation batch and estimated mixing"。
- **MCSE**：已为随机化 estimator **重新计算**——全批 randomized plug-in 在共同目标下为 **+0.00252**，within-replicate paired MCSE = **0.00070**（评估 size 0.02513/0.02513）；正文标注 "recomputed for the randomized estimator"。
- **Mixing bootstrap**：原 [+0.0006,+0.0044] 来自插值全批分析——现**在随机化匹配、固定共同目标下重跑**（500 次重采样，RNG 20260905）：mean +0.00244，95% 区间 **[+0.00052, +0.00449]**，**499/500 为正**；正文引用新区间并注明取代旧区间。
- **来源清单**：§4.5 与 S2 明写三类不确定性（split/阈值估计；有限模拟；mixing 权重估计）**分别考察、未联合传播**，未生成独立重复批次，deconvolution 正态形只在 round-5 敏感性族内考察——"a population-level or cross-batch claim is not made"。
- 修稿历史退出正文主线：摘要与结论不再叙述 "−0.0007 → +0.0025" 的过程；§4.5 保留一句确定性版本近零 + 指向 S2 的诊断（编辑性意见一并处理）。

## 3. "matches the hand rule" 降级为描述性

**接受，采纳您的句子。** §4.4 capped 比较改为 "**the two selection rules yielded similar observed mean benefits; neither superiority nor equivalence was established**"（uncapped 句同样降级："neither superiority nor equivalence of the two selection devices is established"），并补报两个选择集的重叠（atom n=281 与 I≥8 n=291 共享 **169** 个查询）。新增粗分组限定："post-cap mass 在 **36%** 的 forward queries 上等于 0.5，final design 的 triage 证据支持 **coarse risk grouping**，不是 capped 集合内部可靠的逐查询排序"；结论 "demonstrated per-query value is triage" → "**demonstrated value is graded triage --- coarse risk grouping rather than a per-query ranking**"。

## 4. Claim–Evidence Map 与摘要性断言同步

**接受，两处都改。**

- **S1 两条 triage 行**：第一行改为 "uncapped-architecture mechanism diagnostic, and the structure survives the cap"，证据列并排给出 uncapped（+0.16 [+0.08,+0.24]）与 final design（**+0.136 [+0.064,+0.207]**；分窗 +0.144/+0.151/+0.127；281 行 tie atom）；第二行改为 "graded harm avoidance (coarse risk grouping, not a per-query ranking)"，证据列并排 uncapped tertiles（+0.003/−0.054/−0.029）与 final design groups（**+0.005/−0.051/−0.074**，atom CI [−0.121,−0.025]；36% 位于 0.5），注明 selection 比较仅描述性。
- **"only significant" 全部删除**（与 ensemble −0.0356 [−0.0670,−0.0059] 的矛盾您指出得对）：摘要改 "are significant against the EB reference (…; the exploratory five-seed ensemble is as well)"；讨论改 "(alongside the exploratory five-seed ensemble)"；结论删 "only"；S1 改 "the capped-vs-EB intervals are significant, as is the exploratory five-seed ensemble"。grep 验证 "only significant" 两文档 = 0 处。

## 编辑性处理

- **随机化结果成为主线**：Table 8 重构——第一块即随机化共同目标结果（corpus +0.0025 [+0.0018,+0.0032]，200/200 >0，eval sizes 0.02514/0.02514；@0.025 +0.0025；N(0,1) +0.0128；N(0,0.5) +0.0397；location-held +0.0117/+0.0362），确定性插值块降为第二块并标注 "approximately size-adjusted; retained for continuity"，其诊断细节移至/保留在 S2（正文 §4.5 相应削减到一句 + 指针）。
- **位置 vs 离散度**（您的小限定，已用计算回应）：N(0,·) 与 canonical 的均值不同，确非纯 dispersion 实验——在相同 200 个划分上补跑 **location-held mixings**：N(+0.124, 1) → **+0.01171** [+0.01105,+0.01249]；N(+0.124, 0.5) → **+0.03615** [+0.03512,+0.03725]（全部划分为正）。持位置于语料均值、仅收窄 SD 的梯度与零中心版本接近（+0.0128/+0.0397），正文据此写 "the gradient is dispersion-driven"，并明示 N(0,·) 同时改变了 location。
- **可复现性与声明**：Data & Code Availability 改为 "DOI: to be assigned upon acceptance (Zenodo deposit of the tagged release)"，并明确 release 档案包含逐轮分析脚本（`scripts/`、`simulation/gold_standard/`）、逐 split/逐样本输出（`results/tables/`，含本轮 `round11_inference_scope.json` 中的逐 split size-gap 数组与此前各轮的 `canonical_estimates`、`round9_rows.json` 等）以及断言核对脚本（`scripts/verify_round*.py`），供独立复算——您指出的"无法独立复算 77 项断言"正是这些文件所解决的；本轮完整包随投稿附上。声明占位符将在接收前按期刊系统填写完成。

## 其他

- 页数：正文 58 页（+1：Table 8 重构与限定句）；补充 18 页（+2：round-11 段落与 S1 扩行）。
- 验证：`scripts/verify_round11.py` **75 项断言全部通过**；两文档 0 undefined references、0 处 ≥10pt overfull（补充 0 overfull）。

## 主要修改位置速查

| 修改 | 位置 |
|---|---|
| 训练半批精确 / 评估半批近似 两层表述；"exactly size-matched" 清零 | §3.1、§4.5、摘要、术语表、Table 8/9 caption、S1、S2、Fig S1 caption |
| 评估半批 size 差分布（逐 split 存档） | §4.5、S2、`round11_inference_scope.json` |
| Table 9 ≠ 随机化 estimand | Table 8/9 caption、§4.5 |
| conditional 措辞 + MCSE 重算 + mixing bootstrap 重跑 + 来源清单 | §4.5、摘要、结论、讨论、S1、S2 |
| location-held mixings（dispersion 与 location 分离） | §4.5、Table 8、S2 |
| selection 比较描述性 + 重叠 169 + 36% 粗分组 | §4.4、结论、S1 |
| S1 triage 两行并排 uncapped/final design；"only significant" 清零 | S1、摘要、讨论、结论 |
| Table 8 随机化主线重构；修稿历史退出摘要/结论 | Table 8、摘要、结论、§4.5 |
| DOI/复现性表述 | Data \& Code Availability |

*数字来源：`results/tables/round11_inference_scope.json`（+ round-10 artifacts）；全部引用值经 `scripts/verify_round11.py` 程序化对账。*
