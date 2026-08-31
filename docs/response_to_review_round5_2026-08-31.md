# 第五轮逐条审稿回应

**稿件**：Retrospective predictive calibration of oncology trial similarity for Bayesian historical borrowing
**本文档**：对第五轮意见（Major 1–7、其他 1–5）的逐条回应。

---

## 总览

我们采纳了编辑与审稿人指出的可发表路径：**混合校准全面降级为 "corpus-specific decision analysis"，通篇标注 exploratory**，不再声称 "validated net power benefit" 或任何形式的已验证平均频率学控制；论文的核心定位改为审稿人的表述——**"cap 限制学习借用的预测伤害；决策时收益尚未证实"**（结论节现以这句话收尾）。在降级的同时，我们把审稿人要求的硬检验全部做了：**size-matched 功效比较**证实原 +0.0066 中约三分之二确由 0.0007 的宽松度买来——等尺度下最终方案对不借用的差值只有 **+0.0024**（配对 MC se 0.0007；cap 0.3 为 +0.0057；EB-only 为 −0.0015，其表观优势全部是宽松度）；**mixing 稳健性**（非参数/分层/bootstrap/替代分布）表明该差值符号稳健、量级依赖 mixing（语料派生 +0.0010～+0.0063；漂移分布收窄时急剧上升，N(0,0.5) 下 +0.040）；**cap 保护性应力测试**表明其对 donor 精度 ×4 稳健、对 anchor 精度 ×4 不稳健。这些结果全部如实入正文（新表 tab:sizematched、新段落、S1/S2/S3）。

## Major 1：经验 mixing measure 与模拟冲突参数不等价

**接受其全部批评；框架降级 + 稳健性补齐。**

- **命名与地位**：§3.1 校准框架段重写——该分析现名 "corpus-specific decision analysis"，通篇 exploratory；段内逐条写明审稿人指出的四重不等价（观察经验 logit 差 vs 潜在一致漂移；异质 donor 集 vs uniform 格；全语料 vs 部署人群；同一分布用于 null 与 alt 平均）与正态假设的局限，并明确 "\emph{not} a verified average frequentist guarantee"。
- **非参数经验分布**：已补（raw 直方图与逐 query 收缩后直方图两版）；净差 +0.0010 / +0.0063。
- **分层分布**：按年代（≤2021-12-31 / 之后）与可得性（有/无可得 donor）分层拟合再按层份额混合；净差 +0.0025 / +0.0024——与 canonical 几乎不动。
- **去卷积与权重的 bootstrap 传播**：939 个 query 重抽 500 次 → canonical mixing 下净差 95% 区间 **[+0.0006, +0.0044]**。
- **替代合理 mixing 敏感性**：U(−1.5,1.5) +0.0059；N(0,1) +0.0127；N(0,0.5) +0.0403；无漂移点质量 +0.0767——净收益是漂移分散度的减函数，本语料分散度大（SD≈1.33），这正是决策时收益难产生的原因（该解读已写入正文与摘要）。
- **G0/G1 形式化与独立语料**：如实声明为未完成——不存在可用的独立语料；在真实 candidate-set 生成机制下积分需要联合建模 partial conflict 与方向异质性，我们把 partial-conflict 格保留在 0.05 上限的 worst-case 侧并在 S2 声明这一局限。这正是分析保持 exploratory 标签的原因。

## Major 2：独立验证批没有达到 ≤0.025 点目标；功效差部分由宽松度购买

**接受，两点都对，正文现按此陈述。**

- **0.0258 > 0.025 如实声明**：§4.5 现明写 "the frozen thresholds do \emph{not} deliver a point estimate ≤0.025"（并注明不借用参照自身为 0.0251，两者尺度不等——"both facts matter"）。
- **size-matched 比较（按建议 1 实施）**：对每设计在验证批上构建 size–power 曲线，把阈值插值到与不借用实测尺度 0.02513 完全相等：最终方案 +0.0024（se 0.0007）、cap0.3 +0.0057、cap0.1 +0.0026、EB-only −0.0015、cap0.5+SAM −0.0010——原 +0.0066 的约三分之二确系宽松度所购，正文明写。新表 tab:sizematched。
- **UCB≤0.025 校准（按建议 3 实施）**：以平均 type I 的 MC 95% 上界 ≤0.025 为准则重取阈值：不借用 0.5028、最终方案 0.5055、cap0.3 0.5090——排序不变。未重新选择设计，故无需第三批模拟（阈值插值是对同一冻结验证数据的公平重读，不是重新选择；此点在正文声明）。
- **"small but genuine validated net benefit" 撤回**：改为 "marginal on average ... exploratory"（摘要、§4.5、讨论、结论、S1 同步）。

## Major 3：0.05 ceiling 是容忍膨胀，不是安全控制

**接受，措辞全面修正。**

- 0.05 上限全文改称 **"tolerated pointwise inflation ceiling"**，§3.1 明写它是"a declared tolerance --- twice nominal --- not a derived quantity: we have no utility function or regulatory argument that fixes it"，并建议不接受该容忍度的读者直接读逐格 profile。
- "safe" → "predictively neutral"（主分析处，摘要/§4.1/结论）；"safety mechanism" → "harm-bounding mechanism"；"bounded-cost insurance" 删除（结论现为 "bounding the predictive harm ... with a tolerated, not eliminated, pointwise error inflation of up to 0.036 observed against the 0.05 ceiling"）。
- 条件带收益句现随附其代价："+4.4 到 +5.3 点（该处自身逐格 type I 达 0.024–0.036，容忍的膨胀显式陈述）"。风险收益比不做正向论证——正文明确把"是否值得"留给带效用函数的未来工作与读者判断。

## Major 4：mixture-mass cap ≠ 信息/影响力 cap

**接受，更名 + 实测。**

- **更名**：全文 "borrowed-mass cap" → **"historical mixture-mass cap"**（§2.10 定义、术语表、流程图）；§2.10 明写 "it caps component probability, \emph{not} information or influence"、"c=0.5 must not be read as 'at most half the information'"，对等原则降为 heuristic, not a theorem。
- **影响力诊断（实测）**：真实数据上 capped 设计的**后验**历史权重在 28% 的 forward query 超过 0.5（1.3% 超过 0.8，最大 0.92）——审稿人的担忧属实并被量化；但后验均值位移中位数 0.004、99 分位 0.063、最大 0.13（rate 尺度），primary 行最大 0.085；capped weighted nominal borrowed size 中位 6.2、最大 85。
- **donor 精度应力**：donor 臂样本量 ×4 世界（新 DGM 字段，默认路径比特级不变已验证）：uncapped 失败被放大（0.14/0.32/0.48 @ shift 0.5/1.0/1.5），**capped 全程 0.025–0.040**，与不借用同水平——cap 的保护对高信息 donor 稳健。
- **anchor 精度应力**：×4 时 capped 在 shift 1.0 达 **0.059**（越过容忍上限）——cap 的软肋在 anchor 而非 donor，与 EC 电池结论一致，正文如实写出；×1/4 全程 ≤0.050。
- **"fixed a priori" 更正**：全文改为 "fixed before the round-4 cap analyses"，§2.10 并明写 "we do not call it a priori in the strict sense: it was chosen during a revision cycle, after earlier rounds of data contact"。

## Major 5：title-screened 不应再与 strict ORR 混用

**接受。** 表 1 outcome 列改为 "binary response-type, title-screened (≈80% strict-ORR purity)"；正向上界结果处新增警示（"they inherit the title-screened subset's measured ≈80% strict-ORR purity, so they are statements about the screened binary-response estimand, not about a pure ORR population"）；"endpoint contamination was not driving the pattern" 改为方向性表述 + "none of these results should be read as established for a pure-ORR population"；描述级解析 + 双人仲裁保持为 pure-ORR 结论的前提（限制节）。

## Major 6：最终方案在稿件中未完全统一

**接受，六处全改。**

1. **流程图（图 1）重绘**：cap 进入主流程（红色 "Historical mixture-mass cap (final design)" 节点，含投影公式）；SAM 移出主流程，为右侧虚线灰框 "sensitivity analysis only --- not part of the final design"；caption 同步（"followed by the historical mixture-mass cap that completes the final design; the SAM adapter (dashed) is a sensitivity analysis only"）。
2. **表 3（held-out）**：新增粗体 "Final design (cap 0.5)" 行（test 2.8765 / dev 2.8411）；"selective" 行改标 "selective (uncapped ablation)"；caption 的 "primary comparison" 改为 "principal upper-bound comparison --- the primary analysis is Table 2"。
3. **表 4（forward）**：新增粗体最终方案行（2.8027，Δ −0.0413 [−0.0625,−0.0202]）；uncapped 改标 ablation。
4. **表 5（calibration）**：新增最终方案行（PIT KS 0.097；coverage 0.690/0.929/0.979）。
5. **图 3（held-out 图内标题）**：改为 "Hypothetical upper bound I: held-out comparisons vs the EB reference (availability-consistent primary analysis reported separately)"，并加入 final design 条与配对差点（−0.025）。
6. **§4.6 末尾**：该句在上一轮已改（"the final design carries an explicit mixture-mass cap ... SAM remains a sensitivity analysis"），本轮再清扫确认全文无 "required companion" 残留。

## Major 7：不可能性定理的适用范围

**接受。** 每处引用 [28] 现均限定条件：§3.1（"in one-parameter settings of the present kind, where the test with the external information fixed admits a UMP or UMP-unbiased competitor; our single-arm one-sided binomial design is within that scope, and we do not extrapolate beyond it"）、§4.5、引言 (iii)、讨论、结论、摘要（"in designs of this kind"）。"no borrowing method can do better" / "nothing borrowing can offer" 等绝对表述全部删除或限定。

---

## 其他 1–5

1. **"best type-I-controlled power"**：两处（§4.5 第一发现、讨论第四条）改为 "highest observed power among methods whose \emph{estimated} type I error was ≤0.025 in that cell (a Monte Carlo statement at 2,000 replicates, not proof of control)"。
2. **"exact τ\*"**：全文改为 empirical-quantile $\hat\tau^{*}$（"a Monte Carlo estimate, not an exact threshold"），表 6 表头与 caption 同步。
3. **§6.2 引用错误**：availability 分析的引用由 §4.3 改为 §4.1（sec:primary-results）。
4. **等 n 下采样从 3 次增至 8 次**：全部 8 次抽样的借用子组伤害为正（+0.082 到 +0.230，均值 +0.13），每次附 bootstrap 区间；"completing the attribution" 改为 "supporting the attribution"，并声明这是观察性支持而非完成的因果证明。
5. **篇幅**：中度瘦身——Case study 2（含表）与 robust-MAP 调参细节移入补充材料新 S3 节（"Material moved from the main text"），正文保留一句话指针；主表以 final design 行为主、uncapped 统一标 ablation。如实说明：因本轮同时要求补入 size-matched 表、mixing 稳健性、影响力/应力诊断等新内容，主稿净页数为 56 页（移出约 1.5 页、新增约 4 页）；若编辑认为仍需压缩，我们建议下一步把 §4.6（prospective signal 机制分析）整体移入补充材料。

## 主要修改位置速查

| 修改 | 位置 |
|---|---|
| 混合框架降级 + 四重不等价声明 | §3.1 重写、§4.5 第三层重写、摘要、结论、S1 |
| size-matched 比较 + UCB 校准 | §4.5 + 新表 tab:sizematched、S2 |
| mixing 稳健性（非参数/分层/bootstrap/替代） | tab:sizematched 下半、S2、`gold_standard_mixing_robustness.json` |
| tolerated ceiling 更名 + safe 清扫 | §3.1、§4.5、摘要、讨论、结论 |
| mixture-mass cap 更名 + 影响力诊断 + 精度应力 | §2.10、§4.5 新段、图 1、S1/S2 |
| title-screened 纯度警示 | 表 1、§4.3、结论 |
| 最终方案统一（图 1/3、表 3/4/5） | 对应图表 |
| 定理限定 | 所有 [28] 引用处 |
| 瘦身 | 补充材料 S3 |

*数字来源：`results/tables/gold_standard_size_matched.json`、`gold_standard_mixing_robustness.json`、`gold_standard_round5_stress.json`、`influence_diag.json`、`capped_table_rows.json`、`equal_n_extra.json`。*
