# 第六轮逐条审稿回应

**稿件**：Retrospective predictive calibration of oncology trial similarity for Bayesian historical borrowing
**本文档**：对第六轮意见（Major 1–5、Minor 1–4）的逐条回应。

---

## Major 1：cap 并未真正 "bound" 预测伤害或频率学风险

**完全接受。**

- **核心表述采用审稿人原句**（结论节与摘要 Conclusions 逐字采纳）："Across the evaluated retrospective datasets and canonical simulation settings, the cap \emph{attenuated} the observed harm of learned borrowing; neither a general harm bound nor robust type-I-error protection has been demonstrated."
- **"bound/bounding" 全文清扫**：cap 相关的 bound 语义改为 attenuate/attenuation（摘要、§4.1、§4.5、讨论第二/四/六条、限制节、结论；§2.10 中保留的 "cap" 仅指对先验混合概率的数学投影本身）。
- **摘要与结论补入 anchor 应力结果**：不再只报 0.036——现同时报 "0.060 under anchor-precision stress, exceeding the ceiling"。
- **0.059 的推断精度**：不再称 "marginally above"。正文现报 "0.0595（MC se 0.0053；单侧 95% Clopper–Pearson 上界 0.069）——exceeding the tolerated ceiling, and the reason we do not claim robust type-I-error protection for the cap"（§4.5 应力段、S1、S2）。

## Major 2：capped 真实数据证据是 post hoc，不是独立验证

**完全接受。**

- **统一标注**：§4.3 capped 段新增总管句——cap 是在检视过早期 uncapped held-out/forward 结果之后加入的，重新应用于同一批结果集不能恢复 lockbox，因此**本文所有 capped 真实数据结果一律为 "post hoc exploratory redesign evaluation"**；独立确证需要新 registry vintage、未接触时间窗或外部数据（该句同时写入摘要、讨论第二条、结论、S1、S2）。
- **措辞删除/限定**："pre-committed" 全部删除（两处改 "previously fixed"）；"demonstrated"/"independently validated" 用于 capped 真实数据处删除或改为 observed；"improves out-of-time generalisation" 从摘要移除。
- **补齐 capped 上界层的 donor-block bootstrap**（新计算）：forward −0.0413，block 95% CI **[−0.0665, −0.0166]**（365 块，最大 30）；retrained title-screened −0.0485，block **[−0.0815, −0.0189]**；frozen title-screened −0.0440，block **[−0.0707, −0.0179]**（267 块，最大 18）。三条区间在块重采样下仍不含零，已写入 §4.3 与 S1（并同样戴 post hoc 标签）。

## Major 3：size-matched 分析不是同一意义上的独立验证

**接受。**

- **命名区分**：全文把两类分析明确分开——表 8（冻结阈值 → 独立验证批）保持 "independent validation"；表 7 的尺度匹配现称 **"in-validation size-adjusted reanalysis"**，§4.5 明写其阈值是在验证批自身上插值估计的；摘要同步（"in-validation size-adjusted reanalysis"）。结论中不再把两者混在一句里。
- **阈值估计误差的传播（新计算，cross-fit）**：每格 10,000 复制对半拆分，一半估阈值、另一半配对评估（两折平均）：最终方案 **+0.0057**（每折 MC se 0.0010；评估半批实际达成尺度 0.0244/0.0260 对目标 0.0251）。与全批插值的 +0.0024 并报，正文明写：**两个估计量之间约 0.003 的差距本身就是引用的 MC 标准误所不含的阈值估计不确定性的量级，因此尺度调整后的平均差不 pin 定于任何单值，只表述为 "small and positive, of order +0.002 to +0.006"**；mixing 权重不确定性（bootstrap [+0.0006,+0.0044]）与去卷积正态形状仍未传播、并如实声明。表 7 重做：列出匹配阈值、实际达成尺度；UCB 校准块列出阈值/达成尺度/UCB/功效。
- 未新增第三批模拟：cross-fit 即审稿人所列 "nested/cross-fitted calibration" 选项；且分析保持 exploratory 定位。

## Major 4：conditional gain 未做逐点 size matching

**接受，逐点匹配已做（新计算），结论实质性修正。**

对每个 uniform drift 格点，把两个设计的阈值分别插值到**该格 null 拒绝率恰为 2.5%**（同样 cross-fit：一半定阈值、另一半评估；两设计实际达成 0.024–0.026）：

| drift | 0 | +0.25 | +0.5 | +0.75 | +1.0 | +1.25 | +1.5 |
|---|---|---|---|---|---|---|---|
| Δpower vs no borrowing | **+0.074** | +0.051 | +0.020 | +0.014 | **−0.019** | −0.032 | −0.027 |

（每格 MC se ≈ 0.003。）

审稿人的怀疑对一半成立、另一半反转：**drift ≥ 1.0 处的冻结阈值 "+4~5 点收益" 确实是局部宽松度**——逐点匹配后变为显著为负；而 drift ≤ 0.75 处的优势是真实的 ROC 优势，且在零漂移处比冻结阈值下更大（+7.4 点）。正文现表述为："the like-for-like conditional advantage is confined to drifts ≤ 0.75; the apparent frozen-threshold gains at drift ≥ 1.0 were local liberality, and we no longer describe them as conditional power gains"；冻结阈值 profile 全文改称 **rejection-rate differences at unequal per-cell sizes**。摘要、讨论第六条同步。

## Major 5：S1 中 "endpoint contamination was not driving" 过强

**接受。** 该行改为 "The directional upper-bound pattern survives title-based endpoint enrichment"，Status 列明写 "Directional survival only: with ≈20% residual non-ORR endpoints, one rater, and wide PPV intervals, a causal exclusion of endpoint contamination is not supported"。

---

## Minor 1–4

1. **UCB 校准定义**：S2 更正为 "\emph{smallest} threshold whose MC 95% upper bound of average type I is ≤ 0.025 --- smallest, because power falls in the threshold"（正文 §4.5 同句已带此限定；代码实现本就取最小满足阈值，与表 7 数值一致——错的是文字，已改）。
2. **表 7 列**：UCB 校准块现列出每设计的阈值 / 实际平均 type I / 其 UCB / 功效（不再只报功效）；全批匹配块也加了阈值与达成尺度列。
3. **S1 "type-I-controlled power"**：改为 "highest observed power among methods with estimated type I error ≤ 0.025 in that simulated cell"，Evidence 列注明 "a point estimate, not demonstrated control"（与正文两处一致）。
4. **篇幅**：§4.6（prospective signal 机制分析）整节连同两图移入补充材料新 S4，正文留一段摘要（其中同步修正了残留的 "protection rests on … the SAM layer" 表述——现指向 cap）；摘要压缩约三分之一（Results 段收拢为单段网状陈述）。如实报告页数账目：移出约 3 页、本轮新要求的表列扩充与应力/逐点分析新增约 2 页，主稿 56 → 55 页、补充材料 10 → 13 页。若编辑认为仍需压缩，下一候选是 §6.1 数据缺陷叙述史（约 2.5 页）整体移入补充材料——我们未擅自移动，因为它承载 data-integrity 披露义务，听候编辑意见。

## 主要修改位置速查

| 修改 | 位置 |
|---|---|
| attenuate 措辞 + 审稿人核心句 + anchor×4 入摘要/结论 + SE/UCB | 摘要、结论、§4.1/§4.5/讨论/限制节、S1/S2 |
| post hoc redesign evaluation 标注 + block CIs | §4.3 capped 段、摘要、讨论、S1、`round6_block_capped.json` |
| in-validation size-adjusted reanalysis + cross-fit + 表 7 重做 | §4.5、tab:sizematched、S1/S2、`gold_standard_round6_analysis.json` |
| 逐点 size matching（+0.074 → 反转） | §4.5、tab:sizematched 底块、摘要、讨论第六条 |
| S1 两行修正 + UCB smallest | S1、S2 |
| §4.6 → S4 + 摘要压缩 | 正文 §4.6（摘要段）、补充材料 S4 |

*数字来源：`results/tables/gold_standard_round6_analysis.json`、`round6_block_capped.json`；全部引用值经程序化对账。*
