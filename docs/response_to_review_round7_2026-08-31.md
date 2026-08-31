# 第七轮逐条审稿回应

**稿件**：Retrospective predictive calibration of oncology trial similarity for Bayesian historical borrowing
**本文档**：对第七轮意见（1–6）的逐条回应。

---

## 1. Independent validation 与 in-validation reanalysis 的混用

**接受，已拆分为两个命名不同、句子分开的分析。**

- §3.1（Methods）重写为两句：*The independent validation*（阈值在选择批冻结 → 独立验证批重评；"there, and only there, threshold selection and error-rate assessment share no simulation data"）与 *an in-validation size-adjusted reanalysis*（阈值在验证批自身上调整；"carries no independence claim"）。
- 结论节同样拆分（"keeping its two parts distinct: The independent part … The in-validation part …"）。原混合句已删除；全文不再把表 7 称为 independent validation。

## 2. Primary capped analysis 的局部 post hoc 标注

**接受。** §4.1 正文改为审稿人建议的描述式表述——"Because the cap was introduced after uncapped primary results of this kind had been examined, this capped-versus-uncapped contrast is a *post hoc exploratory redesign comparison*, and we state it descriptively rather than causally: on these decision-time rows **the capped version was neutral whereas the uncapped version was harmful** --- consistent with, but not independent evidence for, the attenuation observed in simulation"（"the cap therefore does exactly what it does in simulation" 已删除）。表 2 caption 与图 2（forest）caption 均加入 "post hoc exploratory redesign comparison"；图内标题重绘为 "Primary analysis: capped neutral, uncapped harmful (post hoc exploratory redesign comparison)"。

## 3. 两折 cross-fit 不足以确立符号

**接受，选择了第一个选项（重复重采样）——结果证明您的怀疑完全正确，结论已相应改写。**

新计算：**R = 200 次随机对半划分的重复 cross-fit**（每次划分：一半用二分法解 size-matching 阈值、另一半折外配对评估，两折平均）。结果：分布中心 **−0.0007**（划分间 SD 0.0011；95% 划分区间 **[−0.0014, +0.0039]**；仅 **4%** 的划分为正）。

也就是说：全批插值的 +0.0024 与单次 cross-fit 的 +0.0057 都在阈值估计噪声之内。正文、表 7 caption、摘要、讨论、结论、S1 全部改为：**"all examined point estimates were positive, and the sign is not established once threshold-estimation uncertainty is considered; no average net power difference at matched size is established in either direction"**——比您给出的备选措辞更进一步，因为重复重采样的结果实际支持"中心在零附近"。唯一保留为稳健的是分散度依赖（收益随假设漂移分散度收窄而增大）。"sign-robust" 措辞已删除。

## 4. Pointwise size matching 是诊断，不是可实施设计

**接受，五项全改。**

- **重命名与定性**：全文改称 **"cell-specific ROC-efficiency diagnostic"**，并明写 "Because each cell uses its own threshold, this diagnostic conditions on the true drift --- a quantity unknown in any real trial --- so it describes ROC efficiency cell by cell and is *not* an implementable conditional power benefit"（§4.5、表 7、摘要、讨论、结论、S1）。
- **补全负漂移与 partial-conflict 格**（新计算，全部 22 格）：诊断在漂移上近似对称——零附近 ≈ +0.07（0 处 +0.069、−0.25 处 +0.074、+0.25 处 +0.054），经 ±0.5–0.75 衰减（−0.75 处 +0.022、+0.75 处 +0.007），|drift|≳1 反转（−0.02 ～ −0.03）；partial 格按位移分数梯度（0.5×0.25：+0.060 → 1.5×0.75：−0.007）。|值|<0.01 在划分噪声内，已注明。
- "confined to drifts ≤0.75" 已改为对称表述（"at small drift of either sign … fading through ±0.5–0.75"）。
- **"approximate exchangeability" 降格**：正文明写 "at a logit drift of 0.75 (odds ratio ≈ 2.1) the donors are in any case not approximately exchangeable with the query"；摘要中 "real exactly where approximate exchangeability holds" 已删除，改为 "locates the design's local advantage at small drift of either sign"。
- **图 7 caption 与轴标签**：中/上面板 y 轴改为 "Alt-world / Null-world rejection rate"，caption 中 "power gain" 改为 "alternative-world rejection-rate difference"（该图为冻结阈值、逐格尺度不等，caption 已注明）。

## 5. 新颖性陈述与 S4 同步

**接受，采纳您的句子。** 引言贡献 (ii) 现写："the architecture **permits** separate per-source discounting, but in the final selective model the learned discount response was empirically flat (Supplement S4), and **the observed gain is attributable primarily to the EB anchor and the set-level total-borrowing head**"；donor-level discrimination 与 active per-source discount 均列为"未被实证的架构能力"。Positioning 小节的第二增量声明限定为 fixed-budget 架构（"in the final selective model even that role disappears … we no longer claim an active per-source discount as a working mechanism of the final design"）。结论开篇同步（"an architecture that *permits* … The evidence supports less than the architecture permits"）。我们不主张 discount head 必要性消融——按您给出的二选一，选择了收缩表述。

## 6. 其他

- **365 vs 346 blocks**：两个分析对无 donor 查询的处理不同——本轮 capped 分析的规则（released script）为"按 rule weight 取最强 gated donor；无 gated donor 的查询各自成单例块"（365 块）；此前发布的依赖性分析用其自身的 strongest-gated-donor 赋块（346 块）。§4.3 已作说明，并补了**块规则敏感性**（新计算）：strongest-by-rule-weight [−0.0669,−0.0169]、无 donor 合并为一块 [−0.0665,−0.0182]、strongest-by-gate [−0.0634,−0.0203]——区间对规则不敏感（S2 + `round7_block_rules.json`）。
- **S1 "carry confirmatory weight"**：已限定 "for the pre-specified uncapped architecture comparisons only (not for the post hoc capped redesign)"。
- **统一命名**：全文 "title-screened ORR subset" → **"title-screened binary-response subset"**（定义处与表 1 带 ≈80% strict-ORR purity）。
- **篇幅**：本轮页数持平（55 页）——第 3、4 条新增的重复 cross-fit 与全格诊断结果占用了 §4.6 外移腾出的空间。若编辑仍要求压缩，我们准备把 §6.1 数据缺陷叙述史（约 2.5 页）移入补充材料，等待编辑意见（该节承载 data-integrity 披露，未擅动）。

## 主要修改位置速查

| 修改 | 位置 |
|---|---|
| 两分析拆分命名 | §3.1、结论 |
| primary post hoc 局部标注 + 描述式表述 | §4.1、表 2 caption、图 2 caption + 图内标题（重绘） |
| 重复 cross-fit（R=200）+ 符号不成立结论 | §4.5、表 7 caption、摘要、讨论、结论、S1、`gold_standard_round7_crossfit.json` |
| ROC-efficiency diagnostic 重定性 + 22 格全表 + OR≈2.1 说明 | §4.5、表 7 底块、摘要、讨论、S1 |
| F14 轴标签/caption 改 rejection-rate | 图 7（重绘） |
| 新颖性同步（您的句子逐字采纳） | 引言 (ii)、Positioning、结论 |
| block 规则说明 + 敏感性 | §4.3、S2、`round7_block_rules.json` |
| confirmatory weight 限定 / binary-response 更名 | S1 / 全文 |

*数字来源：`results/tables/gold_standard_round7_crossfit.json`、`round7_block_rules.json`；全部引用值经程序化对账。*
