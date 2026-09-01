# 第九轮逐条审稿回应

**稿件**：Retrospective predictive calibration of oncology trial similarity for Bayesian historical borrowing
**本文档**：对第九轮意见（1–4）的逐条回应。

---

## 1. 直接配对结果的尾部稳健性

**接受，全部计算（新脚本 `round9_tail_robustness.py`，逐行 rows 已发布），您指出的尾部结构确实存在，并且两个层的"典型行为"方向相反——已如实写入。**

您的推算正确：primary binding 行（50/589）条件均值为 **−0.340**。全套补充统计（bootstrap RNG 20260912；`round9_tail_robustness.json`、`round9_rows.json`；正文 §4.1/§4.2 + 补充材料新表 Table S-tail 均已写入）：

| 分析 | binding | 中位数 | IQR | 5–95% | 范围 | trim₁₀ | wins₅ | LOO-query | LOO-block |
|---|---|---|---|---|---|---|---|---|---|
| primary pooled | 50 | −0.030 | [−0.39,+0.07] | [−2.06,+0.36] | [−4.66,+0.44] | −0.154 | −0.273 | [−0.030,−0.021] | [−0.030,−0.021] |
| primary engaged | 50（同一组行）| −0.030 | 同上 | 同上 | 同上 | −0.154 | −0.273 | [−0.148,−0.105] | [−0.152,−0.105] |
| forward | 281 | **+0.021** | [−0.06,+0.11] | [−0.69,+0.30] | [−2.16,+0.52] | **+0.017** | −0.016 | [−0.015,−0.012] | [−0.015,−0.012] |
| title-scr. retrained | 156 | +0.020 | [−0.08,+0.17] | [−0.70,+0.40] | [−3.48,+0.58] | +0.021 | −0.009 | [−0.016,−0.009] | [−0.017,−0.009] |
| title-scr. frozen | 176 | +0.022 | [−0.05,+0.11] | [−0.69,+0.29] | [−1.65,+0.49] | +0.016 | −0.018 | [−0.012,−0.008] | [−0.012,−0.007] |

发现分两层：

- **Primary**：重尾但方向不由单行驱动——binding 中位数 **−0.030** 为负、30/50 行为负、trim₁₀ −0.154、wins₅ −0.273，全体 589 行 1%/99% winsorize 后 −0.017；**所有** leave-one-query-out 与 leave-one-donor-block-out 均值保持为负（删去最大单行改善 −4.66 后剩 −0.021）。按窗口：w1 −0.012（2 行 binding）、w2 +0.004（[−0.003,+0.014]，9 行）、w3 −0.053（[−0.095,−0.020]，39 行）——集中于 w3。
- **Forward**：您怀疑的"少数极端行驱动"在此层成立且已明写为保险式结构——**典型 binding 行付出小额代价**（中位数 +0.021），trim₁₀ 均值翻正（+0.017），均值改善由少数大额纠正承载（5% 的 binding 行改善超过 0.69）；LOO/LODO 仍全负（[−0.015,−0.012]）。正文原句 "improved the paired score" 已改为 "improved the forward-tier mean through its left tail"，并加入与 primary 相反的机制说明（训练良好的 forward 模型通常被 cap 轻微征税、偶尔被从大额过度借入中拯救；primary 的典型 binding 行本身即改善）。

措辞落实：§4.1 现区分 **marginal**（全部行）与 **conditional on binding**（50 行，−0.340）两种效应并给出全部尾部统计；"the cap reduced the harm" 已改为 **"the cap \emph{reduced the mean NLL} in this post hoc comparison"**（摘要同步 "directly measured to reduce in mean"；结论 "showing a mean reduction"）。

## 2. 未经非劣检验支持的"没有代价"表述

**接受，三处原句全部删除，总体表述分层。**

- "leaves the upper-bound results at least as favourable" → "yields:"（纯描述）；
- "…the supported statement is only that the cap did not detectably tax the benefit" → 您的句子："the point estimates favoured the capped version, but the intervals are compatible with both modest improvement and modest worsening, and no directional conclusion is supported there"；
- "cost nothing detectable on the title-screened tiers" → "…reduced the mean paired NLL on the primary rows, improved the forward-tier mean through its left tail, and was directionally inconclusive on the title-screened tiers"。

摘要与结论现按您的三分法表述（不再统称 attenuation）：**primary = "reduced the mean harm"（附 LOO 稳健性）；forward = "improved an already favourable tier mean (a tail-driven, insurance-like effect)"；title-screened = "direction inconclusive"**。S1 两行同步（cap 行改为 "improved the forward-tier mean (a tail-driven, insurance-like effect) and is directionally inconclusive on the title-screened tiers"）。

## 3. Repeated cross-fit 正尾的展示与解释

**接受：直方图已实际放入补充材料（Figure S1），逐 split 诊断已计算，尾部机制已定位，且"dispersion dependence"按您的首选项用重复 cross-fit 重新验证。**

- **展示**：补充材料新增 **Figure S1**——(a) 200 个 canonical split 估计的 20-bin 直方图（标注 mean/median 与 8-split 正尾）；(b) canonical ECDF + 相同 200 个划分在 N(0,1)、N(0,0.5) 下的中位数与 central 95% split-quantile range。
- **逐 split 记录**（`round9_crossfit_diag.py`，同种子 20260908，estimates bit-identical；`round9_crossfit_diag.json` 存每个 split×fold 的两侧阈值、估计半批达成 size 与**评估半批达成 size**）。
- **机制定位——正是您猜测的阈值离散性**：split 估计与评估半批 size 差（cap50 − no-borrowing）的相关为 **0.912**；no-borrowing 的阈值在 400 个 fold-half 上只取 **4 个离散值**（其检验统计量离散；bisection 残差最大 0.0013），典型 split 中它在评估半批上略超尺度（0.0257 vs 目标 0.02513，把估计压负），而 8 个尾部 split 恰是它落在评估半批**欠尺度**（0.0249）的划分；cap50 的评估半批 size 稳定（0.02514–0.02515）。按 cell 分解：尾部与非尾部 split 的逐 cell 贡献几乎一致——**不是特定 cell 的功效结构**。负核与正尾同源于 size-matching 粒度，这一诊断强化（而非削弱）"no average difference established" 的结论，已写入 §4.5 与 S2。
- **命名**：全文（摘要、§4.5、讨论、结论、S1、S2）统一改为 **"central 95% split-quantile range"**。
- **Dispersion dependence**：按您的"最好"选项，在**相同 200 个划分**上以 N(0,1)、N(0,0.5) 混合重复 cross-fit（size 目标 = no-borrowing 冻结阈值在该混合下的全批加权 type I：0.02535/0.02620）：N(0,1) mean **+0.0097**（median +0.0094；range [+0.0090,+0.0149]）；N(0,0.5) mean **+0.0382**（median +0.0373；range [+0.0365,+0.0432]）——**两者 200/200 个划分全部为正**，对照 canonical 的 −0.0007。"is robust" 现有重复验证支撑（"robust now also under repetition"），不再只是 plug-in 模式。

## 4. 图表与最终设计身份

**接受。**

- **Figure 4（forward validation）**：(a) 加入 **final design (cap 0.5) 柱（2.803，绿色）**，红色行全局改名 **"Selective (uncapped ablation)"**（同一构建脚本的 Figures 3/4/5/6 标签一并同步）；(b) 每窗口成对 Δ 现同时画 final design（绿）与 uncapped ablation（红）：capped w1/w2/w3 = −0.043 [−0.087,+0.003] / −0.032 [−0.077,+0.007] / **−0.045 [−0.074,−0.016]**（来自 `round9_tail_robustness.json` 的 fwd 逐窗口重打分；构建脚本以 check() 与 artifact 对账后绘图）。Caption 同步。
- **结论 p.51**：full-corpus forward 与 title-screened 明确分开——"in two populations that we keep distinct: the full-corpus forward tier (n=775, no endpoint screening) and the title-screened reruns (which inherit the subset's measured ≈80% strict-ORR purity)"。

## 其他

- 页数：正文 57 页（+1：尾部稳健性两段）；补充材料 16 页（+2：round-9 段落、Table S-tail、Figure S1）。
- 验证：`scripts/verify_round9.py` **130 项断言全部通过**（尾部统计、逐窗口、诊断相关/阈值离散、alt-mixing 分位数、F2 caption 数字、措辞存在/缺失）；两文档 0 undefined references、无 ≥10pt overfull（补充材料 0 overfull）。

## 主要修改位置速查

| 修改 | 位置 |
|---|---|
| 尾部统计 + marginal/conditional 区分 + LOO/LODO + 逐窗口 | §4.1、§4.2、Table S-tail（补充）、`round9_tail_robustness.json`、`round9_rows.json` |
| "reduced the mean NLL in this post hoc comparison" | §4.1（摘要/结论同步 in-mean 措辞） |
| forward 保险式尾部结构（median +0.021，trim 翻正） | §4.2、S1、摘要/讨论/结论 "tail-driven, insurance-like" |
| 三处"没有代价"句删除 + 三分层总体表述 | §4.2、摘要 Conclusions、结论、讨论、S1 |
| Figure S1（直方图+ECDF）+ 逐 split 诊断 + 尾部机制 | S2、§4.5、`round9_crossfit_diag.json` |
| central 95% split-quantile range 统一命名 | 摘要、§4.5、讨论、结论、S1、S2 |
| N(0,1)/N(0,0.5) 重复 cross-fit（全正） | §4.5、S1、S2 |
| Figure 4 final design 柱 + "Selective (uncapped ablation)" 改名 + 双系列 per-window | `build_main_figures_selective.py`（Figures 3–6 重绘）、图 4 caption |
| forward 与 title-screened 人群分开表述 | 结论 |

*数字来源：`results/tables/round9_tail_robustness.json`、`round9_crossfit_diag.json`（+ round-8 artifacts）；全部引用值经 `scripts/verify_round9.py` 程序化对账。*
