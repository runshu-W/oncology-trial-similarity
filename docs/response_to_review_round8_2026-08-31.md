# 第八轮逐条审稿回应

**稿件**：Retrospective predictive calibration of oncology trial similarity for Bayesian historical borrowing
**本文档**：对第八轮意见（1–5）的逐条回应。

---

## 1. "cap attenuated harm" 需要同一查询上的直接配对对比

**接受，已计算（新脚本 + 新表）。** 按您指定的估计量，对每一行计算
Δᵢ = NLL_capped(0.5),i − NLL_uncapped,i（同一查询、同一 donor 集，仅 cap 投影不同；uncapped 借入质量 ≤ 0.5 的行 Δᵢ 恒等于零，均保留在 pooled 分析中并报告"binding"行数），普通与 strongest-donor block bootstrap 95% CI（B = 4000；RNG 20260910；`round8_direct_paired.json`；正文新增 Table（`tab:directpaired`））：

| 分析 | n | binding | 均值 | 普通 CI | block CI |
|---|---|---|---|---|---|
| primary pooled | 589 | 50 | −0.0289 | [−0.0530, −0.0091] | [−0.0527, −0.0101] |
| primary engaged | 119 | 50 | −0.1428 | [−0.2584, −0.0480] | [−0.2581, −0.0556] |
| forward upper bound | 775 | 281 | −0.0148 | [−0.0301, −0.0011] | [−0.0299, −0.0014] |
| title-screened retrained | 589 | 156 | −0.0147 | [−0.0355, +0.0037] | [−0.0371, +0.0054] |
| title-screened frozen | 589 | 176 | −0.0108 | [−0.0254, +0.0027] | [−0.0260, +0.0029] |

**措辞按结果分层。** primary pooled、primary engaged、forward 三处的直接区间（普通与 block 均）不含零——"attenuated / improved the paired score" 在这三处保留，并且摘要、§4.1、讨论、结论中的 attenuation 声明现在明确以这些直接配对区间为依据。两个 title-screened 区间含零——按您的规则降级；唯一的适配：您给的备选句 "the capped point estimate was less harmful" 面向 harmful 情形，而 title-screened 两层的 capped 与 uncapped 都优于 EB，直译会弄错方向，故写为 "the capped point estimates are more favourable but the paired intervals include zero … the supported statement is only that the cap did not detectably tax the benefit"（§4.2）——同一规则（区间含零 → 只报点估计方向、不声称改进）在方向上的诚实适配。

配套文字修改：§4.2 "\emph{improves} the upper-bound results rather than taxing them" → "leaves the upper-bound results at least as favourable" + 直接对比句；"the harm-attenuating mechanism costs nothing on the favourable tiers" → "measured directly, … improved the forward tier and cost nothing detectable on the title-screened tiers"；S1 "The mixture-mass cap improves, not taxes, …" 行整行改写（直接配对数字前置；title-screened 不声称 within-tier improvement）。

一致性核对（程序化）：forward 配对均值 = 已发布的 capped/uncapped 平均 NLL 之差（2.8027 − 2.8175 = −0.0148）；primary 配对均值 = 已发布逐行 artifact 中 (+0.006) 与 (+0.035) 两列的逐行差（primary 两行直接由已发布的 `rows_strictB.json` 离线重算，未重新打分）。

## 2. "neutral" 未经等价检验

**接受，全文消除。** "neutral" 在正文、图注、S1 中已为 0 处（grep 验证）：摘要、§4.1（设计、结论句、frozen control、proxy-rule 各处）、图 2 caption 与图内标题（重绘）、§4.4 "exactly neutral"、讨论（三处）、结论、S1（两行）→ "not detectably different from the EB reference"（对 control 行为 "shows no gain / no detectable difference"，即只作否定性陈述）。结论 "demonstrates \emph{no} predictive gain" → "does not demonstrate a predictive gain"（§4.1 原有的 "no demonstrated predictive gain" 本就是"未证明有"，保留）。

## 3. Repeated cross-fit 只刻画同批内的 split 变异

**接受，三项全改 + 新 artifact。**

- 摘要 "once threshold-estimation uncertainty is propagated" → "assessed by repeated random splitting"；§4.5 "so that threshold-estimation and split variability actually propagate" 已删改（"assessed rather than hidden"）。
- **均值 vs 中位数**：−0.0007 是均值；摘要、§4.5、结论、S1 现同时给出中位数 −0.0009。分布右偏：8/200 个划分形成分离的正尾（+0.0022 至 +0.0057），其余 192 个划分位于 +0.0001 之下，故均值位于约第 75 划分百分位。
- **补充材料**给出全分位数 2.5/25/50/75/97.5% = −0.00144 / −0.00114 / −0.00094 / −0.00072 / +0.00390 与 20-bin histogram counts（`gold_standard_round8_crossfit_dist.json`；以同一 RNG 20260908 重跑，200 个划分估计 bit-identical，均值/SD 与 round-7 artifact 程序化核对一致）。
- **区间性质**：§4.5、结论、S2 均明写——split interval 是条件于单一验证批的随机对半划分的描述性分位区间，**不是**总体差异的置信区间，不包含批间抽样变异。

## 4. "roughly symmetric" 与数值矛盾；F14 caption 的 "power"

**接受。** §4.5 改为您的表述："locates an observed local advantage at small drift of either sign, with \emph{asymmetric} magnitude"，并把衰减写成不对称对："fades faster on the positive side (+0.057 at −0.5 versus +0.020 at +0.5; +0.022 at −0.75 versus +0.007 at +0.75)"（"roughly symmetric" 已删）。摘要、讨论第六观察（并列出 −0.5/+0.5 对比数）、结论同步 "asymmetric (in) magnitude"。F14 TEX caption："Type I error (top) and power (middle)" → "Null-world rejection rate (top) and alternative-world rejection rate (middle)"；"The final design's power gain concentrates…" → "alternative-world rejection-rate advantage --- a rejection-rate difference at unequal cell-specific sizes (its per-cell null-world rejection runs 0.024–0.036), not a size-matched power gain --- concentrates…"（图轴在第七轮已改为 rejection rate，本轮 caption 同步；图内 ceiling 标注亦改为 "tolerated ceiling 0.05" 与术语表一致，图已重绘）。

## 5. 残留措辞同步

- **引言 (v)（p.6）**："a small mixture-average gain concentrated where donors are approximately exchangeable in simulation" → "no established mixture-average gain in simulation (an observed cell-specific ROC-efficiency advantage at small drift, with the size-adjusted average not established in either direction)"。
- **p.17**："The forward windows … carry the confirmatory weight" → 按您的限定改写："the first two forward windows informed development, and every capped result is a post hoc redesign evaluation, so only the pre-specified uncapped comparisons on the final window --- examined once, in aggregate, under pre-registered criteria and multiplicity control --- carry (limited) confirmatory weight"。
- "all examined point estimates were positive" → "all non-repeated plug-in estimates were positive"（§4.5、讨论、结论三处）。
- "its real structure is \emph{local}" / "a real local advantage" → "an observed local ROC-efficiency advantage"（§4.5、讨论）。

## 其他

- **页数**：正文 55 → 56 页（新增直接对比表）；补充材料 13 → 14 页（round-8 additions 段）。
- **验证**：全部引用数字经程序化对账（`scripts/verify_round8.py`：109 项断言——新旧数字、四舍五入两档、histogram 计数、以及关键措辞的存在/缺失——全部通过）；两文档 0 undefined references、无 ≥10pt overfull。

## 主要修改位置速查

| 修改 | 位置 |
|---|---|
| 直接配对对比（5 项分析，普通 + block CI） | §4.1、§4.2、新 Table（`tab:directpaired`）、摘要、讨论、结论、S1、`round8_direct_paired.json` |
| title-screened 降级措辞（区间含零） | §4.2、S1、摘要（Conclusions 句） |
| "neutral" 全文消除 | 摘要、§4.1、图 2 caption + 图内标题（重绘）、§4.4、讨论、结论、S1 |
| "does not demonstrate a predictive gain" | 结论 |
| "assessed by repeated random splitting" + mean/median + 分位数 + histogram + 非置信区间声明 | 摘要、§4.5、结论、S1、S2、`gold_standard_round8_crossfit_dist.json` |
| asymmetric magnitude + F14 caption rejection-rate 措辞 | §4.5、摘要、讨论、结论、图 7 caption（+ 图内 ceiling 标注重绘） |
| 引言 (v) / p.17 confirmatory weight / plug-in / observed local ROC-efficiency advantage | §1、§3.1 前置段、§4.5、讨论、结论 |

*数字来源：`results/tables/round8_direct_paired.json`、`gold_standard_round8_crossfit_dist.json`（+ 既有 round-4/6/7 artifacts）；全部引用值经 `scripts/verify_round8.py` 程序化对账。*
