# 第十轮逐条审稿回应

**稿件**：Retrospective predictive calibration of oncology trial similarity for Bayesian historical borrowing
**本文档**：对第十轮意见（1–2 + 小问题）的逐条回应。

---

## 1. repeated cross-fit 未真正实现 equal-size comparison

**接受，按您给出的第二种方法（凸组合 / 固定随机化概率）完整重算，并全部改用共同目标——结果推翻了我们此前"中心在零"的结论，这一修正实质重要，感谢您坚持这一点。**

**方法**（`round10_randomized_crossfit.py`；同一 200 个划分，RNG 20260908）：每个 training half、每个设计构造经典随机化检验——v > c 必拒、v = c 以概率 γ 拒——(c, γ) 使 training half 的加权平均 type I **精确等于**目标；随后固定 (c, γ)，在 evaluation half 以**期望**形式应用（即相邻两个可达检验的凸组合，不注入随机化的 Monte Carlo 噪声）。两个设计、三个 mixing 全部使用**同一**目标；目标取 0.02513（沿用论文惯例）与 0.025（名义值）各跑一遍。

**结果**（`round10_randomized_crossfit.json`）：

| run | mean | median | SD | central 95% range | 全正? | eval size (no-borrow / cap50) |
|---|---|---|---|---|---|---|
| canonical @0.02513 | **+0.00247** | +0.00247 | 0.00035 | [+0.00184, +0.00322] | 200/200 | 0.02514 / 0.02514 |
| canonical @0.025 | +0.00249 | +0.00249 | 0.00037 | [+0.00167, +0.00318] | 200/200 | 0.02502 / 0.02501 |
| N(0,1) @0.02513 | +0.01276 | +0.01270 | 0.00036 | [+0.01218, +0.01359] | 200/200 | 0.02514 / 0.02513 |
| N(0,0.5) @0.02513 | +0.03970 | +0.03971 | 0.00055 | [+0.03827, +0.04069] | 200/200 | 0.02515 / 0.02515 |

- **canonical 结论反转**：精确共同 size 下差值为 **+0.0025，200/200 个划分全正**，与全批 plug-in +0.0024 一致（其网格插值本质上就是同一凸组合）。确定性阈值版的 −0.0007 中心即您诊断的残余 liberality（no-borrowing 评估半批平均 0.0257 vs 0.02513）所致，正文现明写其为 "approximately size-adjusted at best"，其近零中心 "reflected the no-borrowing test's residual liberality, not the absence of a difference"。
- **共同目标**：您指出的"不同 target 把 size 效应混入 dispersion 比较"已消除；S2 注明 round-8/9 的 mixing-specific-target 跑被本轮共同目标跑取代。"dispersion dependence is robust under repetition" 现以共同目标数字支撑（+0.0025 → +0.0128 → +0.0397）。
- **改写位置**：摘要 Results 与 Conclusions（"small but consistent once size is matched exactly in expectation…约 0.25 个功效百分点"）、§3.1（随机化边界写入方法定义）、§4.5（完整段落：方法、结果、对确定性版的定性）、Table 8 caption、讨论第六观察、结论、S1、术语表；引言 (v) 同步（"a small exactly-size-matched mixture-average gain…"）。保留的限定：exploratory、条件于该验证批与估计的 mixing（weight bootstrap [+0.0006, +0.0044]；MC se 0.0007）、mixing 估计误差与 deconvolution 正态形未传播。Figure S1(b) 重绘为"确定性（灰）vs 随机化（绿）ECDF + 两个窄 mixing 的中位数/central range"。

## 2. §4.4 的 triage 证据不是 final design 的证据

**接受，您建议的两条都做了（先重算，再明确定位）。**

- **定位**：§4.4 开头明写本节首先是 **uncapped 架构的 mechanism 分析**（benefit 定义处标注 "both uncapped in this diagnostic"），结尾新增 final design 重算段。
- **重算**（`round10_triage_capped.py`；post-cap mass = min(1−λ₀, 0.5)，产生 281 行 0.5 tie atom；benefit = NLL_EB − NLL_capped）：average-rank Spearman **+0.136**（bootstrap CI [+0.064, +0.207]；分窗口 +0.144/+0.151/+0.127）。分组（atom 内 tertile 成员归属任意，故 0.5 atom 单列为一组；非 binding 行按其中位质量二分）：低组 +0.005 [−0.004, +0.014]、中组 −0.051 [−0.086, −0.016]、**0.5 atom −0.074 [−0.121, −0.025]**——顶组由 uncapped 的噪声（−0.031 [−0.116, +0.062]）变为显著，因为 cap 削掉了 over-borrowing 尾部；严格 tertile（稳定并列破序）+0.003/−0.052/−0.075。选择比较（同 capped 打分）：atom（n=281）+0.074 [+0.027, +0.120] vs I≥8 规则（n=291）+0.080 [+0.040, +0.120]——learned triage 仍持平、不超过手工规则。**复现**：uncapped 四分位数 0.15/0.36/0.65、ρ +0.16 [+0.08, +0.24]、分窗稳定（0.16/0.16/0.16）、tertile deltas 在 tertile 边界惯例内复现（±0.002）。
- **归属**：结论中 "demonstrated per-query value is triage" 现直接引用重算数字（"recomputed under the cap: correlation +0.14; graded group deltas +0.005/−0.051/−0.074, the top group significant"）；§4.4 末句明写 final design 的 triage 主张以重算数字为准、uncapped 数字为架构 mechanism diagnostic。
- **Figure 6**：caption 改 "(uncapped-architecture mechanism diagnostic)"、注明 final design 在打分时把历史质量 cap 在 0.5，图内加 0.5 cap 虚线（重绘）。
- **Figure 5**：图例局部字典（第九轮全局改名漏掉的一处）改 **"Selective (uncapped ablation)"**；caption 明写比较的是 uncapped selective 并给出 final design 在 Table 6 的 KS 0.097；图内 suptitle 同步（重绘）。

## 小问题

- **Table 8 caption**："95% split interval" → "central 95% split-quantile range"（并更新为随机化结果的表述）。
- **Table S2**：补上 `Wins₅` 列（−0.273/−0.273/−0.016/−0.009/−0.018），与 caption 一致。
- **Supplement S1**："forest Figure 13" → "forest Figure 2"（Table 2 编号核对无误）。
- **Figure 3 caption**："the only design-time prior below EB" → "the two selective variants --- the final design (capped, 2.877) and its uncapped ablation (2.872) --- are the only design-time priors below the intercept-only EB reference"（2.877 与图内标签及 Table 6 行一致）。

## 其他

- 页数：正文 57 页、补充 16 页（不变）。
- 验证：`scripts/verify_round10.py` **77 项断言全部通过**（随机化 cross-fit 全部数字、eval size、triage 重算与复现、caption 数字、措辞存在/缺失）；两文档 0 undefined references、0 处 ≥10pt overfull（补充材料 0 overfull）。

## 主要修改位置速查

| 修改 | 位置 |
|---|---|
| 随机化边界 + 共同目标重算（结论反转为 +0.0025 全正） | §3.1、§4.5、摘要、Table 8 caption、讨论、结论、S1、术语表、引言 (v)、`round10_randomized_crossfit.json` |
| "approximately size-adjusted" 定性确定性阈值版 | §4.5、Table 8 caption、S1、S2 |
| dispersion dependence（共同目标、全正） | §4.5、S1、S2、Figure S1(b)（重绘） |
| §4.4 mechanism 定位 + final design 重算段 | §4.4、结论、`round10_triage_capped.json` |
| Figure 5/6 身份标注（重绘） | `build_main_figures_selective.py`、图 5/6 caption |
| Table 8 / Table S2 / S1 Figure 2 / Figure 3 caption | 各处 |

*数字来源：`results/tables/round10_randomized_crossfit.json`、`round10_triage_capped.json`；全部引用值经 `scripts/verify_round10.py` 程序化对账。*
