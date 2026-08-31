# 第三轮逐条审稿回应

**稿件**：Retrospective predictive calibration of oncology trial similarity for Bayesian historical borrowing
**本文档**：对第三轮意见（Major 1–6、Minor 1–10）的逐条回应。章节号、表号指本轮修订稿（48 页主稿 + 7 页补充材料）。

---

## 总览

六条 Major 全部接受，其中四条（1、2、3、4）以新计算落实：阈值重校准与二维冲突网格研究、借用上限变体、严格 ORR 子集主分析重跑、EC 锚失配电池。两条（5、6）以明确的证据边界声明与探索性降级落实。本轮最重要的结构变化：**论文的真实数据证据改为两层报告，主分析 = 注册库可得性代理 × 严格 ORR 估计对象，其结果为"无收益、且可得性一致训练的模型在借用发生处有害"；全部有利结果降为"假设性完全信息上界"**。频率学安全性主张收缩为"点态缓解，非验证的强控制"。

---

## Major 1：“SAM restores type-I error control”不成立

**完全接受，措辞与实质都改了。**

- **措辞**：摘要、§4.5、讨论、结论、图 F8 标题、S1 中所有 "restores control" 改为 "mitigates (but does not eliminate) the inflation"，并在每处标注 worst case 0.070 = 名义值 2.8×。
- **实质（全部按建议完成，§4.5 新增 "Calibrated designs" 段 + 新表 tab:calibrated）**：
  - **二维冲突网格**：预设网格 = 均匀 shift {0–1.5} × 全冲突 + 部分冲突格 f∈{0.25,0.5,0.75}×shift{0.5,1.0,1.5}（DGM 扩展经比特级种子不变性验证；全冲突格复用已发布种子并复现 round-2 拒绝率）。逐复制后验概率全部存档，任意阈值的拒绝率可离线计算。
  - **最坏情形报告 + 阈值重校准**：τ* = 网格最坏 type I ≤0.025 的最小阈值。关键发现：**独立 selective 无 τ*（0.975–0.9995 内无解）——阈值重校准救不了它**；selective+SAM 的 τ*=0.9945，功效从 0.594 跌到 0.388。
  - **借用上限**：新增 cap0.5 变体（历史质量投影 ≤0.5）。它是全比较中最有效的安全机制：未校准最坏情形 0.044（对独立 0.243、+SAM 0.070）。
  - **点态 vs 强控制**：明确区分，并指出字面 0.025 连不借用参照都达不到（离散性，网格最坏 0.038），故给出**参照匹配校准**（最坏 ≤ 不借用自身的 0.038）：capped+SAM 只需 τ=0.979，功效 0.521 对参照 0.536——**强控制下借用的净功效收益归零**。
  - **功效损失同步报告**：表 tab:calibrated 两种校准的功效列齐全。
- **结论措辞**：正文明确 "we do not describe the selective prior, with or without SAM, as having verified frequentist safety"；可主张的是文档化网格上的点态缓解，最有效机制是质量上限。

## Major 2：可得性一致分析应为主要真实数据分析

**完全接受，证据层级已倒转（§4.2 重排、摘要与结论重写）。**

- §4.2 现在开篇即 "Primary analysis: registry-availability proxy, strict-ORR estimand"（并与 Major 3 的严格估计对象合并为同一主分析），其后所有未过滤内容置于 "Hypothetical full-information upper bound" 段下，表 3 与图 3 说明均标注 hypothetical、"carries no decision-time evidential weight"。
- 摘要 Results 段第一句即主分析结论：**no demonstrated predictive gain on decision-time-available data**；结论段按证据权重顺序重写。
- **术语**：全文 "strict availability replay" 改为 "registry-availability proxy"，并写明其两个缺陷：trial 级日期无法证明具体终点行当时已以现有形式发布；快照非历史版本（以审计发现的 4/12→4/13 漂移实例作证，Minor 10 同步落实）。
- **文献渠道**：按要求不再用"文献早于注册发布"来缓和结论——§6.2 明确写 "we have not implemented or validated a literature-retrieval pipeline, so we treat this strictly as an unimplemented extension and do not use it to discount the null primary result"。
- ensemble 等有利结果的定位见 Major 6。

## Major 3：终点混杂威胁主要 estimand（选择路径 1）

**接受路径 1，已完成规范化与主分析重跑。**

- **规范器**：规则式终点规范器（标题级；排除规则先于纳入规则；未匹配即排除）。**以您的 80 行审计为验证集**：74 条无歧义行上准确率 98.6%（严格 ORR 灵敏度 100%、非严格特异度 91.7%）；唯一 FP 恰是标题-定义不匹配案例（ORR 标题、PFS≥6mo 定义）——标题级规则的已知盲区，如实写为 documented residual（描述字段未在当前抽取中保留，双人独立判定无法实施——单审计者局限已声明，见 Major 6）。语料结果：1,028/1,407 查询（73.1%，与审计加权纯度估计一致）、84.0% 组件为严格 ORR。§2.11 新增 "Endpoint canonicalisation and the primary estimand" 段。
- **主分析重跑（严格子集 × 可得性）**：严格窗口 128/155/306，as-of-cutoff 训练池 221/336/444。结果：合并 Δ vs EB **+0.035 [+0.007,+0.067]，借用发生子组（n=119）+0.173 [+0.037,+0.324]、胜率 38%——可得性一致训练的模型在借用处显著有害**；冻结的全量训练模型在同一过滤下中性（−0.002 / 子组 −0.011）——伤害源于训练池饥饿，已如实归因。
- **严格子集上界敏感性**：重训严格主分析（不加可得性）合并 −0.034 [−0.074,+0.007]；冻结模型严格过滤 −0.033 [−0.068,+0.001]——**上界层的模式不因终点混杂而改变**（正好回应"count-only≈0 提示正向结果可能来自弱质量记录"的担忧：严格化后方向与幅度保持）。
- **strict-endpoint sensitivity** 即上项，已与 count-only 并列。CR/PR/CBR/pCR 的处理规则在规范器中显式列出并随代码发布。

## Major 4：EC 的 EB anchor 过于理想化

**接受，扰动电池已完成（§4.7 新段 + 表 tab:ec-anchor）。**

按建议逐项：均值偏移（±0.4 logit，精度保持）、精度高估/低估（×4、×1/4）、无信息（flat）、对照率异质性（新世界：真对照率 logit SD 0.3）、EB 超参不确定性（以扰动族覆盖）。结果：

- 均值低 0.4 → type I 0.104–0.126（全 drift）；均值高 0.4 → type I ~0.001 但功效 0.229（低于 internal-only）；
- 精度 ×4 → 0.033（异质世界 0.076）；**异质性本身把 canonical 锚抬到 0.041–0.049**——同质世界确实美化了它；
- 精度 ×1/4 是唯一全情景安全（≤0.029 含异质）且有用（功效 0.433 对 0.237）的变体——建设性结论：混合对照设计应下调边缘锚权重而非信任其拟合精度；
- selective 行与 eb_only 行处处重合——正文明确："the EC safety is the anchor's, not the method's"，不再作为方法级安全性证据（摘要同步）。

## Major 5：完整方法的真实数据证据缺失

**接受，证据边界已明写（讨论新增第五条 bounded observation）。** 原文陈述：设计时 selective 组件有真实数据预测评价（其主分析结果为 null-to-harmful）但无独立频率学安全性（0.243 且无阈值可救）；selective+SAM 有模拟证据（点态最优功效但仅缓解、强控制下净收益归零）且**原理上不能**做同一套 out-of-sample NLL 评价（SAM 消费被评分结局）；"No single evidence stream currently supports both practical gain and inferential safety for the full method at once; we present the two partial validations side by side and do not combine them into a claim about the complete pipeline."

## Major 6：验证独立性与审计推断

**接受。**

- **ensemble 降为 exploratory**：预登记发生在修订周期内部、非独立前瞻注册——正文、摘要、S1 全部改标 exploratory，且限定 upper-bound tier；更可信验证路径（新 registry vintage / 修订后新增记录 / 完全未参与选择的外部数据）已写入 §2.11 provenance 段与 §6.2。
- **设计加权审计估计**：按各层抽样概率（donor 层：top-shared 10/10 普查、百分比层 32/404、计数层 18/261；query 层 10/395、10/380）与有限总体校正给出语料级估计：donor 四项数字完整性联合 93.5% [87.1,100]；ORR 定义 donor 75.9% [64.6,87.1]、query 70.4% [52.7,88.1]（与规范器独立得到的 73.1% 相互印证）。原始样本率与加权估计并列报告（§6.1）。
- **审计者描述如实化**："expert audit" 改为 "single-rater author audit"（单一作者审计者、对照当前注册库、无独立双人提取与仲裁），§6.1 与 §6.2 一致。

---

## Minor 1–10

1. **图 3 图内 "leakage-free" 字样**：已重绘（图内文字改 "outcome-fit temporal split (past-only refits)"，caption 加 upper-bound 标注）。
2. **图 2 横轴 "clean test split"**：已重绘为 "held-out split"。
3. **"credible-interval coverage"**：真实数据校准处改为 "prediction-interval coverage (equal-tailed intervals of the posterior predictive distribution)"；模拟中对 θ₀ 的覆盖保留 credible（对象确为后验可信区间），两处对象不同、术语现已各自正确。
4. **表 6 "Prior ESS" 列**：改为 "Nominal borrowed size"。
5. **"effective sample-size discounts" 残留**：全文再扫描，§6.1/limitations 无残留（仅 Morita 参考文献标题保留原文）。
6. **"no design-time method removed inflation"**：改写为逐情景精确表述——最坏 drift（−1.2）下所有实际借用的方法超过 internal-only；不借用的方法（EB-only/selective）与参照一致（abstention 而非成功选择），部分格点低于 0.025 属点态现象，并接入锚电池的限定。
7. **表 5 shift=1.5 功效列**：已补全，并在 caption 说明高冲突下 alt 世界拒绝率部分是借来的偏倚（selective 0.836 需与 null 列同读）。
8. **θ_obs 的取值范围**：已写明——偏倚在概率尺度上相加后 clip 到 [0.01,0.99]（典型偏倚幅度 ≤0.1，clip 很少触发），§3.1。
9. **block bootstrap 近似性**：已声明 "approximate dependence adjustments, not exact"；新增第二方案敏感性（top-2-donor 并集：58 块、最大块 89%——退化，其区间 [−0.139,−0.017] 仅为透明起见展示、不作证据）；明确二级共享 donor 的跨块相关未消除、任何区间都不应读作 fully dependence-adjusted。
10. **4/12→4/13 版本漂移的显著性**：已提升——写入 §4.2 主分析段（proxy 缺陷的具体实证）与 §6.2 第三条（"the concrete demonstration that snapshot reconstruction is needed"），不再只在审计段出现。

---

## 仍未完成（如实声明）

1. 注册库 as-of-cutoff 快照重建（§6.2；漂移实例已量化）。
2. 文献检索管线（明确标注为未实现扩展，不用于缓和主分析结论）。
3. 双人独立审计与仲裁（单审计者局限已声明；描述字段级终点定义解析列为下一步）。
4. 独立前瞻验证（新 registry vintage；§2.11 已声明为唯一真 lockbox）。

## 主要修改位置速查

| 修改 | 位置 |
|---|---|
| 校准研究（τ*/参照匹配/上限/二维网格/功效损失） | §4.5 新段 + tab:calibrated、摘要、结论、S1/S2 |
| "mitigates" 措辞清扫 | 摘要、§4.5、§6.2、结论、图 F8、S1 |
| 主分析层级倒转 + proxy 术语 | §4.2 重排、表 1/表 3/图 3、摘要、结论、术语表 |
| 严格 ORR 规范器 + 主分析重跑 | §2.11 新段、§4.2 主分析段、S1 三行、S2 |
| EC 锚电池 | §4.7 新段 + tab:ec-anchor、摘要、S1 |
| 证据边界声明 | 讨论第五条 observation |
| exploratory 降级 + 加权审计 + 审计者描述 | §4.2、§6.1、S1 |
| Minor 1–10 | 对应图表与正文小节 |

*数字来源：`artifacts/round3/`（calibration/ 逐复制后验概率、ec_battery/、strict_orr/、audit_weighted.json）；图形脚本断言与种子不变性验证均随代码发布。*
