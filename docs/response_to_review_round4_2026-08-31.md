# 第四轮逐条审稿回应

**稿件**：Retrospective predictive calibration of oncology trial similarity for Bayesian historical borrowing
**本文档**：对第四轮意见（Major 1–7、Minor 1–7）的逐条回应。章节号、表号指本轮修订稿。

---

## 总览

七条 Major 全部接受，其中五条以新计算落实。本轮的结构性变化有三个。

**第一，最终方案唯一化并形式化**：最终方案 = **capped selective prior**（selective 分配 + EB 锚 + 借用质量上限 0.5 + 混合校准阈值），Methods 新增 §2.10（cap 的数学定义、先验理由、"final design" 权威定义段）。SAM **不再是最终方案组件**——在 cap 之下其边际贡献被测量为冗余（最坏情形仅改善 ≤0.004，混合功效损失 1–2 个百分点），降为 uncapped 先验的敏感性分析。这同时回应 Major 1（cap 形式化）与 Major 6（三选一）。

**第二，校准框架诚实化**：撤回 "reference-matched = strong control" 的表述（Major 2 完全正确，该框架已整体删除）。代之以三层结构：(i) 无限制最坏情形框架——引入 Kopp-Schneider et al. (2020, Biometrical Journal 62:361–374) 的一般结果：严格最坏情形 type I 控制下任何借用方法都不可能保有功效收益；我们的扩展阈值搜索以数字复现了这一点（standalone selective 的精确 τ\*=0.9999，届时无冲突功效 0.069——"可控但退化"）。(ii) **经验混合框架**（新的操作性框架）：用语料实测的 donor–query 漂移分布（逐 query 平均漂移，经验 logit 尺度，矩法去卷积：N(+0.12, 1.33²)，两侧网格离散化）作为混合测度，在混合平均 type I ≤0.025 下校准阈值，并附**预先指定的最坏情形硬上限 0.05**。(iii) 阈值选择（2,000 重复选择批）与错误率评估（**全新种子 10,000 重复/格独立验证批**）完全分离，最坏格报告 Clopper–Pearson 95% 置信上界。

**第三，展示层级真正倒转**：Results 现在以 §4.1 主分析开篇（专用主表 tab:primary + forest 主图 F13：按窗口的 n、有可得 donor 数、借用发生数、EB/最终方案 NLL、Δ 与普通及最强-donor 块自助区间），表 2/3、图 2/3 全部改题为 "Hypothetical upper bound I/II"。

在这套诚实框架下，最终方案的有效性证据为（分层表述、全文一致；下列数字均为**独立 10k 验证批**）：**条件于方法前提成立**（漂移 0 至 +1 logit）功效收益 +4.4 到 +5.3 个百分点；**按语料实测漂移分布平均**净收益小而确凿（混合功效 0.5154，对不借用逐复制配对 +0.0066、MC se 0.0007；对纯 EB 锚 +0.0021）；**对抗性漂移下**任何方法都无收益（定理），cap 的作用是把犯错代价压到 ≤0.05（验证批最坏格 0.036，CP 95% 上界 0.039）。真实数据侧：cap 把主分析从"显著有害"（+0.173）转为**中性**（子组 +0.030 [−0.021,+0.079]），并使 forward 与 title-screened 上界层出现**唯一的显著优于 EB 的区间**（−0.041 [−0.063,−0.020] 等）——安全机制在有利层级上不但不收税、反而作为正则化改善泛化。

---

## Major 1：新"最终设计"缺乏闭合验证链

**接受，四项子要求逐一落实。**

1. **cap 的数学定义与实施步骤**：§2.10 形式化——混合先验历史质量 m=Σλᵢ，投影 λᵢ←λᵢ·min(1,c/m)、λ₀←1−Σλᵢ；确定性、设计时、模拟与真实数据打分同一实现、无需重训、保分配限总量。
2. **0.5 的独立依据 + 完整 cap–power–error 曲线**：依据 = 先验承诺（上一轮修订中先行固定）+ 对等原则（历史成分的先验质量至多与代表当前试验语境的锚成分对等，Pocock 精神）；本轮补齐完整曲线（新表 tab:capcurve，c∈{0.1,…,0.9}）：混合功效在 c≤0.5 平台（0.519–0.521），最坏情形随 c 单调上升，c>0.7 违反 0.05 上限；c=0.5 与更小 cap 的区别在**条件功效形状**（有利带 +5 个百分点对 c=0.1 的 +1–2）。我们**不**据曲线事后重调，保留先验承诺值。
3. **capped design 在所有真实数据层级评价**：已全部完成并写入正文——held-out test（−0.025 [−0.059,+0.010]）、forward（**−0.041 [−0.063,−0.020]，显著**；uncapped 为不显著的 −0.027）、retrained title-screened（**−0.049 [−0.074,−0.023]，显著**）、frozen title-screened（**−0.044 [−0.067,−0.022]，显著**）、主分析（合并 +0.006 [−0.005,+0.016]、借用子组 +0.030 [−0.021,+0.079]，块自助一致）；cap 0.3/0.7 邻域敏感性全层级并报。cap 已是正式方法组成（Methods §2.10、术语表 "Final design" 行）。
4. **SAM 与完整方案的证据组合问题**：结构性解决——最终方案不含任何 outcome-adaptive 组件，因此**同一个对象**同时拥有真实数据预测评价（上条）与已知真值 operating characteristics（Major 2 的校准+验证）；不再存在"两条证据流不合成"的问题。SAM 的冗余是被测量的（tab:capcurve：cap0.5+SAM 最坏 0.040 对 0.044，混合功效 0.510 对 0.521），并如实写入 §2.9/§2.10。

## Major 2：阈值校准不能视为已验证的 type I 控制

**接受，三项统计问题逐一修正。**

1. **"no threshold rescues" 的范围限定**：已改为精确表述并扩展搜索至 1——精确最坏情形阈值 τ\*=0.9999 存在，但届时无冲突功效 0.069（不借用参照 0.448）：**解存在但退化**（§4.5 第一层）。同时引入 [28] 说明这不是本方法特有的失败而是借用的一般规律。
2. **Monte Carlo 选择偏倚**：完全按建议执行——阈值与 cap 在 2,000 重复选择批上确定后**冻结**，在全新种子基（20260901）的 10,000 重复/格独立验证批上重评（44 格 × 最终方案与全部参照）；报告逐格 type I、最坏格及其 Clopper–Pearson 95% 上界（点态与 Bonferroni 同时上界两者都给）、混合平均及其 MC 标准误、配对功效差（同批复制内配对，MC 标准误）。验证结果（tab:calibrated，正文只引用这批数字）：最终方案冻结 τ=0.9755 下，漂移加权平均 type I = **0.0258（MC se 0.0005）**——校准后的不借用参照自身验证为 0.0251，即阈值选择在平均错误率上的不确定性约 ±0.001，这正是设置独立验证批的意义；最坏 null 格 **0.0360**，Clopper–Pearson 95% 上界 **0.0392**（22 个 null 格 Bonferroni 同时上界 0.0416），均在 0.05 上限内；混合功效 **0.5154**，对不借用的**逐复制配对净收益 +0.0066（MC se 0.0007）**、对纯 EB 锚 +0.0021；分漂移 profile 验证 +4.4 到 +5.3 个百分点（漂移 0 至 +1）；n₀ 分层净收益 +0.0/+1.3/+0.7 个百分点（小/中/大）。同批还验证了排除 SAM 的决定（cap0.5+SAM 混合功效 0.5057，**低于不借用 0.31 个百分点**）与 uncapped 的不可用（其 τ_mix 下最坏 0.0948、混合功效 0.339）。
3. **"reference-matched ≠ strong control"**：接受并整体删除 reference-matched 框架及其"strong type-I control"措辞。正文明确：在 0.025 真校准（最坏情形）下 cap+SAM 功效 0.435 < 不借用 0.448——与 [28] 一致，任何借用方法在该框架下都无净收益；最终方案的功效主张全部转移到经验混合框架，并逐处声明"average control under an empirically grounded mixing measure plus a bounded worst case（0.05 上限）——不是任一单点不利漂移下的严格控制"。

## Major 3："audit-validated strict ORR"表述过强

**接受。**

- **更名**：全文 "strict-ORR estimand/subset" 的操作对象改称 **audit-informed, title-screened ORR subset**（正文、表 1、主表、术语表、S1）；"strict ORR" 仅保留为目标 estimand 概念与规则类别名。
- **Resubstitution 如实声明**：§2.11 明确写出规则曾按同一批 80 行审计迭代开发，98.6% 为 **apparent (resubstitution) estimate、likely optimistic**；分层过采样、单审计者、歧义行排除、仅标题、已知 title–definition mismatch 假阳性逐项列出。
- **73.1% vs 70–76% 不再称相互验证**：改为 "internally consistent…not independent corroboration"（§6.1 与 §2.11 两处）。
- **新的样本外验证（本轮新计算+新人工审计）**：规则冻结（脚本 SHA-256 记入 key 文件与 S2），从**未被首轮审计触及的试验**中按角色×冻结预测分层抽 60 行，**预测对审计者隐藏**（盲式判定），由同一审计者对照注册库完整 outcome 描述判定。结果（2 行判定不能确定、按预设排除）：58 行上**准确率 82.8%（精确 95% CI 70.6–91.4；语料分层加权 81.9%）、灵敏度 85.7%、特异度 80.0%**——**证实并量化了 resubstitution 的乐观偏差**（98.6% → 82.8%）。PPV = **title-screened 子集自身的实测纯度 = 80.0%（CI 61.4–92.3）**：系统性偏倚终点族（pCR/PFS 定义/DCR/分子学）被整族移除，但 "response rate" 标题下的 PSA、骨扫描、abscopal、CR-变体复合终点仍会漏入，保守排除的 "clinical response" 标题偶尔藏着真 CR+PR。**我们刻意不按这些错误重调规则**（那会重启 resubstitution 循环）；冻结规则连同其实测样本外工作点就是本文 "title-screened" 的定义。全部数字写入 §2.11 与 §6.1 新段（错误解剖逐条列出），两份审计表与打分脚本随代码发布。双人独立判定与描述字段级解析仍声明为未完成工作（限制节）。

## Major 4：证据层级只在文字上倒转

**接受，展示层级本轮真正倒转。**

- §4.1 = 主分析（开篇即是）：专用主表 tab:primary（按窗口 n=128/155/306、有可得 donor 7/28/84、借用≥1% 质量 5/19/61、EB 与最终方案 NLL、窗口 Δ 与 CI）+ 合并行（最终方案/uncapped 消融/frozen 对照，普通与最强-donor 块自助区间并列；块结构 534 块、最大块 7）+ forest 主图 F13。表 2/3、图 2/3 改题 "Hypothetical upper bound I/II"，图 2 顶栏文字同步改为 upper-bound 标注。
- **主分析的依赖敏感性**：已加（上表块自助列；主分析行的块结构远好于全量层级）。
- **"training-pool starvation" 从推断升级为机制实验**：(i) 可得性一致模型 4 个新种子重训——伤害方向全部复现（子组 +0.102 到 +0.150，原种子 +0.173）；(ii) **等 n 下采样对照**：全量 title-screened 池随机下采样到可得性池同 n（221/336/444）训练、在同一主分析打分行上评价——三次抽样的借用子组伤害 **+0.100 / +0.162 / +0.131**，与可得性一致训练的 +0.10–0.17 同量级：**同等规模、无可得性限制的池产生同样的伤害，伤害追踪的是池规模而非可得性构成**；(iii) 规模剂量曲线（25%/50%/100% 全量池，trainable 107–179 / 214–356 / 432–711）：子组 **+0.219 → +0.198 → +0.080**，全语料 canonical 模型 −0.011——单调走向中性。归因由此从"推断"升级为"机制实验证明"：小训练池使学习分配过度自信，cap 限制其后果，注册库增长消除其原因（§4.1 新段 "Why the uncapped ablation misfires"）。

## Major 5：核心方法学主张需重新定位

**接受其证据学实质，定位按"分层主张"而非"纯警示"处理**（与作者沟通后确定：保留建设性定位，但一切有效性主张只在其被证明的框架内提出）。

- **架构能力 ≠ 实证发现，已在贡献列表显式分级**（引言 (ii)）：三个决策解耦是架构性质；数据支持的是 set 级决策（是否借、总量多少、cap 约束下）；**donor 级判别力未被实证**——原"均被独立有效学习"表述撤回。
- 审稿人列举的六项模拟事实全部在正文明文承认（per-source discount 近常数；prospective signal 不改善判别；一致漂移下所有设计时方法 ROC-AUC<0.5；可学习总量致 0.243；无限制真安全校准下无净收益 [28]；决策时数据 null-to-harmful——最后一条现于 capped 方案下修正为 null-to-neutral）。
- 论文现在的有效性主张即第六条 observation 的三层分级（讨论节）：条件带 +5 点／混合平均小而正／对抗框架无人有收益。互补的警示性贡献（漂移分布实测、可得性层级、终点混杂、锚归因）全部保留并前置。

## Major 6：SAM 同时被称为 optional 和 required

**接受，矛盾从根上消除**：最终方案三选一的答案是**第三种的修正版——capped selective + 校准阈值，且不含 SAM**。§2.9 开头改写（"a named sensitivity analysis for the uncapped prior…not part of the final design"）；§2.10 "The final design" 段为全文唯一权威定义；讨论中 "control is restored only by the anchored SAM companion" 残留句已改写（改为：无伴随物恢复严格控制；SAM 仅缓解 0.070；有界性来自 cap 0.044≤0.05；平均控制来自混合校准）；限制节 "required companion" 残留同步改写；术语表 SAM 行注明 "not part of the final design (the cap supersedes it)"。全文 optional/required 语义唯一化。

## Major 7：EC anchor 分析不是超参数不确定性传播

**接受。**

- **措辞**：全文该分析仅称 anchor stress test / misspecification battery；结论限定为"EC 世界的良好表现来自特定 anchor，且对 anchor 错设高度敏感；不支持 selective 方法级 EC 安全性"（§4.7 与讨论第四条已是此表述，本轮再清扫）。
- **新增真正的抽样不确定性传播**：对 EC 训练抽样（300 个对照臂）做非参数 bootstrap（B=500；重拟合精确复现已发布锚，maxdiff=0）——锚的 logit 均值 95% 抽样范围 **±0.06**、精度比 **0.70–1.24**；电池扰动（±0.4 logit、×4/÷4）分别是其约 **7 倍**与**数倍到一个量级**，即电池远远覆盖抽样不确定性。再以**逐复制随机抽取 bootstrap 锚**的方式将抽样不确定性传播进 operating characteristics：null 网格 type I 0.011–0.019（canonical 0.010–0.019）、异质格 0.036–0.044（canonical 0.041–0.049）、功效同量级——**全部在 Monte Carlo 精度内不变**。结论（正文 §4.7 原句）：锚拟合抽样不确定性不是风险所在；系统性错中心、过拟合精度与未建模的对照率异质性才是，而这些不是"传播"意义上能被 account for 的对象——电池对它们的覆盖以 stress test 的名义呈现。

---

## Minor 1–7

1. **"expert audit" 残留**：表 1 行、§4.2 文内、S1 状态列全部改为 single-rater author audit 措辞（§6.1 的定义段本就如此）。
2. **§6.2 "effective sample-size discounts" / §6.3 "reported ESS"**：分别改为 "borrowed-size discounts" 与 "the weighted nominal borrowed sample size reported as bookkeeping (not a Morita-style prior ESS)"。
3. **图 4 caption 与图例不符**：caption 改为只声明图中实有的三条（selective、EB、weak-only），fixed-budget 指向表 4；不再声称图中比较了 fixed-budget。
4. **"all methods over-cover" 与表 4 矛盾**：改为精确表述——离散预测分布的保守性仅对 well-specified 预测律成立；表中 optimistic 先验实际 under-cover（如 0.384/0.936），正文与表 4、图 4 caption 三处一致修正。
5. **§2.8 "Sections 4.1, 4.2, and 4.2"**：重写为按层级命名（主分析节、两个上界节、校准分析位置），消除重复引用；重排后编号自动更新。
6. **图 3（forward）图内标题仍突出 "matches or exceeds EB"**：已重绘——图内顶栏改为 "Hypothetical full-information upper bound: forward validation (availability-consistent primary analysis reported separately)"。
7. **摘要过长**：已大幅压缩重写（原 ~3 页 → 约 1 页内），绝大多数数字移入正文；保留主分析结论、最终方案定义与两个框架的关键 operating characteristics。

---

**编号说明**：Results 重排与两幅新图（主分析 forest、混合校准 profile）使渲染后的图表编号整体后移（如原"图 3"forward 图现为图 4；主分析表为新表 2）。上文与速查表按新稿编号与文件名（F1、F2、F13、F14 等）双轨标注。

## 仍未完成（如实声明）

1. 注册库 as-of-cutoff 快照重建（漂移实例已量化，§6.2）。
2. 文献检索管线（未实现扩展，不用于缓和主分析结论）。
3. 双人独立终点判定与描述字段级解析（样本外单审计者盲式验证已补，双人仲裁仍缺）。
4. 独立前瞻验证（新 registry vintage 为唯一真 lockbox）。
5. 混合校准的 mixing measure 来自本语料自身（限制节已声明；per-shift profile 全部公开以便读者按自己的漂移先验重加权）。

## 主要修改位置速查

| 修改 | 位置 |
|---|---|
| cap 形式化 + 最终方案定义 | §2.10（新）、术语表 |
| SAM 降为 uncapped 敏感性 | §2.9 开头、§2.10、讨论、限制节、术语表 |
| 不可能性定理 + 精确 τ\* + 退化论证 | §3.1 校准框架段（新）、§4.5 第一层、[28] |
| cap–power–error 曲线 | §4.5 + tab:capcurve（新） |
| 经验混合框架（漂移分布→权重→τ_mix→0.05 上限） | §3.1（新段）、§4.5 第三层、F14（新图） |
| 独立 10k 验证批 + CP 上界 | §3.1、§4.5 + tab:calibrated（重做）、S2 |
| n₀ 分层 OC | §3.1、§4.5 |
| Results 重排 + 主表/主图 | §4.1（新）+ tab:primary + F13、§4.2/4.3 改题 |
| capped 全层级真实数据 | §4.1、§4.3 "final design on the upper-bound tiers" 段 |
| 饥饿机制实验（种子/等n/剂量） | §4.1 + S2 |
| title-screened 更名 + resubstitution + 样本外盲审计 | §2.11、§6.1、术语表、S1 |
| EC stress-test 措辞 + bootstrap 传播 | §4.7、S2 |
| Minor 1–7 | 对应位置 |

*数字来源：`artifacts/round4/`（capsweep/、negcells/、validation/、calibration/、real_capped/、starvation/、ec_boot/、drift/、audit/）；cap=0.5 列与第三轮发布值比特级一致性已验证（calibration JSON `cap50_consistency`）。*
