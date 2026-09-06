# Oncology trial similarity — journal placement recommendation

**Manuscript:** Retrospective predictive calibration of oncology trial similarity for Bayesian historical borrowing  
**Type:** biostatistics / pharmaceutical statistics METHODS（ClinicalTrials.gov retrieve–rerank → selective two-head DeepSets mixture prior → EB-referenced predictive calibration + OC simulation）  
**As of:** 2026-09-06（UTC+8）  
**Claim strength assumed（当前稿，非理想未来版）:** 主真实数据分析相对 intercept-only EB reference **largely null**（capped final design 未可检测地更好；uncapped ablation 有害）；贡献定位为 **methodology prototype + harm attenuation（cap）+ evaluation design**，**不是** big predictive win。无 expert borrowability labels。

> 本文不保证录用，不编造 acceptance rate。Impact factor 仅在有第三方公开汇总值时注明，并给出来源 URL；**未核对 Clarivate JCR 原始表**，不作官方 IF 承诺。

---

## Bottom line（一段）

**第一选择仍是 Pharmaceutical Statistics（Wiley / PSI）。** 该刊明确面向制药统计实践、欢迎方法+案例+监管语境下的讨论，且近年持续发表 historical borrowing / robust MAP / adaptive Bayesian borrowing 邻域文章；本稿已按该刊方向写作，而「相对 EB 主分析诚实 null + 无 cap 有害 + OC 显示 harm attenuation」在 industry biostat 读者中可被读成**有用的边界/警示结果**，前提是摘要与 highlights **绝不卖预测大胜**。相对 2026-07 内部排序，**唯一重要下调是 Statistics in Medicine**：在当前 claim strength 下它更吃「方法优势清晰 + 验证更硬」，容易把本稿判成 prototype / insufficient advance；实务第二梯队应插入 **Journal of Biopharmaceutical Statistics**，并把 **BMC Medical Research Methodology** 作为「impact-agnostic、接受诚实方法学+模拟」的开放获取退路。**不要**冲纯 ML venue。

---

## Ranked list（1–5）

### 1. Pharmaceutical Statistics（Wiley）— 首选

- **Fit / 范围:** PSI 官方刊；aims 强调药企全流程统计实践、案例、监管文件讨论与从业者沟通，明确偏好 **methodology + application**，而非纯理论（来源：PSI 介绍页；Wiley issue information 摘要）。受众：industry statisticians 及药研协作方。文章类型：practical papers、case studies、reviews/tutorials。
- **邻域发表证据:** 持续收 Bayesian historical borrowing。例：Zhang et al. EB-rMAP（*Pharm Stat* 2023，doi:10.1002/pst.2315）；Viele et al. historical control review（*Pharm Stat* 2014）；Mukhopadhyay et al. prospectively specified adaptive Bayesian borrowing（*Pharm Stat* 2026，doi:10.1002/pst.70051）。与本稿「robustification / EB 参照 / OC」同生态位。
- **Why（针对*当前*主张）:** retrieve→borrowability→mixture prior 管线、λ vs *a* 分离、capped selective design、相对 EB 的诚实校准、conflict OC——正是该刊读者会问「能不能用、会不会伤」的内容。**主分析 null 在此刊可辩护**，若把贡献写成：（i）可审计的 registry→prior 原型；（ii）uncapped 有害、cap 衰减伤害；（iii）EB-referenced evaluation protocol——而不是「学会了更好的 ORR 预测」。
- **Risks:** （1）篇幅/图表面过满（预审已指出 claim 边界、bib、模板仍要收）；审稿人会直问「不优于 EB，贡献是什么」——须在 Abstract/Highlights 前置回答。（2）DeepSets / embedding 细节过重会被嫌偏 ML——需压成 enabling infrastructure。（3）第三方汇总值称近期刊 IF ≈1.5（journalsearches.com；Resurchify Impact Score ≈1.49）——**声望靠社区与 PSI，不靠 IF**；勿用 IF 叙事。
- **What to change before submit:** 收紧 claim 边界与图表标题（calibration ≠ validation）；补齐参考文献 DOI/卷期页；Highlights 对齐「prototype + harm attenuation + evaluation」；尽量压主文页数、工程细节进 Supplement；明确最终设计 = capped，uncapped 仅 ablation。

### 2. Journal of Biopharmaceutical Statistics（Taylor & Francis）— 强备选

- **Fit:** 生物制药 R&D 统计应用；含 Phase I–III、复杂创新设计、RWD/external controls、Bayesian methods（Scimago / 刊方描述）。文章类型：full/short methods、reviews、会议精选。
- **邻域发表证据:** CA-MAP historical borrowing（doi:10.1080/10543406.2024.2330206）；historical-bias power prior with EB（doi:10.1080/10543406.2024.2429461）；constrained power-prior / type I control（doi:10.1080/10543406.2025.2575940）。Bayesian Analysis 亦有 dynamic borrowing 综述指向同一问题域（doi:10.1214/24-bjps598，*Bayesian J. Pharm. Stat.* 系列语境）。
- **Why:** 若 Pharm Stats 拒稿或周期过长，**叙事几乎不用大改**；对「新 prior 变体 + 模拟 OC」胃口大。诚实 null 可包装为 **robustness / type-I / harm** 结果，与上述 constrained-borrowing 文献同调。
- **Risks:** 比 Pharm Stats 更偏「方法细节」；管线/informatics 过长会被要求砍；若审稿人期待「新方法在真实数据上显著赢 baseline」，null 主分析仍是压力点——需把 **vs uncapped / vs naive borrowing** 的对照推到与 vs EB 同等中心。
- **Before submit:** 一张对照表（MAP / rMAP / EB-rMAP / SAM / 本稿 selective+cap）；代码与 seeds 可复现声明；弱化 Trial2Vec/ClinicalBERT 为附录。

### 3. Clinical Trials（Sage / Society for Clinical Trials）— 备选（需改包装）

- **Fit:** SCT 会刊；aims 强调对临床试验共同体**可推广**的 design / conduct / analysis / synthesis / ethics / regulation 知识；收统计方法，不收普通试验结果（Sage aims 摘要；Wikipedia 范围描述作交叉核对）。受众含 trialists，不只是生物统计师。
- **邻域发表证据:** Neuenschwander et al. summarizing historical controls（*Clin Trials* 2010，doi:10.1177/1740774509356002）是经典 MAP 前史；期刊持续收 design/methodology，但对「registry NLP + DeepSets」不如 Pharm Stats / JBS 自然。
- **Why:** 若把贡献写成「从 ClinicalTrials.gov 到借用决策的**可审计工作流** + 监管口径下的 OC 报告方式 + availability 限制下的诚实评价」，比「新神经网络架构」更贴。诚实 null 可支撑「不要过度借用 registry 证据」的 trialist 信息。
- **Risks:** 深度学习/检索细节过多会被 desk 嫌偏 informatics；需要更强的 trialist 可读叙事；统计新颖性审稿可能冷淡。
- **Before submit:** 重写 Introduction/Discussion 面向 trialists；压缩公式；突出 decision workflow、ESS、conflict、availability proxy 局限；把 DeepSets 降为「learned allocation module」。

### 4. BMC Medical Research Methodology — 开放获取退路（诚实 null 友好）

- **Fit:** OA；鼓励临床试验、meta-analysis、数据质量与统计建模方法；**明确声明不按 perceived impact 做编辑决定**，以科学有效性为准（官方 aims：https://bmcmedresmethodol.biomedcentral.com/about）。
- **邻域发表证据:** 2025 年 MSDB / dynamic borrowing 框架（doi:10.1186/s12874-025-02691-2）；SAM prior 用于 group-sequential medical device（doi:10.1186/s12874-025-02520-6）。说明该刊当前活跃接收 historical borrowing + OC 仿真稿。
- **Why:** 对「主分析 null + 方法学原型 + 完整模拟」最宽容的主流方法刊之一；适合作为 Pharm Stats / JBS 拒稿后的**快速、开放获取**落地，而不必为冲 SiM 硬凑预测胜利。
- **Risks:** APC；在部分 industry 读者中声望低于 Wiley/Sage 统计刊；需接受「OA 方法刊」定位。勿当作第一声望目标。
- **Before submit:** 按 BMC 结构（Background/Methods/Results/Discussion）；强调 reproducibility（code URL）；保留诚实 null，勿改写成虚假胜出。

### 5. Biometrical Journal（Wiley）— 可复现方法学备选 / SiM 条件冲刺的替代

- **Fit:** 生命科学统计方法；要求方法由实际问题驱动并含应用；强调 **reproducible research**（代码/数据作为 Supporting Information，有 RR editor 传统）（doi:10.1002/bimj.200900154；Wikipedia 概述）。
- **邻域发表证据:** Gravestock & Held multiple historical power priors（*Biom J* 2019）；Han et al. normalized power prior inference（*Biom J* 2023）；Okada et al. test-then-pool / type I（*Biom J* 2024）。
- **Why:** 本稿代码与 artifact 重，**可复现性卖点**在此刊比「预测赢 EB」更对口；诚实 null 可作为应用节的真实局限。
- **Risks / 与 SiM 比较:** *Statistics in Medicine*（aims：influence practice；新方法需实质应用或全面评价；排除纯数学；来源 LetPub 转引 Wiley aims / doi:10.1002/sim.6726）近年也发 dynamic borrowing（如 SPx，doi:10.1002/sim.70567），但**对当前 null 主分析更苛刻**——容易要求 expert labels 或更强优势展示。故将 **Biometrical Journal 列为第 5**，**SiM 不进前五主序**，仅作「证据加强后的冲刺项」（见 Strategy）。
- **Before submit（Biom J）:** 打包可运行代码与关键表；压缩 oncology NLP 细节；突出统计贡献与模拟。
- **Before submit（若仍想冲 SiM）:** 至少完成更干净的 forward retraining 叙事，并认真考虑小规模专家标注；把「相对 uncapped / fixed-budget 的稳定改进」做成不可忽视的中心结果——**不要带着当前主张硬投**。

---

## 候选刊速查（未进前五但仍评估）

| 期刊 | 结论 |
|---|---|
| **Contemporary Clinical Trials (Elsevier)** | **相关但非首选。** 有 elastic MAP / historical hybrid design 先例（如 Zhang et al. 2021 CCT；Duan et al. 2021 CCT），偏 trial design / protocol 落地。本稿更偏「借用管线+预测校准+评价协议」；且若同组另有桥接/设计稿瞄准 CCT，不宜挤占同一策略位。可作拒稿后的远备选，需改打 design/operating-characteristics 包装。 |
| **JCO Clinical Cancer Informatics** | **统计新颖性被质疑时的 informatics 退路**（ASCO：clinically relevant biomedical informatics on cancer data/trials）。适合改打 oncology trial similarity / borrowability tooling；需大改受众，OC 深度可能被压缩。不作为统计方法主航道第一落点。 |
| **JCO Precision Oncology** | 有 oncology Bayesian borrowing **比较综述**（doi:10.1200/PO.21.00394），但是临床肿瘤读物，不是本稿这种 pipeline-methods 的自然家。 |

---

## Journals to avoid（及原因）

1. **纯 ML / NLP venue**（NeurIPS/ICML 主会、ACL、KDD、纯 arXiv ML workshop 主场）：贡献不在 retrieval SOTA；主结果也不是 embedding benchmark 胜利。会被按 ML 标准碾压，且丢失目标读者（药统/试验设计）。
2. **高 IF 临床肿瘤主刊**（*JCO* 主刊、*Lancet Oncology* 等）：不是试验结果，也不是诊疗指南级证据；方法学深度放不下。
3. **Biometrics / 过纯理论贝叶斯专刊（作为第一选择）:** 会嫌 applied/pipeline 过重；robust MAP 经典虽发在 *Biometrics*（Schmidli 2014），但不代表本稿这种 registry 管线应首投该处。
4. **把 *Statistics in Medicine* 当「第二自动跳板」且不改稿:** 在当前诚实 null + 无专家标签下，被要求 major rewrite / 拒稿的风险高于 JBS 或 BMC MRM。
5. **Contemporary Clinical Trials 作为唯一策略位（若已有另一篇 CCT 计划）:** 主题重叠与策略挤占；本稿更贴 Pharm Stats / JBS。

---

## Strategy: first choice + backup cascade

1. **先投 Pharmaceutical Statistics**（与当前 `.tex` 一致）。投稿前完成 claim 收紧、bib、Highlights、页数/Supplement 分流——否则不是「venue 错」，是「稿件未就绪」。
2. **若拒稿/desk 理由偏「统计新颖性不足 / 偏软件管线」** → 小改投 **Journal of Biopharmaceutical Statistics**（对照表+压缩 informatics）。
3. **若理由偏「受众/可读性 / 试验共同体相关性」** → 改包装投 **Clinical Trials (SCT)**。
4. **若理由偏「验证不够」** → **不要立刻冲 SiM**；先决定是否补 forward 叙事/专家小样本。需要快速 OA 落地则投 **BMC Medical Research Methodology**；强调可复现则投 **Biometrical Journal**。
5. **证据加强后**（clean forward retraining + 可选专家标注 + 更锐利的 vs-naive 中心结果）再考虑 **Statistics in Medicine**。
6. **统计航道走不通**再转 **JCO Clinical Cancer Informatics**（informatics 重写，非换皮重投）。

**不保证录用。** 同生态位已有 EB-rMAP、SAM、MSDB、CA-MAP 等；本稿差异化必须是 **registry similarity→selective mixture→EB-referenced honest evaluation + harm attenuation**，而不是「又一个 borrowing prior」。

---

## 相似论文落点（placement evidence，节选）

| 主题 | 代表工作 | 落点期刊 |
|---|---|---|
| EB robust MAP / adaptive external data | Zhang et al. 2023 | *Pharmaceutical Statistics* |
| Adaptive Bayesian borrowing（前瞻设计） | Mukhopadhyay et al. 2026 | *Pharmaceutical Statistics* |
| Historical controls 方法比较 | Viele et al. 2014 | *Pharmaceutical Statistics* |
| CA-MAP / power prior EB / constrained borrowing | 2024–2025 系列 | *Journal of Biopharmaceutical Statistics* |
| Dynamic borrowing 综述 | Lesaffre et al. | *Bayesian Journal of Pharmaceutical Statistics*（doi:10.1214/24-bjps598） |
| SPx synthetic prior with covariates | 2025/26 | *Statistics in Medicine* |
| MSDB / SAM+GSD dynamic borrowing | 2025 | *BMC Medical Research Methodology* |
| Power priors（多历史、binary） | Gravestock & Held 2019 | *Biometrical Journal* |
| Robust MAP（经典） | Schmidli et al. 2014 | *Biometrics* |
| Historical controls 综述（MAP 前史） | Neuenschwander et al. 2010 | *Clinical Trials* |
| Oncology borrowing 方法比较 | 2022 | *JCO Precision Oncology* |
| Trial similarity search（无借用） | 2022 | Medical informatics 会议录（非药统主航道） |

**解读:** 「borrowing 方法 + OC」密集落在 **Pharm Stats / JBS / Biom J / BMC MRM / SiM**；「trial similarity 检索」单独走 informatics。本稿二者杂交——**必须以 borrowing/calibration 为主叙事**，检索为输入模块，才能留在药统航道。

---

## Sources（实际检索/抓取）

### 官方或准官方范围页
- PSI — Pharmaceutical Statistics 介绍: https://psiweb.org/publications  
- Pharmaceutical Statistics aims（Issue Information 摘要）: https://doi.org/10.1002/pst.2235  
- Statistics in Medicine aims（Wiley 表述，经 LetPub / issue info 转引）: https://www.letpub.com/index.php?journalid=7639&page=journalapp&view=detail ；https://doi.org/10.1002/sim.6726  
- Statistics in Medicine author guidelines 入口: https://onlinelibrary.wiley.com/page/journal/10970258/homepage/forauthors.html  
- Clinical Trials（SCT）aims（Sage 描述）: https://uk.sagepub.com/en-gb/eur/journal/clinical-trials （部分区域 404；另用 journals.sagepub.com/home/ctj，Cloudflare 拦截）  
- BMC Medical Research Methodology aims（成功抓取）: https://bmcmedresmethodol.biomedcentral.com/about  
- Journal of Biopharmaceutical Statistics（刊方/Scimago 描述）: https://www.tandfonline.com/toc/lbps20/current ；https://www.scimagojr.com/journalsearch.php?q=23035&tip=sid  
- Biometrical Journal RR 政策: https://doi.org/10.1002/bimj.200900154 ；期刊主页 https://onlinelibrary.wiley.com/journal/15214036  
- ASCO / JCO CCI 范围摘要: https://ascopubs.org/authors ；历史 aims 存档 https://web.archive.org/web/20170129090900/http:/cci.jco.org/  

### 邻域论文（落点证据）
- https://doi.org/10.1002/pst.2315 （EB-rMAP, Pharm Stat）  
- https://doi.org/10.1002/pst.70051 （ABB, Pharm Stat 2026）  
- https://doi.org/10.1002/pst.1589 （Viele historical controls, Pharm Stat）  
- https://doi.org/10.1080/10543406.2024.2330206 （CA-MAP, J Biopharm Stat）  
- https://doi.org/10.1080/10543406.2024.2429461 （power prior EB, J Biopharm Stat）  
- https://doi.org/10.1080/10543406.2025.2575940 （constrained borrowing, J Biopharm Stat）  
- https://doi.org/10.1002/sim.70567 （SPx, Stat Med）  
- https://doi.org/10.1186/s12874-025-02691-2 （MSDB, BMC MRM；全文已抓取）  
- https://doi.org/10.1186/s12874-025-02520-6 （SAM+GSD, BMC MRM）  
- https://doi.org/10.1177/1740774509356002 （Neuenschwander, Clin Trials）  
- https://doi.org/10.1111/biom.12242 （Schmidli robust MAP, Biometrics）  
- https://doi.org/10.1200/PO.21.00394 （oncology borrowing comparison, JCO PO）  
- https://doi.org/10.1214/24-bjps598 （dynamic borrowing review）  

### IF / 计量（第三方，非 Clarivate 原表；仅供参考）
- Pharm Stat IF≈1.5（称 2026 JCR 汇总值）: https://journalsearches.com/journal.php?title=pharmaceutical+statistics  
- Clinical Trials IF≈2.4（同上类站点）: https://journalsearches.com/journal.php?title=clinical+trials  
- Statistics in Medicine IF≈2.1（journalmetrics.org）: https://www.journalmetrics.org/journal/statistics-in-medicine  
- Resurchify Pharm Stat Impact Score≈1.49: https://www.resurchify.com/impact/details/21124  

### 本地稿件与既有评审
- `/workspace/oncology-placement/manuscript_full_draft_pharm_stats.tex`（主分析 null / capped vs uncapped / EB reference）  
- `/workspace/oncology-placement/manuscript_pre_submission_review_pharm_stats.md`  
- `/workspace/oncology-placement/PROJECT_REVIEW_publication_readiness.md`（2026-07 排序与就绪度）  

**抓取限制说明:** Wiley Online Library、Taylor & Francis、Sage、ScienceDirect、部分 ASCO 页面在本环境触发 Cloudflare/406，范围文字交叉核验自 PSI、BMC 官方 about、DOI issue information、LetPub 转引 Wiley aims，以及成功抓取的 BMC 全文。缺失官方 HTML **不等于**期刊不存在。
