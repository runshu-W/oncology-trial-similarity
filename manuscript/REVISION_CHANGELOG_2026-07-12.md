# 修订变更记录 — manuscript_full_draft_pharm_stats.tex

**日期:** 2026-07-12
**修订依据:** `overleaf_manuscript_package/manuscript_pre_submission_review_pharm_stats.md`（投稿前预审，2026-07-09）、`PROJECT_REVIEW_publication_readiness.md`、`TRACKB_improvement_plan.md`，以及新增两篇文献（FDA 2026 贝叶斯草案指南、Externally Controlled Trials 因果综述）。
**原则:** 逐条、可追溯、不做一次性大改。语气软化类修改（预审 Edit 1–10）在本 `.tex`（2026-07-09 18:37 版）中**已经落地**，本轮聚焦仍未解决的 must-fix。

---

## 手稿 `.tex` 修订（overleaf_manuscript_package/manuscript_full_draft_pharm_stats.tex）

### R1 — 完整重写参考文献表（预审 Section C.6 / J「must fix」）
- **改了什么:** 将 11 条含 `TODO-verify`、缺 DOI/卷期页的草稿参考文献，替换为 18 条核验过的条目；补全全部卷/期/页/DOI；删除所有 `TODO-verify`。
- **新增条目:** ref12 Trial2Vec (Wang & Sun, EMNLP Findings 2022)、ref13 Bio\_ClinicalBERT (Alsentzer et al. 2019)、ref14 SECRET (arXiv:2505.10780, 2025)、ref15 RBesT (Weber et al. 2021, JSS)、ref16 FDA 2026 贝叶斯方法草案指南、ref17 Ji et al. 2026 (arXiv:2601.14701)、ref18 Zhu et al. 2026 ECT 因果综述 (arXiv:2605.03282)。
- **为什么:** 参考文献不完整是投稿前硬阻塞；补引用是预审明确要求（SECRET/Trial2Vec/ClinicalBERT/RBesT）。
- **诚实标注:** ref14(SECRET) 与 ref17(Ji et al.) 仅从 arXiv 摘要页确认了标题/编号/年份，完整作者名单需作者本人从 arXiv 记录誊录（已在 bibitem 内以方括号标注）。其余 16 条元数据均已核实。

### R2 — Introduction 发展 prior work + 嵌入监管动机（预审 Section F「prior work underdeveloped」；新文献 Step 5-#1/#2）
- **改了什么:** 把原来笼统的「[1--7] ... [8, 9]」两句，扩写为按先验族命名的段落（power prior [3]、commensurate [6]、MAP [4] 及 robust 混合 [5]、ESS [7]、heavy-tailed/self-adapting [10,11]），并新增：FDA 2026 贝叶斯草案指南 [16,17] 作为「which historical trials 可借用」这一选择问题的监管动机，以及 ECT 因果综述 [18]。第二段给检索方法补引用（Trial2Vec [12]、SECRET [14]、ClinicalBERT [13]）。
- **为什么:** 预审指出 prior work 太薄；Step 5 要求把 FDA 2026 作为「监管动机」写进 Introduction、把 ECT 综述作为方法学引用。

### R3 — Methods Stage 1 检索后端补引用
- **改了什么:** ClinicalBERT-style embeddings [13]、Trial2Vec-style [12]、SECRET-style [14]；「not a complete reproduction of the SECRET method [14]」。
- **为什么:** 命名的方法必须给出出处（预审 must-fix）。

### R4 — Methods SAM 小节补引用
- **改了什么:** 「a conflict adapter in the style of the self-adapting mixture (SAM) prior [11]」。
- **为什么:** SAM 是核心方法组件，需引用原文（Yang et al. 2023 Biometrics）。

### R5 — Results robust-MAP/RBesT 补引用
- **改了什么:** 「a tuned robust meta-analytic-predictive (MAP) prior [5] in the style of the RBesT tools [15]」。
- **为什么:** 与领域标准方法对比时需给出方法与软件出处（预审 must-fix）。

### R6 — Discussion「Regulatory and practical implications」合规对齐（新文献 Step 5-#1 的 Discussion 落点）
- **改了什么:** 将框架各组件逐一映射到 FDA 2026 草案指南要求（预设成功准则/决策阈值、先验论证与 ESS、仿真 OC 含 I 类错误、先验敏感性与 prior-data conflict、可复现），并引用 [16,17]、ICH E9(R1) [9]、外部对照指南 [8]、ECT 因果综述 [18]。**诚实用语:** 明确写成「retrospective methodological *alignment with*, not a demonstration of *compliance under*, that guidance」。
- **为什么:** Step 5 要求评估 FDA 2026 作为「合规对齐」写进 Discussion；同时守住预审强调的 claim 边界（不得暗示监管就绪）。

**编译验证:** `latexmk -pdf` 成功（exit 0，24 页，无致命错误，全部 `[1]`–`[18]` 编号与 thebibliography 自动编号一致）。

---

## 代码修订（oncology-trial-similarity/scripts/run_borrowing_simulation.py）

### C1 — borrowing simulation 增加 Monte Carlo 标准误（预审 Section J「Add OC simulation design details: ... Monte Carlo SE」；TRACKB 计划 G3）
- **改了什么:** `summarize_replicates` 新增 `bias_mcse / mean_ci_width_mcse / mean_pr_success_mcse / go_probability_mcse / coverage_like_mcse` 五个字段（比例量用 sqrt(p(1-p)/n)，均值量用 sd/√n）；markdown 摘要表在 GO 和 coverage 列显示 `值 ± MCSE`。
- **为什么:** 让 Track A 模拟报告蒙特卡洛不确定性，回应「500/1000 次单点、无 MC SE」的评审意见。
- **兼容性:** 纯加列，原有字段、种子、数值**逐一不变**；`replicates=250` 的可复现性测试与其余 21 个测试全部通过。

---

---

## 第二批：投稿前缺口修复（同日追加）

### 4(a)1 — 重生成并归档 n=1407 借用头对头表 + 修 `nan`
- **根因**：存档 `results/tables/borrowing_baseline_summary.csv`（6/10）是**陈旧** run（n=1414、`two_head_trained` mass=`nan`），早于 7/8 的 checkpoint 修复；手稿 Table 1（n=1407、genuine checkpoint）来自其后一次未回写的重跑。`nan` 源于 `learned_nll_rows_from_csv` 把 historical_mass 设 None，代码后来已加 `_optional_mean`/`_format_table_float` 处理但 CSV 未重生成。
- **做法**：用 genuine examples 文件重跑 `run_borrowing_baseline_comparison.py` 与 `run_robust_map_baselines.py`（均 `--examples-jsonl artifacts/twohead_deepsets_examples_with_model.jsonl`，n=1407）。结果与手稿 Table 1 **10 行逐一精确吻合**（two_head_sam 2.9328/cov0.988 … robust_map_w0.9 3.4784/cov0.897），**无 nan**。
- **归档**（`results/tables/`，旧文件备份至 scratchpad）：`borrowing_baseline_summary.csv`、`borrowing_baseline_nll.csv`、`robust_map_head_to_head.{csv,json,md}`（Table 1 真实来源）、重生成的 `table_borrowing_baseline_head_to_head.tex`（10 行、含 coverage）。

### 4(a)2 — 归档 5,000-iter OC sweep + conflict-detection ROC-AUC
- **根因**：手稿 §OC 的 5,000-iter sweep + ROC-AUC（0.54→0.74）已算出（`artifacts/oc_sweeps_twohead/`，7/8，genuine examples，iterations=5000，seed=20260707）但**未归档到 `results/tables/`**。
- **做法**：核对 `oc_conflict_detection.csv`——AUC shift0.5=0.540、shift1.5=0.744，与手稿**精确吻合**；归档到 `results/tables/oc_sweeps_5000iter/`（sweep+detection+λ0 frontier+SE+图+README）。

### 4(a)3 — Pin 依赖
- 生成 `oncology-trial-similarity/requirements.lock.txt`（+ worktree 副本）：fastapi==0.136.1、numpy==2.4.4、pypdf==6.11.0、python-multipart==0.0.29、torch==2.11.0、transformers==5.8.0、uvicorn==0.47.0，及关键传递依赖（tokenizers 0.22.2、safetensors 0.7.0、huggingface_hub 1.14.0、scipy 1.17.1 等）。对应 Data & Code Availability 承诺的 `requirements.lock.txt`。

### 4(a)4 — 同步三份手稿副本
- **重复 .tex**（`overleaf_manuscript_package 2/`）：原为更旧快照（仍带未软化的 "withholds…detects" 表述）→ 用修订后主稿覆盖，`supplement.tex` 一并复制。
- **源稿 .md**（worktree `docs/manuscript_full_draft_pharm_stats.md`）：References 段补全为 18 条核验版（删 TODO-verify）、Introduction 同步监管动机+prior work+检索引用。
- **说明**：.md 在最新结构性改动（数据集表、附录移 supplement）上仍落后于 .tex，建议投稿前从最终 .tex 回生成 .md 源稿。

### 4(b)1 — F8/F9 处理
- F8/F9 为**孤儿图**（final_v2 README 只记录 F1–F7，主稿不引用），且无 conflict-case 数据 artifact，不能编造 Case 3。→ 两个 overleaf 包中把 F8/F9 移入 `figures/supplementary_unused/`（移出编译路径、保留原图 + README 说明如何将 F9 转为真正的冲突案例）。

### 4(b)2 — 评估数据集汇总表
- Methods §2.9 新增 `Table~\ref{tab:datasets}`：8 行覆盖 Stage1 检索(1470)/借用头对头(1407)/two-head 训练(80-20)/rolling-origin(629,810,1017→785,604,397)/OC 模拟(500+5000)/sham(1407×3)/multi-endpoint(300,38,300,20)/extraction 审计(7173)，列出 size、split/leakage 状态、endpoint 范围。

### 4(b)3 — Claim–Evidence Map 移入 supplement
- 新建 `overleaf_manuscript_package/supplement.tex`（可独立编译，Table S1）；主稿 Appendix B 删除、代之以指向 supplement 的注释。Terminology（Appendix A）保留在主稿。

**编译验证**：主稿 `latexmk` exit 0（24 页、0 致命错误、新数据集表各行已在 PDF 确认）；`supplement.tex` exit 0（2 页）。

---

## 仍未完成（需作者信息或真实发布，非我能凭空生成）
- 作者/单位、Acknowledgements、Conflicts of Interest 占位符。
- Data & Code Availability 的真实 Zenodo DOI。
- ref14(SECRET)/ref17(Ji et al.) 的完整作者名单（需从 arXiv 记录誊录）。
- .md 源稿在数据集表/附录移动上与 .tex 的完全同步（建议投稿前从最终 .tex 回生成）。
