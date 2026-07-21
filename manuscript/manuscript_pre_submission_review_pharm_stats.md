# Pharmaceutical Statistics 投稿前预审报告

**Manuscript:** `/Users/wang/Documents/New project/overleaf_manuscript_package/manuscript_full_draft_pharm_stats.tex`  
**审查日期:** 2026-07-09  
**目标期刊:** Pharmaceutical Statistics, based on manuscript comment line 2  
**审查范围:** 完整读取主 `.tex` 文件；检查同目录 figures、package 文件和 LaTeX 编译状态。未修改 manuscript 源文件。

## Package Fact Check

- 主文件：`manuscript_full_draft_pharm_stats.tex`，1144 行。
- 同目录未发现独立 `.bib`、tables、supplementary `.tex` 或 Wiley template 文件。
- 参考文献内嵌在 `thebibliography` 中，且有 TODO。
- figures 目录包含 9 个 PDF：`F1_headtohead_nll.pdf` 至 `F9_case_conflict.pdf`。
- 主文实际引用 figures：F1-F7。F8/F9 存在但未被正文 `includegraphics` 引用。
- 本次编译命令：`latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript_full_draft_pharm_stats.tex`。
- 编译结果：成功生成 23 页 PDF；第二遍后 cross-references 解析。主要问题为 overfull/underfull boxes、占位符和参考文献未完成。

## A. 一句话总评

这篇稿件已有 Pharmaceutical Statistics 方法论文的基础，但还不接近可直接投稿；主要卡点是 claim 比证据走得快、结果叙事过满、参考文献和投稿声明未完成，以及图表标题多处把 retrospective predictive calibration 写得像 validation/generalization。

## Manuscript Fact Base

### 核心问题

稿件要解决的问题是：如何从 ClinicalTrials.gov oncology trial records 中发现可能用于 Bayesian historical borrowing 的历史证据，并把 retrieval/reranking 连接到 beta-binomial mixture-prior construction 和 prior-data conflict sensitivity。

### 核心方法链条

当前稿件的方法链条为：

1. ClinicalTrials.gov oncology records parsing。
2. Structured trial summaries，包括 disease context、population、regimen、endpoint text、follow-up/time-frame、eligibility、result availability、endpoint observations、red flags。
3. Stage 1 retrieval：hashing、ClinicalBERT-style embeddings、optional Trial2Vec-style retrieval、SECRET-inspired deterministic section pool。
4. Stage 2 explainable reranking：overall similarity、disease match、regimen match、endpoint match、follow-up match、eligibility match、result quality、red-flag severity、information size。
5. Endpoint extraction：得到 candidate endpoint observation `(y_i, n_i)`。
6. Beta-binomial historical components：`alpha_i = 1 + a_i y_i`, `beta_i = 1 + a_i(n_i - y_i)`。
7. Mixture prior：weak component plus historical components, with `lambda_i` as mixture mass and `a_i` as effective sample-size discount。
8. Two-head DeepSets：分别输出 mixture allocation `lambda_i` 和 per-candidate discount `a_i`。
9. SAM-style prior-data conflict adapter：在 observed pseudo-query outcome 可用时，根据 historical predictive 与 weak predictive 的 ratio 调整历史质量。
10. Evaluation：ORR pseudo-queries、held-out beta-binomial NLL、rolling-origin forward folds、calibration diagnostics、sham controls、simulation OC sweeps、baseline comparisons、feature/weight sensitivity、software tests。

### 数据与评价

- 主证据集中于 ORR pseudo-queries。
- Stage 1 paired retrieval benchmark：1,470 common ORR pseudo-queries。
- Borrowing/calibration comparison：1,407 ORR pseudo-queries。
- Rolling-origin future-fold validation：cutoffs 2020-12-31, 2021-12-31, 2022-12-31；future-fold sizes 785, 604, 397。
- Multi-endpoint check：ORR/CR/PR n=300, DCR n=38, DLT n=300, treatment discontinuation n=20；但这是 borrowing-layer robust-MAP check，不是 full retrieval/two-head endpoint-specific pipeline。
- Case studies：NCT02551432 和 NCT03046953 两个 concordant low/moderate response examples。
- Simulation：known-truth prior-data-conflict sweep，5,000 iterations reported。

### Manuscript 声称的主要贡献

1. A reproducible methodology prototype linking oncology trial retrieval to Bayesian historical borrowing。
2. Distinguishing retrievability from borrowability。
3. Explicit separation of mixture weight `lambda_i` and effective sample-size discount `a_i`。
4. Two-head DeepSets candidate-specific learned allocation/discount model。
5. SAM-style transparent conflict-sensitivity layer。
6. Retrospective predictive-calibration evidence package without expert borrowability labels。
7. Leakage-free rolling-origin forward evaluation for learned and closed-form rows。
8. Sham controls and simulation operating-characteristics evidence。

### 当前证据真正支持什么

当前证据支持：

- 在 ORR pseudo-query 设置中，SECRET-inspired pool 相比 hashing 改善了 selected retrieval/readiness metrics。
- 在同一 ORR retrospective evidence package 中，two-head + SAM 在 evaluated methods 中有最低 NLL 和较好的 selected calibration diagnostics。
- 在三个 rolling-origin future-fold splits 中，per-fold retrained learned rows 和 closed-form rows 均表现出较低 future-fold NLL。
- SAM-style adapter 在 constructed sham-history 和 simulated prior-data-conflict settings 中表现出 conflict sensitivity。
- `lambda_i` 和 `a_i` 的概念区分清楚，统计建模动机合理。

### 当前证据不支持或容易过度解释的 claim

当前证据不支持：

- 任何 individual historical candidate 的 expert-approved borrowability。
- 临床可借用性标签验证。
- prospective clinical validity。
- regulatory qualification。
- real-trial decision recommendation。
- beyond-ORR full retrieval/two-head generalization。
- broadly validated “non-comparable evidence” withholding；目前只是 sham controls 和 simulation evidence。

必须维持的边界是：

> This is a reproducible retrieval/reranking and retrospective predictive-calibration feasibility/prototype framework for prior-borrowing-oriented oncology historical-trial discovery, not a validated clinical decision system.

## B. Submission Readiness Score

**62/100**

这不是编辑决定，只是内部投稿准备度评估。

### 主要扣分项

- **-12:** References 不完整，含 `TODO-verify`，无 DOI/volume/pages，不能投稿。
- **-8:** 作者、affiliations、Acknowledgements、Conflicts、Zenodo DOI 均为占位符。
- **-8:** Claim boundary 多处仍偏强，尤其 generalization、withholding、detection、calibration restoration。
- **-5:** Results 过于密集，多个 evidence layers 混在一起，读者难判断哪个是 leakage-free，哪个是 model-development retrospective comparison。
- **-5:** Multi-endpoint 和 case-study evidence 被写得比实际更强。
- **-4:** Figure titles/captions 多处结论型标题过强。
- **-4:** Appendix 更像内部审查材料，不像投稿稿件附录。
- **-3:** Data/Code Availability overfull 可见，且 release DOI 未完成。
- **-3:** Prior work positioning 太薄，Introduction 和 Discussion 需要补关键文献与比较框架。

### 加分项

- 方法链条完整，读者能看出 pipeline 从 parsing 到 prior construction 的端到端流程。
- `lambda_i` 和 `a_i` 区分清楚，是稿件的统计亮点。
- 已反复声明 no expert borrowability labels、not clinical recommendation system、need expert review。
- 有 rolling-origin forward split、sham controls、simulation OC、robust-MAP baseline、data-quality audit，证据包较丰富。

## C. Major Issues Before Submission

### 1. Retrospective/simulation evidence 被写成较强的 validation/generalization

**位置:** Abstract lines 80-86; Results lines 544-569, 607-653, 712-744, 786-827; Discussion lines 938-951; Conclusion lines 1004-1017  

**问题:** 目前多处使用 “confirmed forward-in-time generalization”, “best calibrated”, “withholds borrowing from non-comparable evidence”, “detects prior-data conflict”, “benefit generalizes beyond ORR” 等结论式表达。

**为什么重要:** 这些表达容易让审稿人认为作者在声称 expert/clinical validation 或 broad external generalization。但当前证据是 retrospective predictive calibration、constructed sham controls 和 simulation evidence。

**建议怎么改:** 全文统一改成限定语：

- `in this retrospective ORR pseudo-query evaluation`
- `in three rolling-origin future-fold splits`
- `under constructed sham-history conditions`
- `in known-truth simulation settings`
- `exploratory borrowing-layer check`

**类型:** 措辞调整 + claim boundary。

### 2. Learned-row retrospective comparison 和 leakage-free generalization 混在同一叙事中

**位置:** Methods lines 436-448; Results lines 478-491; Discussion lines 938-947  

**问题:** Methods 明确说 head-to-head/calibration comparison 中 learned rows include queries that contributed to model development；但 Results 又将其与 leakage-free rolling-origin evidence 合并成整体 superiority narrative。

**为什么重要:** Pharmaceutical Statistics 审稿人会特别关注 leakage、training/evaluation separation、model selection bias。  

**建议怎么改:** Results 应分成两层：

1. **Retrospective model-development comparison:** 1,407 common ORR pseudo-queries，主要用于 method behavior and calibration diagnostics。
2. **Leakage-free future-fold evaluation:** rolling-origin per-fold retrained models，作为 learned models 的主 generalization evidence。

**类型:** 改写文本。

### 3. Multi-endpoint claim 明显过强

**位置:** Methods lines 196-204; Results lines 786-798; Table/Figure lines 800-827; Limitations lines 981-984  

**问题:** Results 写 “demonstrates that the beta-binomial borrowing and calibration layer generalizes beyond ORR”；Figure title 写 “Borrowing benefit replicates across binary endpoint families”。但 evidence 是 leave-one-trial-out pooled robust-MAP borrowing，不是 full retrieval/two-head endpoint-specific pipeline；DCR n=38、treatment discontinuation n=20 很小。

**为什么重要:** 审稿人可能认为作者在暗示 full framework generalizes beyond ORR，而 Methods 又承认 full retrieval/two-head pipeline remains future work。

**建议怎么改:** 改成：

> As an exploratory borrowing-layer check, leave-one-trial-out robust-MAP borrowing showed NLL reductions in selected binary endpoint families. This does not establish endpoint-general performance of the full retrieval and two-head pipeline.

**类型:** 重大措辞调整；可考虑移 supplement。

### 4. SAM 的 “withholds non-comparable evidence” 需要降级

**位置:** Abstract lines 83-85; Results lines 657-674, 700-708; Discussion lines 948-951; Conclusion lines 1015-1017  

**问题:** Sham controls 是 label-free discrimination test，不等同于真实 clinical non-comparability。mass-based AUC 0.52 也很弱，不适合支撑强 claim。

**为什么重要:** “non-comparable evidence” 在历史借用语境里具有临床/统计 adjudication 含义，不能由 sham controls 单独证明。

**建议怎么改:** 将 “withholds borrowing from non-comparable evidence” 改成：

> reduced historical influence under constructed sham-history and simulated conflict settings.

**类型:** Claim boundary。

### 5. Case studies 重复，且用语过度贴近 truth

**位置:** Results lines 830-923; Tables lines 858-913  

**问题:** 两个 case 都是 historical evidence concordant with low/moderate response，SAM did not trigger。它们说明“compatible history sharpens prediction”，但不能展示 conflict behavior。另有 F9 conflict case PDF 存在但未引用。

**为什么重要:** 如果要说服读者 SAM 有 conflict sensitivity，一个 conflict case 比两个 concordant cases 更有解释力。当前表述 “almost exactly the true”, “borrows decisively”, “moves prediction onto observed” 容易显得 cherry-picking。

**建议怎么改:** 保留一个 concordant case，另一个换成 F9 conflict case；或两个 case 都移 supplement。文本改成 single retrospective illustration。

**类型:** 结构调整 + 图表处理。

### 6. References 仍是草稿状态

**位置:** References lines 1039-1072  

**问题:** 文中有注释 “Volume/issue/page/DOI intentionally omitted”；ref8/ref10/ref11 有 `TODO-verify`。无 `.bib`，引用为手工 `[1--7]`。

**为什么重要:** 这是投稿前硬阻塞。Pharmaceutical Statistics 审稿前技术检查就可能退回。

**建议怎么改:** 使用 BibTeX 或 Wiley-compatible bibliography，完整核验：

- Pocock historical controls
- Viele et al. historical controls
- Ibrahim and Chen power prior
- Neuenschwander MAP
- Schmidli robust MAP
- Hobbs commensurate priors
- Morita ESS
- FDA external controls guidance
- ICH E9(R1)
- O'Hagan/Pericchi conflict/heavy-tailed prior reference
- SAM self-adapting mixture prior paper
- SECRET, Trial2Vec, ClinicalBERT/Bio_ClinicalBERT references
- RBesT package / robust MAP implementation citation if used

**类型:** 补文献，must fix。

### 7. Data and Code Availability 未完成且排版溢出

**位置:** lines 1021-1030  

**问题:** `DOI:<to be assigned on Zenodo release>` 是占位符；编译后该段长文件名明显越出右边界。

**为什么重要:** 投稿时 data/code availability 是核心透明度材料。占位符和排版溢出会显得稿件未完成。

**建议怎么改:** 拿到真实 Zenodo DOI/URL；长路径用 prose 或 `\url{}`；避免 `\texttt{}` 长字符串不可断行。

**类型:** 投稿材料 + LaTeX formatting。

### 8. Appendix 更像内部审查表，不像投稿附录

**位置:** Appendix lines 1076-1142  

**问题:** Terminology 和 Claim-Evidence Map 对内部修改有用，但当前版面 underfull/overfull 明显，Claim-Evidence Map 还写入 “Verifier flags 4 mislabeled MLP checkpoints”，像内部 QA 记录。

**为什么重要:** 投稿稿件应把内部 QA 变成可复现 supplement，而不是暴露未整理的工程审查语言。

**建议怎么改:** 主文删除或移 supplement；如果保留 supplement，应加 exact scripts/artifacts/versions，而不是简短 claim table。

**类型:** 结构调整。

## D. Claim-Evidence Alignment Table

| Manuscript claim | Current evidence | Status | Recommended revision |
|---|---|---|---|
| SECRET pool improves Stage 1 retrieval metrics over hashing | 1,470 common ORR pseudo-queries; paired bootstrap CIs | Supported for selected ORR retrieval/readiness metrics | “SECRET-inspired pooling improved selected retrieval/readiness metrics in ORR pseudo-queries.” |
| Trained two-head calibration improves held-out NLL over rule | 1,407 ORR retrospective comparison; learned rows include model-development queries | Supported as retrospective/model-development evidence | “In a retrospective ORR comparison, two-head priors had lower NLL than the rule prior.” |
| two-head + SAM is the best-calibrated method | NLL, PIT KS, coverage, reliability slope among evaluated methods | Partly supported; “best” too broad | “showed the lowest NLL and favorable calibration diagnostics among evaluated methods.” |
| Conflict-adapted borrowing generalizes forward in time | 3 rolling-origin cutoffs with future folds | Overstated | “showed lower future-fold NLL in three rolling-origin splits.” |
| SAM withholds borrowing from non-comparable evidence | Sham donor/rate-permuted controls; mass-based AUC 0.52; simulation conflict sweep | Overstated | “reduced historical influence under constructed sham-history and simulated conflict settings.” |
| SAM detects prior-data conflict | Known-truth simulation ROC-AUC up to 0.74 | Supported only in simulation | “showed conflict sensitivity in known-truth simulations.” |
| Learned prior outperforms robust-MAP | Retrospective ORR comparison against tuned robust-MAP | Supported for evaluated ORR setting | “outperformed the evaluated robust-MAP baseline in retrospective ORR predictive NLL.” |
| Benefit generalizes beyond ORR | Leave-one-trial-out pooled robust-MAP on selected endpoint families | Overstated | “exploratory borrowing-layer check across selected binary endpoint families.” |
| Individual candidate borrowability is established | No expert labels | Not supported | Keep “no expert borrowability labels; not expert adjudication.” |
| Framework can support real trial decisions | No prospective validation, no protocol-specific OC for intended decision | Not supported | “requires disease-specific expert review, endpoint-specific validation, protocol-level pre-specification, and formal OC evaluation.” |

## E. Statistical and Methodological Review

### Formula and notation

The manuscript is strongest where it separates `lambda_i` and `a_i`.

- Lines 270-273 define beta component parameters using `a_i`.
- Lines 280-288 define mixture prior and explicitly distinguish `lambda_i` from `a_i`.
- Lines 334-350 define two output heads: allocation head for `lambda_i`, discount head for `a_i`.

This distinction should be preserved. It is a real methodological contribution.

### Weighted beta-binomial / posterior / prior borrowing

The prior predictive NLL formula at lines 422-431 is clear:

```tex
P(y_0 \mid n_0) = \sum_k \lambda_k \binom{n_0}{y_0}
\frac{B(y_0+\alpha_k, n_0-y_0+\beta_k)}{B(\alpha_k,\beta_k)}
```

However, the manuscript repeatedly refers to posterior behavior, posterior mean, and borrowing decisions. The current Methods should add either:

- posterior mixture updating formula after observing current trial data, or
- a clear statement that the primary evaluation is prior predictive / held-out predictive scoring, while posterior summaries are illustrative.

### SAM adapter

The SAM formula is understandable. The boundary sentence at lines 402-404 is good:

> “SAM adaptation is interpreted as a transparent conflict-sensitivity mechanism, not as a regulatory qualification procedure.”

But one crucial point should be more explicit: SAM uses the current/pseudo-query outcome to assess prior-data conflict. Therefore, in a real trial design setting, it cannot be used to pre-select historical evidence before observing the trial outcome unless embedded in a pre-specified dynamic borrowing design. This should be stated in Methods or Discussion.

### Training objective and leakage

The training objective includes a listwise target based on held-out log predictive values, but the manuscript says this term is disabled by default (`eta = 0`). This is acceptable, but for auditability the paper should state:

- whether all reported main results used `eta = 0`;
- whether any hyperparameters were selected using future-fold information;
- whether per-fold rolling-origin models used the same fixed hyperparameters;
- the exact number of training examples at each cutoff, currently only partly in Limitations.

### Evaluation design

The evaluation is rich but needs clearer hierarchy:

1. ORR retrieval benchmark: tests candidate discovery/readiness, not borrowability.
2. ORR retrospective NLL/calibration: tests predictive calibration, partly model-development evidence.
3. Rolling-origin: strongest leakage-free evidence for learned models.
4. Sham controls: label-free negative-control behavior, not clinical non-comparability validation.
5. Simulation OC: known-truth stress test, not real-world operating characteristic guarantee.
6. Multi-endpoint: borrowing-layer generality, not full pipeline generality.
7. Case studies: interpretability examples, not validation.

### Missing or weak analyses

- No expert borrowability labels.
- No blinded clinical/statistical adjudication.
- No external validation on a second registry/source.
- No prospective validation.
- No full endpoint-specific retrieval/two-head pipeline beyond ORR.
- No full retraining ablation for the two-head model; feature-weight proxy is not equivalent.
- No clear model checkpoint provenance in manuscript.
- No complete reproducibility manifest in the package itself beyond prose.

## F. Section-by-Section Comments

### Abstract

**Strength:** Structured and comprehensive; boundary sentence in Conclusions is good.  

**Issue:** Too much is packed into Results. The abstract reports retrieval, NLL, rolling-origin validation, sham controls, OC simulation, and multi-endpoint replication all at once. It reads more like an evidence manifest than an abstract.

**Line-specific concerns:**

- Lines 80-86: “improved”, “showed”, “detects”, “replicated” should be softened.
- Lines 88-93: Good boundary language; keep and consider moving part of this boundary earlier.

**Suggested replacement for Results closing sentence:**

> In these retrospective and simulation evaluations, conflict-adapted priors showed lower predictive NLL and more conservative behavior under constructed prior-data conflict, but these analyses do not establish expert borrowability of individual historical trials.

### Introduction

**Strength:** Lines 114-123 clearly distinguish retrievability from borrowability. This is the conceptual backbone of the paper.

**Issue:** Prior work is underdeveloped. Lines 111-112 cite `[1--7]` and `[8,~9]` as blocks, but do not explain what each class of prior contributes or fails to address.

**Needed additions:**

- robust MAP and MAP priors;
- power priors;
- commensurate priors;
- effective sample size;
- prior-data conflict and robustification;
- external controls guidance;
- clinical-trial retrieval / representation learning literature;
- SECRET and Trial2Vec context if named.

### Methods

**Strength:** The method chain is clear. `lambda_i` and `a_i` are consistently separated.

**Issues:**

- Stage 1 includes ClinicalBERT and optional Trial2Vec, but Results benchmark shown in the main text focuses hashing vs SECRET pool. Need clarify which retrieval backends are evaluated versus implemented.
- The description of “SECRET-style” is conservative and good; keep “not a complete reproduction”.
- Evaluation protocol should include exact pseudo-query construction counts, exclusion rules, and split criteria in a table.
- The rolling-origin section should specify whether hyperparameters and model architecture were fixed before future-fold scoring.

### Results

**Strength:** There is a serious evidence package.

**Issue:** Results currently read as a chain of victories. For a statistical methods journal, this should read as a set of bounded questions:

- Does retrieval improve candidate readiness?
- Does learned allocation improve predictive NLL?
- Does conflict adaptation improve behavior under conflict?
- Does the result survive leakage-free future-fold scoring?
- Which evidence is exploratory?

**Recommended restructuring:**

1. Stage 1 retrieval benchmark.
2. ORR prior predictive performance and calibration.
3. Leakage-free rolling-origin evaluation.
4. Conflict/sham/simulation behavior.
5. Robustness and data-quality checks.
6. Exploratory endpoint and case-study supplements.

### Discussion

**Strength:** Lines 963-1001 contain strong boundary language.

**Issue:** Lines 938-951 are too assertive before the limitations. Phrases like “genuine calibration”, “generalizes forward in time”, and “method withholds borrowing” should be rewritten with evidence boundaries.

**Suggested replacement for lines 938-951:**

> The results support several bounded observations. Retrieval backend choice affected downstream ORR candidate readiness. In retrospective ORR pseudo-query evaluations, mixture-prior calibration and SAM-style conflict adaptation were associated with lower predictive NLL and favorable calibration diagnostics among the evaluated baselines. In three rolling-origin splits, per-fold retrained learned priors retained lower future-fold NLL. Sham-history controls and known-truth simulations suggested conflict sensitivity, but these label-free analyses do not establish clinical exchangeability or expert borrowability.

### Limitations

**Strength:** The limitations are unusually honest and should be preserved.

**Needed strengthening:**

- Add the phrase `heuristic and unvalidated` for reranking/gates/discounts.
- Explicitly state: `no expert borrowability labels were available`.
- Explicitly state: `retrospective predictive calibration is not clinical validation`.
- Explicitly state: `expert review and external validation are prerequisites`.

### Data and Code Availability

**Status:** Not submission-ready.

Must resolve Zenodo DOI, release manifest, environment, and path wrapping.

### References

**Status:** Not submission-ready.

Current entries are placeholders and require full verification.

## G. Figure and Table Comments

### Figure 1: `F1_headtohead_nll.pdf`

**Conclusion shown:** two-head + SAM has lowest held-out NLL among evaluated methods.  
**Problem:** Figure title says “has the lowest held-out NLL”; acceptable if limited to ORR retrospective setting. Caption says “best-calibrated method”, which is too broad.  
**Recommendation:** Use “lowest NLL among evaluated priors in ORR pseudo-queries”.

### Table 1: Borrowing-prior head-to-head

**Conclusion shown:** two-head + SAM lowest NLL; rule + SAM close; robust-MAP baseline behind.  
**Problem:** Coverage alone does not define calibration; weak-only has high 95% coverage but poor informativeness.  
**Recommendation:** Include one sentence explaining that high coverage can reflect diffuse predictions.

### Figure 2: `F2_forward_validation.pdf`

**Conclusion shown:** two-head + SAM lower future-fold NLL across three cutoffs.  
**Problem:** Figure title “generalizes forward in time, leakage-free” is too strong.  
**Recommendation:** “Lower future-fold NLL in three rolling-origin splits.”

### Table 2: Rolling-origin forward validation

**Conclusion shown:** learned rows are leakage-free after per-fold retraining.  
**Problem:** Need exact per-cutoff training sizes and whether seeds/hyperparameters are fixed. Limitations mention default unfixed seed, which weakens reproducibility.  
**Recommendation:** Add per-cutoff rows or supplement table with train/test sizes and seeds.

### Figure 3: `F3_calibration.pdf`

**Conclusion shown:** SAM improves calibration diagnostics.  
**Problem:** “restores calibration” is too strong; reliability slope 1.28 is not perfect calibration, PIT still deviates from uniformity.  
**Recommendation:** “improves selected calibration diagnostics.”

### Table 3: Predictive calibration diagnostics

**Conclusion shown:** two-head + SAM lowest NLL and favorable PIT KS.  
**Problem:** Reliability slope definition is not fully explained.  
**Recommendation:** Define how reliability slope is estimated and how bins are formed.

### Figure 4: `F4_oc_safety_and_detection.pdf`

**Conclusion shown:** SAM responds to increasing conflict in simulation.  
**Problem:** Figure title “controls type I error” is too strong; type I error still rises.  
**Recommendation:** “mitigates conflict-related operating-characteristic degradation in simulation.”

### Figure 5: `F5_sham_controls.pdf`

**Conclusion shown:** sham evidence worsens fixed borrowing; SAM-adapted priors are less harmed.  
**Problem:** Caption says “evidencing that the method withholds borrowing from non-comparable evidence”; too strong for label-free sham control.  
**Recommendation:** “suggesting sensitivity to constructed sham-history corruption.”

### Table 4: Sham-borrowing negative controls

**Conclusion shown:** SAM trigger rate increases for unrelated donor history.  
**Problem:** mass-based AUC 0.52 is weak; do not overemphasize discrimination.  
**Recommendation:** Focus on NLL behavior and trigger/mass changes; avoid strong “non-comparable” wording.

### Figure 6 and Table 5: Multi-endpoint generality

**Conclusion shown:** selected binary endpoints show NLL reductions under robust-MAP borrowing.  
**Problem:** Not full pipeline; small endpoint counts; DCR and discontinuation are especially underpowered.  
**Recommendation:** Rename to “Exploratory borrowing-layer endpoint check”; consider supplement.

### Figure 7: `F7_pipeline_schematic.pdf`

**Conclusion shown:** pipeline overview.  
**Strength:** Clear and useful; good first-figure candidate.  
**Issue:** Figure footnote says “the two-head DeepSets prior with the SAM adapter decides how much to borrow”; this can sound decision-system-like.  
**Recommendation:** Replace “decides” with “estimates heuristic prior weights and discounts for retrospective calibration/sensitivity analysis.”

### F8 and F9

**Status:** Present in figures directory but not referenced in `.tex`.

- F8 corresponds to a case study already represented as Table 6.
- F9 is a conflict case that may be more useful than the second concordant case.

**Recommendation:** Either remove unused PDFs from Overleaf package, move them to supplement, or cite them properly.

## H. Language and Framing Edits

### Edit 1

**Original:** “Rolling-origin forward validation confirmed forward-in-time generalization”  
**Suggested replacement:** “Rolling-origin splits evaluated future-fold predictive performance.”

### Edit 2

**Original:** “the conflict-adapted priors improved future-fold NLL”  
**Suggested replacement:** “the conflict-adapted priors showed lower future-fold NLL in these three rolling-origin splits.”

### Edit 3

**Original:** “SAM adapter withholds borrowing from non-comparable evidence and detects prior-data conflict”  
**Suggested replacement:** “SAM reduced historical influence under constructed sham-history and simulated prior-data-conflict settings.”

### Edit 4

**Original:** “The borrowing benefit replicated on additional binary endpoint families.”  
**Suggested replacement:** “An exploratory borrowing-layer analysis showed similar NLL reductions in selected binary endpoint families.”

### Edit 5

**Original:** “The two-head + SAM prior is best calibrated.”  
**Suggested replacement:** “The two-head + SAM prior had the lowest NLL and favorable calibration diagnostics among the evaluated ORR priors.”

### Edit 6

**Original:** “Conflict adaptation drives calibration.”  
**Suggested replacement:** “Conflict adaptation improved selected calibration diagnostics in the ORR pseudo-query evaluation.”

### Edit 7

**Original:** “Adaptive prior controls type I error and detects conflict as it grows.”  
**Suggested replacement:** “SAM-style adaptation mitigated conflict-related degradation in simulation and showed increasing conflict sensitivity.”

### Edit 8

**Original:** “This demonstrates that the beta-binomial borrowing and calibration layer generalizes beyond ORR.”  
**Suggested replacement:** “This exploratory analysis suggests that the beta-binomial borrowing layer can be evaluated on other selected binary endpoint families, but endpoint-general performance of the full retrieval and two-head pipeline remains untested.”

### Edit 9

**Original:** “borrows decisively, predicting 0.241 -- almost exactly the true 0.231”  
**Suggested replacement:** “assigned more historical mass and yielded a lower held-out NLL in this single retrospective example.”

### Edit 10

**Original:** “validated admission rule”  
**Suggested replacement:** “synthetically checked admission rule” or “admission rule that passed a synthetic round-trip check.”

## I. LaTeX / Compilation / Formatting Issues

### Compilation

Command run:

```bash
cd /Users/wang/Documents/New\ project/overleaf_manuscript_package
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript_full_draft_pharm_stats.tex
```

Result:

- Compilation succeeded.
- Output PDF had 23 pages.
- Cross-references resolved after latexmk rerun.
- No missing figure errors.

### Main warnings and visible issues

1. **Overfull hbox at Data and Code Availability**
   - Source lines: 1022-1031.
   - Visible issue: DOI placeholder and long file paths overflow right margin.
   - Fix: use real DOI/URL and `\url{}` or prose instead of long `\texttt{}` fragments.

2. **Appendix longtable underfull/overfull**
   - Source lines: 1078-1142.
   - Issue: narrow columns create awkward line breaks.
   - Fix: move appendix tables to supplement, use landscape, or use `tabularx`/smaller prose table.

3. **References are not BibTeX-managed**
   - Source lines: 1039-1072.
   - Issue: hand-numbered references are fragile; entries incomplete.
   - Fix: create `.bib` and use journal-compatible bibliography style.

4. **Placeholders remain**
   - Author line 41.
   - Zenodo DOI lines 1025-1027.
   - Acknowledgements lines 1032-1033.
   - Conflicts of Interest lines 1035-1036.
   - References TODO lines 1061-1072.

5. **Unused figure files**
   - `F8_case_study.pdf` and `F9_case_conflict.pdf` exist but are not referenced.
   - Fix: remove, move to supplement, or cite properly.

### Missing Overleaf/package components

- No `.bib`.
- No supplementary file.
- No Wiley class/template in package.
- No separate table files.
- No source data tables visible in package.
- No release manifest included in this directory, despite being mentioned in Data and Code Availability.

## J. Prioritized Revision Checklist

### Must Fix Before Submission

- [ ] Replace all author/affiliation placeholders.
- [ ] Complete Acknowledgements.
- [ ] Complete Conflicts of Interest.
- [ ] Add funding statement if applicable.
- [ ] Add author contributions if journal requires it.
- [ ] Replace Zenodo DOI placeholder with real DOI/URL.
- [ ] Verify Data and Code Availability against actual archived files.
- [ ] Complete all references with full bibliographic metadata and DOI where available.
- [ ] Remove all `TODO-verify` text from manuscript.
- [ ] Add missing citations for SECRET, Trial2Vec, Bio_ClinicalBERT/ClinicalBERT, RBesT/robust-MAP implementation, and any software packages central to the method.
- [ ] Downgrade “validated/generalizes/withholds/detects/controls/replicates” language throughout.
- [ ] Explicitly state that retrospective predictive calibration is not expert or clinical validation.
- [ ] Explicitly state that no expert borrowability labels were available.
- [ ] Explicitly state that expert review and external validation are prerequisites.
- [ ] Clarify which results are leakage-free and which are model-development retrospective comparisons.
- [ ] Fix visible LaTeX overflow in Data and Code Availability.
- [ ] Decide whether F8/F9 are removed, moved to supplement, or cited.

### Should Improve

- [ ] Reorganize Results into evidence layers rather than a long sequence of wins.
- [ ] Add a table summarizing evaluation datasets, query counts, train/test split, leakage status, and endpoint scope.
- [ ] Add posterior mixture updating formula or clarify that primary evaluation is prior predictive.
- [ ] Add exact bootstrap unit and CI computation details.
- [ ] Add calibration slope definition.
- [ ] Add OC simulation design details: null/alternative, type I error definition, decision rule, number of replicates, Monte Carlo SE.
- [ ] Move Claim-Evidence Map appendix to supplement or internal review document.
- [ ] Consider replacing second concordant case with F9 conflict case.
- [ ] Add exact provenance for model checkpoints and verification scripts.
- [ ] Clarify that multi-endpoint results are borrowing-layer only.

### Optional Polish

- [ ] Shorten Abstract and reduce numeric overload.
- [ ] Convert conclusion-style figure titles into descriptive titles.
- [ ] Use consistent phrase: “methodology prototype and retrospective predictive-calibration evidence package.”
- [ ] Use “SECRET-inspired” everywhere, not “SECRET” alone when implying reproduction.
- [ ] Replace “clinical gate” with “heuristic compatibility gate” unless expert-derived.
- [ ] Make title slightly more conservative if needed: “A retrospective predictive-calibration prototype for oncology trial retrieval and Bayesian historical borrowing.”
- [ ] Move internal software-test counts to supplement unless journal welcomes software evidence in main text.

## Recommended Conservative Framing Paragraph

The following paragraph could be inserted near the end of the Introduction or start of Discussion:

> This framework is intended as a reproducible methodology prototype for prior-borrowing-oriented oncology historical-trial discovery and retrospective predictive calibration. The reranking scores, effective-sample-size discounts, mixture weights, and conflict-adaptation outputs are heuristic and unvalidated. No expert borrowability labels were available, and the retrospective predictive and simulation analyses reported here do not establish clinical exchangeability, regulatory suitability, or borrowing recommendations for individual historical trials. Expert clinical and statistical review, endpoint-specific validation, external validation, protocol-level pre-specification, and formal operating-characteristics evaluation remain prerequisites for any decision-informing use.

## Suggested Revised Abstract Conclusion

Current conclusion is directionally good. A more conservative version:

> The framework provides a reproducible methodology prototype and retrospective predictive-calibration evidence package for prior-borrowing-oriented oncology historical-trial discovery. In the evaluated ORR pseudo-query and simulation settings, conflict-adapted mixture priors showed favorable predictive and conflict-sensitivity behavior. These results do not establish expert borrowability or clinical decision validity; expert review, external validation, and protocol-specific operating-characteristics evaluation are prerequisites for any applied borrowing use.

## Final Readiness Judgment

The manuscript should not be submitted in its current form, but it is within reach after a focused revision. The most important revision is not adding more experiments; it is tightening the evidentiary boundary so the paper reads as:

> a reproducible retrieval/reranking and retrospective predictive-calibration prototype for oncology historical borrowing,

not as:

> a validated system that identifies clinically borrowable historical trials.

If the claim boundary is tightened, references completed, submission placeholders removed, and Results reorganized by evidence type, this can become a credible Pharmaceutical Statistics methods submission.
