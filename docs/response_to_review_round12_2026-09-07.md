# 第十二轮逐条审稿回应

**稿件**：Retrospective predictive calibration of oncology trial similarity for Bayesian historical borrowing
**本文档**：对第十二轮意见（技术性与编辑性修订）的逐条回应。感谢您确认上一轮的统计意见可以结案；本轮只做您列出的四项修订，不新增分析。

---

## 1. Table 8 被裁切（p.39）

**属实，已重排并做了页面几何核验。** 您指出得对：上一轮回复信所称"无 ≥10pt overfull"是错的——我们的日志检查按字符串而非数值排序，宽度超过 100pt 的溢出框被排到了列表尾部而未被看到。本轮做了两处改正：

- **表格重排**：随机化结果改为真正的逐行表（每个 mixing 一行：Target、Mean Δ、Central 95% range、评估半批 size（no borrow / final）、Splits > 0），共六行（corpus @0.02513 与 @0.025、N(0,1)、N(0,0.5)、两个 location-held）；旧插值分析降为同一表号下的第二个 tabular，并标注 "Legacy full-batch deterministic-threshold interpolation (approximately size-adjusted)"。两个 location-held run 的评估半批 size 此前未存档，已补算（`round12_eval_sizes.py`，同种子重跑，估计与 gap 与 round-10/11 artifact 逐项断言一致）：N(+0.124,1) 0.02514/0.02513、N(+0.124,0.5) 0.02515/0.02514。
- **核验方法升级**：（i）日志溢出框改为数值解析（本轮正文最大 3.25 pt，补充材料 0）；（ii）新增 `scripts/check_page_geometry.py`，用 `pdftotext -bbox` 对交付 PDF 逐页取最右字形坐标，任何页面超过正文区右边界（540 pt；页宽 612 pt）即报错——两份 PDF 全部通过；（iii）对第 39 页（Table 8）与补充材料第 15 页（Figure S1）另做了渲染目视检查，+0.0362 等全部数值在可见页面内。

## 2. Figure S1 图内文字

**属实，已重绘。** 面板 (b) 标题改为 "**Randomized-boundary size standardization, common target** (training-half size exact; medians and central 95% ranges)"，图例删除 "(exact size)"（现为 "randomized boundary, common target"）。图注不变。程序化核对：对图 PDF 抽取文字，不再含 "exact size" / "Exact expected"。

## 3. 新旧 mixing-bootstrap 区间

**接受，逐处标明来源，不做全局替换。**

- §4.5 随机化段：新区间 [+0.0005, +0.0045] 明确标注 "re-run under randomized matching at the fixed common target"（已在上一轮）。
- §4.5 dispersion 段（p.38）：旧区间改写为 "weight bootstrap **of that legacy interpolated analysis** [+0.0006, +0.0044], against [+0.0005, +0.0045] for the randomized estimator above"。
- Table 8 底块：改为两行——"bootstrap of mixing weights, **legacy interpolated estimator**: 95% interval [+0.0006, +0.0044]" 与 "(randomized estimator, common target: [+0.0005, +0.0045]; see the first block and text)"。
- 补充材料 S2 round-11 段原已写明新区间"取代"旧区间**仅就随机化 estimator 而言**；措辞保留，与正文一致。

## 4. 可复现性表述

**接受，改为您给出的两状态分离表述。** Data and Code Availability 现写："The complete versioned reproduction package --- code for the full pipeline together with lightweight summary tables and manuscript figures --- accompanies the submission as supplementary material; upon acceptance the corresponding tagged version will be deposited in Zenodo and its DOI added here."，并列出包内内容（逐轮分析脚本、逐 split / 逐样本输出、`verify_round*.py` 断言脚本）。

关于"本次向我提供的仍是两份 PDF 和回复信"：完整复现包（`revision_round12_2026-09-07.tar.gz`，含本轮及此前各轮的脚本、artifact 与断言脚本；本轮 `verify_round12.py` 48 项、上一轮 `verify_round11.py` 75 项）将作为 supplementary material 随本次修订一并上传，供您独立运行；`scripts/check_page_geometry.py` 亦在其中。作者信息与利益冲突占位符将在最终版本按期刊系统填写完成。

## 其他

- 页数：正文 58 页、补充 18 页（不变）。
- 验证：`scripts/verify_round12.py` **48 项断言通过**（Table 8 六行数值逐项对 artifact、区间来源措辞、图内文字、页面几何、数值解析的溢出框）；两文档 0 undefined references；页面几何检查无任何页面超出正文区。

## 主要修改位置速查

| 修改 | 位置 |
|---|---|
| Table 8 逐行随机化表 + legacy 标注 | Table 8（正文 p.39）、`round12_eval_sizes.json` |
| Figure S1 图内标题/图例 | `build_round9_figures.py`（重绘）、补充 p.15 |
| 新旧 bootstrap 区间来源标注 | §4.5（两处）、Table 8 底块 |
| 可复现性两状态表述 | Data and Code Availability |
| 页面几何核验工具 | `scripts/check_page_geometry.py`、`verify_round12.py` |
