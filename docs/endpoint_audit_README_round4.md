# 第四轮终点审计说明（样本外验证，60 行）

## 这次审什么

只审**一件事**：每行的终点（endpoint）按注册库记录的实际定义，是否属于**严格 ORR**。

严格 ORR 的判定标准（与上一轮一致）：
- 是"客观缓解 / 总体缓解"类终点：CR+PR（RECIST 或同类标准下的 objective/overall/best overall response、response rate、confirmed response）；
- **不是**下列任何一种：CR-only（不含 PR）、pCR、PFS/OS/生存类、DCR（含 SD）、CBR（clinical benefit）、分子学/MRD 缓解、血液病复合缓解（CR+CRi、CRc 等）、免疫相关 response（irRC 单独定义）、major response、时间型（duration/time-to）、landmark（"at 6 months"型时点响应除非定义就是该时点的客观缓解率）等。

## 怎么填

打开 `registry_link`，找到与 `endpoint_title` 对应的 outcome measure（用 `extracted_responders`/`extracted_denominator` 帮助定位是哪一个 outcome 的哪一行）。**请看注册库页面上该 outcome 的完整描述（description/measure description），不要只看标题**——标题写 ORR 但描述定义成别的东西的情况正是我们要抓的。

- `human_strict_orr`：填 `yes` / `no` / `ambiguous`（描述不足以判断时填 ambiguous，并在 notes 里说明）。
- `human_actual_endpoint_type`：一两个词，如 `ORR (CR+PR, RECIST 1.1)`、`pCR`、`PFS at 6mo`、`DCR`、`CR only`。
- `auditor_notes`：可选，特殊情况说明。

## 盲法说明（重要）

这份表**故意不含**规范器的预测结果。规则已在抽样前冻结（脚本 sha256 `df78ad8a…`，随代码发布），样本从**未被上一轮 80 行审计覆盖的试验**中按分层随机抽出。您交回后我才把人工判定与冻结预测对合，计算样本外灵敏度/特异度/准确率。所以请完全按注册库记录独立判断，不用猜规范器会怎么判。

审完直接把 CSV 发回对话即可。
