<!--
This document defines the root-cause remediation plan for card quality in paper2bullet.
Main sections: background, gap analysis, target architecture, single-paper validation protocol, and phased rollout.
Data focus: three-level card logic (overall/local/detail), evidence grounding, duplicate governance, and measurable acceptance criteria.
-->

# Root-Cause Remediation Plan (Before Saturation)

## 1) 背景与问题定义

本项目的核心不是“把论文摘要说顺”，而是从论文中提炼可进入课程的 `aha moment` 卡片。  
依据 `CONCEPT.md`，系统必须满足：

- 卡片是认知跃迁，不是论文复述
- 卡片以论文证据为主体，且“它在课程里变成什么”是必答题
- 卡片颗粒度按三层扫描：`整体` / `局部` / `细节`
- 允许一篇论文产出 `0~N` 张，不凑数
- 卡片保持原子性，聚类不合并
- 三色标记用于分流人工注意力

当前代码已经做了“结构化字段 + 证据筛选 + 门禁 + 去重”的一轮改造，但这仍然是治理增强，不是最终根解。  
根因仍在：系统没有稳定实现“先理解整篇论文贡献结构，再决定出什么卡、出几张卡、每张卡的层级价值”。

---

## 2) 需求目标（本次方案的唯一北极星）

在不牺牲可审计性的前提下，实现以下能力：

1. 先完成整篇论文的贡献结构理解（而不是先截几段）
2. 明确区分并规划三层卡片：`整体` / `局部` / `细节`
3. 决策“要不要出卡、出几张、每张层级与价值”基于全局结构，不基于局部片段偶然性
4. 卡片生成后仍可治理（证据、去重、校准、三色、排除透明）
5. 在进入 saturation 之前，先把“单篇论文端到端流程”测通并可复盘

---

## 3) 当前代码与理想方案的差距

## 3.1 现在已有能力（已实现）

- section 结构标注（abstract/body/front matter 等）
- 证据段评分与证据包组装（context/primary/supporting）
- extraction/judgement 合同字段增强（object/claim/evidence level 等）
- deterministic gates（abstract/front/object/body）
- 同论文重复治理（cluster + representative）
- review 诊断与质量指标 API

## 3.2 距离“完美解决需求”的关键缺口

1. **缺少“整篇理解先行”的显式阶段**
   - 当前仍是“证据包 -> extraction”的主路径
   - 没有一个稳定持久化的“论文贡献结构模型”

2. **缺少“先规划后生成”的卡片决策层**
   - 当前没有 Card Plan（哪些候选应出、层级是什么、覆盖关系如何）
   - 导致层级判断仍偏后置、偏隐式

3. **门禁偏拒绝，缺少“可恢复治理”**
   - 目前门禁主要是 fail/drop
   - 缺少“回补上下文 -> 重判 -> 再决策”的恢复链路

4. **三层框架评价未成为第一公民**
   - 有字段但缺少“层级质量”主指标与强约束验收
   - 无法直接证明整体/局部/细节逻辑是否被正确捕捉

---

## 4) 从根上解决：目标架构

目标不是再加一层 prompt，而是改成“认知-决策-生成”三阶段架构。

### Stage A: Paper Understanding Pass (全文理解)

输入：整篇论文（全部 sections + figures）  
输出：`Paper Understanding Record`（持久化）

最小结构：

- `global_contribution_objects`: 论文最关键的贡献对象（可多个）
- `contribution_graph`: 对象之间关系（支撑、依赖、对比、边界）
- `evidence_index`: 每个对象绑定的证据锚点（section/figure）
- `candidate_level_hints`: 每个对象更适合整体/局部/细节哪一层

要求：

- 必须使用整篇可用正文，不允许只看前 N 段
- 输出必须可审计（每个节点有证据锚点）

### Stage B: Card Planning Pass (先决策出卡)

输入：`Paper Understanding Record` + topic + calibration  
输出：`Card Plan`（持久化）

最小结构：

- `planned_cards[]`
  - `level`: overall/local/detail
  - `target_object_id`
  - `why_valuable_for_course`
  - `must_have_evidence_ids`
  - `optional_supporting_ids`
  - `disposition`: produce/exclude
- `coverage_report`: 哪些对象被覆盖、哪些被排除及原因

要求：

- 先决定“出什么”和“为什么”，再写卡内容
- 允许 `0 卡`，但必须给出结构化排除理由

### Stage C: Card Realization Pass (按计划生成卡)

输入：`Card Plan` + evidence  
输出：最终 cards / excluded content

要求：

- 一张卡只能对应一个主要对象（原子性）
- 必须回答“它在课程里变成什么”
- 必须保留证据主体 + 极简分析
- 若证据不足先触发恢复链路，不直接硬拒绝

### Stage D: Governance Pass (治理与可恢复)

- 证据治理：不满足正文支撑时先补检索/补上下文
- 重复治理：同对象同证据簇只保留最佳代表
- 校准治理：边界样本回流 prompt/rubric
- 诊断治理：每次拒绝、替换、降级都有结构化理由

---

## 5) Benchmark（与 CONCEPT.md 对齐）

本方案以以下验收准则为准：

### 5.1 硬约束（必须满足）

- 能回答“它在课程里变成什么”（名词化）
- 可行动（认知/态度/方法至少其一）
- 原文证据为主体
- 原子卡片，不合并
- 允许 0 卡，不凑数
- 排除清单透明可审计

### 5.2 主质量指标（用于方案比较）

- belief-gap 命中率
- aha 有效率（本质性/反直觉/tacit-to-explicit）
- course-transformability rate
- paper-specific-object presence rate
- duplicate escape rate
- 假阳性率 / 假阴性率（假阴性权重更高）

### 5.3 三层框架指标（语料级，不要求每篇齐全）

- level labeling accuracy（整体/局部/细节）
- cross-level coherence（下层是否可回指上层对象）
- anti-padding rate（无“为补层级而凑卡”）

---

## 6) 先测通“一篇文章”流程（在 Saturation 之前）

这是本次执行的硬前置条件。没有单篇测通结果，不进入 saturation。

## 6.1 单篇测通目标

给定 1 篇论文，完整走通：

1. 全文理解（Stage A）
2. 卡片规划（Stage B）
3. 卡片生成（Stage C）
4. 治理与诊断输出（Stage D）

并输出一份可审计报告，让人能直接判断“是否真的理解了三层结构”。

## 6.2 单篇测通的必交付物

- `paper_understanding.json`
- `card_plan.json`
- `final_cards.json`
- `excluded_content.json`
- `single_paper_validation_report.md`

报告必须包含：

- 论文贡献对象列表（含证据锚点）
- 每张卡所属层级（整体/局部/细节）与理由
- “它在课程里变成什么”是否可命名
- 未出卡内容及原因
- 重复抑制与替代关系

## 6.3 单篇通过标准

全部满足才算通过：

1. 至少有一条清晰的贡献对象链路被识别（可单层或多层）
2. 每张输出卡都能映射到明确对象与证据锚点
3. 卡片层级判定可解释，且不出现明显层级错配
4. 出卡与排除决策能被人类复核并复现
5. 不出现“看起来通顺但无法课程命名”的伪卡

---

## 7) 实施顺序（严格顺序）

1. 定义并持久化 `Paper Understanding Record` 数据契约
2. 实现 Stage A（全文理解）并写最小单元测试
3. 定义并持久化 `Card Plan` 数据契约
4. 实现 Stage B（先决策）并写最小单元测试
5. 改造 Stage C（按计划生成）
6. 将门禁改为“可恢复治理优先，硬拒绝兜底”
7. 跑通单篇端到端，产出完整报告（必须）
8. 通过后才开始 saturation 相关治理

---

## 8) 明确不做（本阶段）

- 不先做大规模 saturation 扩展
- 不先追求 UI 美化
- 不做与单篇测通无关的性能优化

---

## 9) 风险与防错

主要风险：

- 全文理解阶段不稳定，导致后续计划漂移
- 规划阶段过度保守，误杀潜在好卡
- 规划阶段过度激进，导致凑卡

防错策略：

- 先用小样本（1 篇）深度调试
- 对每次决策保留可追踪证据
- 以假阴性高代价原则调整阈值（先召回、后收敛）

---

## 10) 结论

这份方案的核心不是“再调 prompt”，而是把系统从“局部抽段+后置筛选”升级为“全文理解 -> 先决策 -> 再生成 -> 可恢复治理”。  
并且执行上明确要求：**先测通单篇论文流程并提交可审计效果，再进入 saturation 阶段。**

