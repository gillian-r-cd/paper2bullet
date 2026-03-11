# 双模式研究系统改造方案

## 0. 这份文档要解决什么问题

当前系统已经较完整地支持了 `Aha exploration`：

- 按 `topic` 做 discovery
- 解析论文全文
- 提取 `candidate cards`
- 做 judgement / review / export

但它还没有显式支持另一类需求：

- 我已经有一个命题、框架或维度集合
- 我不是要系统帮我“发现新的 aha”
- 我是要系统帮我“为这个命题找支持证据、边界条件、反证和高质量引用”

同时，系统还新增了两个跨模式的共性需求：

1. 对任意一篇论文，用户可以直接向 AI 提问，AI 读取该论文全文后给出回答。
2. 用户对 `card` 或 `matrix item` 的 `accept / reject / comments` 不应只是静态记录，而应成为一个可审查、可迭代、可回滚的“偏好记忆”来源，用来改善后续 planning 和 filtering。

本文目标是：

- 从第一性原理澄清问题
- 在遵循奥卡姆剃刀原则的前提下提出**最小可跑通方案**
- 明确哪些部分可复用现有系统
- 明确哪些地方必须由你拍板确认
- 尤其明确“决策闭环”应该如何设计

---

## 1. 第一性原理分析

### 1.1 当前系统真正优化的对象是什么

当前系统优化的不是“学术研究任务的一切形态”，而是一个非常具体的对象：

> 从论文中发现、提炼、判断并输出值得进入课程的 `aha moment candidate cards`

这意味着当前系统的隐含前提是：

- 输入对象主要是 `topic`
- 搜索目标主要是“找未知但高价值的发现”
- 输出对象主要是 `card`
- 评价标准主要是：
  - 是否构成 `aha`
  - 是否可课程化
  - 是否值得 review / export

这套前提在 `Aha exploration` 任务里是成立的。

### 1.2 你提出的新需求，本质上是另一类任务

以这个需求为例：

> 帮我证明五个维度能导向领导力升级和团队绩效提升

它不是在问：

- 这个领域里还有什么新的认知重构值得发现？

它在问：

- 我已有一个框架，这个框架的每个维度分别有哪些学术证据支持？
- 支持证据强不强？
- 有哪些边界条件、限制和反证？
- 最终能不能形成一个对领导可汇报的证据矩阵？

这类任务的正确对象不是 `card`，而是：

- `claim`
- `dimension`
- `outcome`
- `evidence matrix item`

### 1.3 两类任务的根本差异

| 维度 | `Aha exploration` | `Claim-driven evidence search` |
| --- | --- | --- |
| 输入对象 | `topic` / research goal | 已知 `claim` / 维度 / 结果变量 |
| 搜索目标 | 发现未知高价值 insight | 为已知命题找支持、限制、反证 |
| 中间步骤 | topic expansion | claim decomposition + search plan |
| 输出对象 | `card` | `matrix item` |
| 评价标准 | 是否 `aha`、是否可课程化 | 证据强度、直接性、外部效度、是否有反证 |
| 停止条件 | saturation 是否趋平 | 每个维度/结果变量的 coverage 是否足够 |

### 1.4 真正缺失的不是“一个搜索按钮”，而是一个 planning layer

当前系统没有显式支持“定向搜索”的根因不是少了一个 prompt，而是少了一个中间层：

> `Research Brief -> Search Plan`

没有这个 planning layer，系统就只能把所有需求都粗暴压成 `topic`。

一旦把“帮我证明一个命题”压成 `topic`，就会立刻丢失：

- 这个命题的维度拆解
- 目标结果变量
- 学术检索同义词
- 证据强度要求
- 是否需要主动找反证

因此：

> 当前系统不是“还差一个定向搜索步骤”，而是“还没把 claim/evidence 任务建模成一等公民”。

### 1.5 但这不意味着要重做一套系统

从第一性原理看，`Aha exploration` 和 `claim-driven evidence search` 并不是两套完全独立的系统。

它们共享同一套底座：

- discovery source 接入
- access queue
- PDF / HTML parsing
- `paper_sections`
- `figures`
- review ledger
- export service

真正应该分叉的只有三层：

1. 输入与 planning
2. 输出对象
3. 评价标准

这就是本方案遵循奥卡姆剃刀的核心：

> **共用底座，只在必要处分叉。**

---

## 2. 设计原则

### 2.1 奥卡姆剃刀原则下的目标形态

在“能跑通”的前提下，优先选择：

- 复用现有 `run/discovery/parse/review/export` 主干
- 不新建第二套 discovery 系统
- 不新建第二套 review 系统
- 不做全自动黑盒意图路由
- 不做不可审查的自动学习
- 不做多轮 agent 自主规划闭环

### 2.2 推荐的总方案

推荐采用：

> **显式双 mode + LLM 提示建议 + 统一的 Draft Plan / Confirm Plan 步骤**

也就是：

1. 用户输入需求
2. 系统建议更像哪种任务
3. 用户可显式确认或切换 mode
4. 系统先生成 `Draft Search Plan`
5. 用户确认 `Search Plan`
6. 系统再开始跑 discovery / parsing / generation

这里最关键的不是“mode”本身，而是：

> **任何任务都不应该直接从 brief 跳到搜索，而应先经过可见、可改、可确认的 search plan。**

### 2.3 为什么不推荐“纯意图分析自动决策”

纯意图分析自动决策看起来更顺滑，但不是本阶段最优解。

原因很简单：

- 两条 pipeline 的目标函数不同，不是轻量路由
- 一旦路由错，整批结果方向都错
- 用户很难知道错误发生在“意图识别”还是“检索质量”
- 难以 debug、难以做回归验证

所以建议：

- `intent analysis` 只作为建议器
- 最终 mode 由用户确认

---

## 3. 推荐的最小目标架构

### 3.1 两种 mode

- `aha_exploration`
- `claim_evidence`

### 3.2 一个统一的中间对象：`Search Plan`

所有任务在真正执行前，系统都先产出一个结构化 `Search Plan`。

对于 `aha_exploration`，它回答：

- 我们到底搜哪些 topic anchors？
- 这些 topics 背后的搜索策略是什么？
- 最后要产出什么类型的 `cards`？

对于 `claim_evidence`，它回答：

- 这个命题被拆成哪些维度？
- 目标结果变量有哪些 proxy？
- 每个维度对应哪些学术锚点和同义词？
- 搜索时要优先什么证据，回避什么噪音？
- 是否要求主动寻找反证或限制条件？

### 3.3 一个统一的执行容器：继续复用 `run`

不建议新建第二套 `task engine`。

最小改法是：

- 继续把 `run` 作为一次执行容器
- 在 `run` 的 metadata 中记录：
  - `task_type`
  - `task_brief`
  - `draft_plan`
  - `confirmed_plan`
  - `memory_snapshot`

这样可以避免一开始就引入新的顶层表结构。

如果未来任务形态显著增多，再考虑把 `run` 之上的 `research_task` 独立成新表。

### 3.4 两种输出对象

- `aha_exploration` 输出：`candidate_cards`
- `claim_evidence` 输出：`evidence_matrix_items`

这一步必须显式分开，不建议把 `matrix item` 硬塞进 `candidate_cards`。

原因：

- 两者语义不同
- 两者 judgement 标准不同
- 两者 export 形态不同
- 后续 review 和 memory 的信号也不同

---

## 4. 端到端流程

### 4.1 推荐的统一主流程

1. 用户输入 brief
2. 系统给出 mode 建议
3. 用户确认 mode
4. 系统生成 `Draft Search Plan`
5. 用户确认 `Search Plan`
6. 系统执行 discovery
7. 系统解析全文并写入 `paper_sections` / `figures`
8. 系统生成输出对象
9. 用户对输出对象做 `accept / reject / keep / needs_manual_check / comments`
10. 系统基于这些反馈起草 `Memory Update Draft`
11. 用户确认或拒绝这份 memory update
12. 被确认的 memory 才影响下一次任务

### 4.2 `aha_exploration` 流程

- 输入：topic / research goal
- plan 输出：search-friendly topic anchors
- 执行：尽量沿用现有 discovery + extraction + judgement
- 输出：`card`
- review：沿用现有 `card/excluded` review 流程

### 4.3 `claim_evidence` 流程

- 输入：claim brief
- plan 输出：
  - claim statement
  - dimensions
  - outcomes / proxies
  - search anchors
  - evidence rules
  - contradiction policy
- 执行：
  - 按计划生成 query anchors
  - 复用 discovery / parse / evidence store
  - 生成 `matrix item`
- 输出：`matrix item`
- review：进入统一 review surface

---

## 5. 系统调整清单

### 5.1 Intake / UI / API

#### 最小改动建议

新增一个“先起 plan，再启动 run”的入口，不直接废掉现有 `Start Run`。

推荐新增：

- `Draft Research Plan`
- `Confirm Plan & Start Run`

而不是直接把旧入口彻底推翻。

#### UI 变化

`Start Run` 面板建议变成：

- `Research Brief`
- `Mode`
  - `Aha Exploration`
  - `Claim Evidence`
- `Suggest Mode`
- `Draft Search Plan`
- `Confirm Plan & Start Run`

#### API 变化

建议新增：

- `POST /api/research-plans/draft`
- `POST /api/runs` 扩展支持 `task_type` 和 `confirmed_plan`

建议保留：

- 现有 `POST /api/runs`
- 现有 `POST /api/discovery/recommend-search-terms`

但后者应逐步降级为 `aha_exploration` 的子能力，而不是总入口。

### 5.2 Planning Layer

这是整个改造里最关键的新层。

#### `aha_exploration` 的 `Draft Plan` 最小结构

```json
{
  "task_type": "aha_exploration",
  "research_goal": "Find high-value management dialogue insights for course design.",
  "recommended_topics": [
    "leader listening",
    "manager coaching",
    "psychological safety"
  ],
  "search_profile": "aha_default",
  "output_type": "card"
}
```

#### `claim_evidence` 的 `Draft Plan` 最小结构

```json
{
  "task_type": "claim_evidence",
  "claim": "The five dialogue dimensions support leadership upgrading and team performance.",
  "dimensions": [
    "expression",
    "listening",
    "questioning",
    "empathy",
    "action_enabling"
  ],
  "outcomes": [
    "leadership effectiveness",
    "team performance",
    "trust",
    "psychological safety",
    "employee commitment"
  ],
  "search_anchors": [
    "leader communication clarity",
    "leader listening",
    "managerial questioning",
    "leader empathy",
    "empowerment leadership"
  ],
  "evidence_policy": {
    "prefer": ["meta-analysis", "field study", "high-citation classic", "recent empirical"],
    "surface_contradictions": true,
    "minimum_supporting_papers_per_dimension": 3
  },
  "output_type": "matrix_item"
}
```

#### 关键原则

- planner 输出必须结构化，不能只是解释性 prose
- planner 只负责“把任务翻译成可执行计划”，不负责直接下最终结论
- plan 必须先给用户看，再执行

### 5.3 Discovery Strategy

#### `aha_exploration`

保留当前 discovery profile 为默认：

- `core`
- `mechanism`
- `application`
- `recency`

#### `claim_evidence`

不建议直接复用当前 `application_focus = case study` 逻辑。

最小可行改法是给 `claim_evidence` 单独一套更贴近证据搜集的 profile：

- `core`
- `outcome_focus`
- `evidence_focus`
- `recency`

示例：

- `leader listening`
- `leader listening psychological safety`
- `leader listening employee performance`
- `leader listening` + `recent_window`

#### 原则

- 不做复杂多轮 query agent
- 不做自动无限迭代改写 query
- 由 planner 一次性产出结构化 anchors
- discovery 只执行计划，不自行发散

### 5.4 Parsing / Evidence Store

这部分尽量不改。

直接复用现有：

- `papers`
- `paper_sections`
- `figures`
- `access_queue`

这意味着：

- 论文全文问答可以直接基于 `paper_sections`
- `matrix item` 的证据也应直接引用 `paper_sections`
- 不需要另建第二套全文存储

### 5.5 输出对象：新增 `evidence_matrix_items`

这是本次真正建议新增的核心表。

建议最小字段：

- `id`
- `run_id`
- `paper_id`
- `topic_id` 或 `anchor_topic`
- `dimension_key`
- `dimension_label`
- `outcome_key`
- `outcome_label`
- `claim_text`
- `verdict`
  - `supporting`
  - `mixed`
  - `contradictory`
  - `context_only`
- `evidence_strength`
  - `strong`
  - `medium`
  - `weak`
- `summary`
- `limitation_text`
- `citation_text`
- `evidence_json`
- `created_at`

#### 为什么不复用 `candidate_cards`

因为 `card` 的本体是“课程化 insight”，而 `matrix item` 的本体是“命题证据单元”。

把两者揉在一起会造成：

- judgement 语义混乱
- export 结构混乱
- review signals 被污染
- memory 学习对象混乱

### 5.6 Review Surface

这里应尽量复用现有机制。

#### 可以直接复用的现有对象

- `review_decisions`
- `review_item_comments`

它们现在本来就是按：

- `target_type`
- `target_id`

设计的。

因此最小改法是：

- 在代码层允许 `target_type = matrix_item`
- 不新建第二套 review/comment 表

#### Review List 的最小扩展

现有 `Review List` 建议扩展为：

- `card`
- `excluded`
- `matrix_item`

每个 `matrix_item` 也支持：

- `accept`
- `reject`
- `keep_for_later`
- `needs_manual_check`
- `comment`

### 5.7 Export

继续复用现有 export service，但新增一种 formatter：

- `card export formatter`
- `matrix export formatter`

不建议新建第二套 export 基础设施。

#### `matrix` 的推荐导出结构

每个维度按以下结构导出：

- 维度名称
- 核心结论
- 代表文献
- 证据强度
- 支持 / 混合 / 反证
- 边界条件 / limitations

### 5.8 观测与评估

需要新增的不是大而全 dashboard，而是最小可见性：

- 当前 `run` 的 `task_type`
- plan 是否已确认
- 当前使用的 active memory 版本
- `matrix item` 数量
- `matrix item` 的 review 分布
- 每个维度是否达到最小 evidence coverage

同时应把 `matrix item` 纳入后续 calibration / evaluation 体系，而不是只让 `card` 有校准逻辑。

---

## 6. 额外功能一：对单篇论文直接向 AI 提问

### 6.1 第一性原理

“问一篇论文的问题”不是 discovery 任务，也不是 `aha judgement` 任务。

它本质上是：

> 已知 corpus 内的 retrieval + answer synthesis

因此它不需要一套新的大系统，只需要：

- 取到这篇论文的全文段落
- 检索与问题最相关的 sections
- 把相关 sections 和可用 figure/caption 交给 LLM
- 要求 LLM 基于证据回答，并显式说出证据不足的情况

### 6.2 最小可行方案

#### 输入

- `paper_id`
- `question`

#### 执行

- 从 `paper_sections` 中取全文
- 做一次轻量 retrieval
  - 优先可用 hybrid：embedding + keyword overlap
  - 如果实现成本过高，V1 可以先用 lexical retrieval
- 选出 top-k sections
- 附带相关 figure caption（如果命中）
- 调用 LLM 生成回答

#### 输出

- `answer`
- `used_sections`
- `used_figures`
- `confidence_note`
- `cannot_answer_from_paper` 标记

### 6.3 V1 不建议做的事

- 不做多轮 persistent chat thread
- 不做跨论文 agent 搜索
- 不做自动生成课程结论
- 不做隐式使用外部知识且不标注来源

### 6.4 推荐 API / UI

推荐新增：

- `POST /api/papers/{paper_id}/qa`

UI 入口建议放在：

- card detail
- matrix item detail
- single paper validation

都可跳转到同一个 `Ask This Paper` 面板。

---

## 7. 额外功能二：基于 reaction 的“记忆”与自迭代

### 7.1 第一性原理

这里的“记忆”不应该被理解成一个神秘的黑盒状态。

它的本体应当是：

> 从用户的显式 reaction 中提炼出的、可审查的、可激活/停用的偏好规则摘要

所以它不是：

- 在线强化学习
- 模型权重微调
- 黑盒自动优化

它应当是：

- review decisions 的结构化提炼
- comments 的规则化归纳
- 可被用户确认的 `memory draft`

### 7.2 可直接复用的现有信号源

当前系统已经有：

- `review_decisions`
- `review_item_comments`
- `accepted / rejected / keep / needs_manual_check`
- `calibration_sets`
- `evaluation_runs`

这意味着 V1 不需要另建一个巨大的“memory database”。

### 7.3 推荐的最小方案

#### Memory 的 source of truth

以这些现有数据为准：

- `card` 的 review decisions
- `excluded` 的 reopen / confirm exclusion
- `matrix_item` 的 review decisions
- 所有 persistent comments
- export 时被最终选入的对象

#### Memory 的产出形态

系统不直接“自动改 prompt”，而是先产出一个 `Memory Update Draft`：

```json
{
  "scope": "project",
  "mode": "claim_evidence",
  "prefer": [
    "Prefer empirical papers with direct outcome linkage.",
    "Prefer evidence that names boundary conditions explicitly."
  ],
  "avoid": [
    "Avoid pure taxonomy recaps.",
    "Avoid indirect background theory without outcome linkage."
  ],
  "review_signals": [
    "User repeatedly rejects context-only evidence as too weak.",
    "User repeatedly accepts mixed-evidence items when limitations are explicit."
  ]
}
```

#### Memory 的应用位置

推荐只作用于三个位置：

1. `Draft Search Plan`
2. `Judgement / filtering`
3. `Matrix generation`

V1 不建议让 memory 直接改动：

- parser
- access queue
- database schema
- export selection

### 7.4 为什么要坚持“先 draft，再激活”

如果让系统自动把 reaction 直接变成生效策略，会有四个风险：

1. 一次偶然决策被永久放大
2. 临时偏好污染长期判断
3. 你无法知道系统到底学到了什么
4. 出现回退时无法定位问题

因此推荐机制是：

- 系统先起草 memory update
- 用户确认是否激活
- 只有被确认的 memory 才影响后续任务

### 7.5 V1 的最佳最小化形态

V1 不建议一开始就做专门的新表。

最简路径是：

- 运行时从 `review_decisions + comments` 派生 `memory draft`
- 把当前生效的 `memory snapshot` 写入 `run.metadata_json`
- 在 prompt 和 plan 中把该 snapshot 作为显式上下文传入

如果后续 memory 版本管理需求增大，再引入独立 `preference_profiles` 表。

### 7.6 与 calibration 的关系

`memory` 解决的是“你的偏好如何进入未来任务”。

`calibration` 解决的是“系统改完以后是否回退”。

两者不应混为一谈，但应连接起来：

- memory update 被激活后
- 后续 prompt / planner 变化应能进入 evaluation
- 避免“学会了你的偏好，却破坏了整体质量边界”

---

## 8. 决策闭环设计

这是本次方案里最需要你拍板的一步。

### 8.1 推荐的闭环

推荐采用：

> **Human-confirmed closed loop**

流程如下：

1. 用户输入 brief
2. 系统起草 `Draft Search Plan`
3. 用户确认 `Confirmed Plan`
4. 系统执行 run
5. 用户 review `card / matrix item`
6. 系统起草 `Memory Update Draft`
7. 用户选择：
   - `Activate`
   - `Reject`
   - `Activate with edits`
8. 下一个 run 明确显示“本次使用了哪个 active memory”

### 8.2 为什么不推荐“自动闭环”

不推荐：

> 用户一 react，系统立刻自动改后续策略

原因：

- 不可审查
- 不可回滚
- 不可解释
- 误学一次，后续全偏

### 8.3 推荐的状态机

建议最小状态为：

- `brief_ready`
- `plan_drafted`
- `plan_confirmed`
- `run_running`
- `outputs_reviewable`
- `memory_update_drafted`
- `memory_active`

### 8.4 闭环里唯一必须人工确认的两步

如果要压缩到最小，真正必须人工确认的只有两步：

1. `Confirm Search Plan`
2. `Activate Memory Update`

其余都可以自动推进。

---

## 9. 最小实现路径（MVP Build Order）

### Phase 1：把“任务对象”补出来

- 新增 `task_type`
- 新增 `Draft Plan` / `Confirm Plan`
- `run.metadata_json` 中记录：
  - `task_type`
  - `task_brief`
  - `draft_plan`
  - `confirmed_plan`

### Phase 2：跑通 `claim_evidence`

- 为 `claim_evidence` 增加专用 discovery profile
- 新增 `evidence_matrix_items`
- 新增 `matrix item` generation
- 把 `matrix item` 接进统一 review list

### Phase 3：加上 shared paper QA

- 新增 `POST /api/papers/{paper_id}/qa`
- UI 增加 `Ask This Paper`
- 返回 section-level citations

### Phase 4：加上 memory draft / activation

- 从 `review_decisions + comments` 生成 `memory draft`
- 用户确认后写入 active memory snapshot
- 后续 plan / judgement 使用该 snapshot

### Phase 5：补最小 observability 和 evaluation

- 显示当前 run 的 `task_type`
- 显示 active memory 是否生效
- 显示 matrix coverage / acceptance metrics
- 把 `claim_evidence` 纳入 evaluation

---

## 10. 明确不建议在 V1 做的事

为了保证方案高效跑通，以下能力建议明确延后：

- 纯 LLM 自动 mode 决策并自动执行
- 多轮 autonomous query rewriting
- 多轮持久化 paper chat session
- 独立的第二套 review 系统
- 独立的第二套 export 基础设施
- 黑盒自动学习和自动 prompt 改写
- 向量数据库重构
- 复杂 recommendation / ranking 学习器

---

## 11. 代码落点建议

如果按本方案推进，最可能改到这些文件：

- `app/static/index.html`
  - 新增 `Draft Plan`
  - 新增 `Mode`
  - 新增 `matrix item` review 展示
  - 新增 `Ask This Paper`
- `app/main.py`
  - 新增 plan draft endpoint
  - 扩展 run create
  - 新增 paper QA endpoint
  - 新增 memory draft / activate endpoint
- `app/schemas.py`
  - 新增 `task_type`
  - 新增 plan payload
  - 新增 matrix item request/response
  - 新增 paper QA request
- `app/services.py`
  - planner
  - `claim_evidence` discovery profile
  - matrix item persistence / listing / review
  - paper QA service
  - memory distillation service
- `app/llm.py`
  - `draft_search_plan`
  - `generate_matrix_items`
  - `answer_paper_question`
  - `distill_preference_memory`
- `app/db.py`
  - 新增 `evidence_matrix_items`
  - 或最少量 migration
- `tests/test_app.py`
  - 覆盖 plan confirm
  - matrix item review
  - paper QA
  - memory activation

---

## 12. 需要你确认的决策点

以下是我认为必须由你拍板的地方。

### D1. 是否确认总架构采用“显式双 mode + LLM 建议 + Confirm Plan”

推荐答案：

- `是`

原因：

- 这是最稳、最可解释、最便于 debug 的方案。

### D2. 是否确认 `claim_evidence` 的原子输出对象是 `matrix item`，而不是整块 narrative

推荐答案：

- `是`

原因：

- 只有原子化后，review、memory、export 才能精细工作。

### D3. 是否确认 `claim_evidence` 默认必须显式展示四类信息

- supporting
- mixed
- contradictory
- limitations

推荐答案：

- `是`

原因：

- 否则系统会天然滑向 cherry-picking。

### D4. 是否确认 `Confirm Plan` 是真正的硬门槛

推荐答案：

- `Aha`：可以是一键确认
- `Claim Evidence`：必须人工确认后再跑

### D5. 论文问答功能，V1 是否接受“单轮、无持久线程、基于单篇全文回答”

推荐答案：

- `是`

原因：

- 这是最小但已足够可用的版本。

### D6. Memory 的作用域先做哪种

可选项：

- `project-level` 全局记忆
- `reviewer-level` 个人记忆
- `project + reviewer overlay`

推荐答案：

- `project-level`

原因：

- 最省实现成本。
- 如果当前主要就是一个决策人，这已经足够贴近“你的偏好”。

### D7. Memory update 是否允许自动生效

推荐答案：

- `否`

推荐机制：

- 系统只起草 `Memory Update Draft`
- 由你确认后才激活

这是本方案里我最建议你明确拍板的地方。

### D8. Memory 在 V1 先作用于哪些环节

可选项：

- `Draft Search Plan`
- `Judgement / filtering`
- `Matrix generation`
- `Discovery ranking`

推荐答案：

- 先作用于：
  - `Draft Search Plan`
  - `Judgement / filtering`
  - `Matrix generation`

不建议 V1 直接改 discovery provider ranking。

### D9. `claim_evidence` 跑完后，是否还需要顺手产出 `aha cards`

可选项：

- `默认不产出`
- `默认同时产出`
- `用户勾选时产出`

推荐答案：

- `用户勾选时产出`

原因：

- 默认同时产出会增加噪音和算力成本。

### D10. `matrix item` 是否进入现有统一 Review List，而不是单独开新页面

推荐答案：

- `是`

原因：

- 复用现有 review ledger，最省成本。

### D11. Export 是否沿用同一套 Google Docs 基础设施，只新增 `matrix formatter`

推荐答案：

- `是`

原因：

- 不值得新建第二套 export infra。

---

## 13. 我的最终建议

如果只保留一句话：

> 不要把“定向搜索”理解为在当前 pipeline 里再塞一个 prompt；应该把它建模成与 `Aha exploration` 并列的第二种任务类型，但继续复用同一套 discovery / parsing / review / export 底座。

最小可跑通方案就是：

1. 显式双 mode
2. 先 Draft Plan，再 Confirm Plan
3. `claim_evidence` 新增 `matrix item`
4. 论文问答用现有 `paper_sections` 直接支撑
5. reaction 不直接自动学习，而是先形成 `Memory Update Draft`，由你确认后再生效

如果这个方向确认，下一步最合理的工作不是马上写全量代码，而是：

1. 先拍板 `D1-D11`
2. 先确定 `claim_evidence` 的 plan schema 和 `matrix item` schema
3. 再按 `Phase 1 -> Phase 5` 顺序落地
