<!--
This file summarizes the implemented dual-mode research system changes.
Main sections: shipped capabilities, validated cases, current limits, and local verification steps.
Data shown: API/UI surfaces, test coverage, and remaining live-environment checks.
-->
# 双模式研究系统实施总结

## 已落地能力

1. 双模式 planning 已上线
   - 新增 `POST /api/research-plans/draft`
   - 支持 `aha_exploration` 与 `claim_evidence`
   - run 创建时支持 `task_type`、`research_brief`、`confirmed_plan`、`use_active_memory`
   - claim mode 可从 confirmed plan 自动派生 topics，不再要求手工先写 topics

2. claim evidence 主链已跑通
   - 新增 `evidence_matrix_items` 表
   - 新增 matrix item 的生成、列表、详情、review、comment、导出
   - export 面板可在 `cards` / `matrix_items` 间切换
   - topic run summary 已显示 `matrix` 数量

3. 跨模式 paper QA 已上线
   - 新增 `POST /api/papers/{paper_id}/qa`
   - 基于 `paper_sections` 做 retrieval
   - 返回 grounded answer、used section ids、used figure ids

4. 偏好记忆闭环已上线
   - 新增 `GET /api/memory/active`
   - 新增 `POST /api/memory/draft`
   - 新增 `POST /api/memory/activate`
   - 只有显式 activate 后，memory 才会进入下一次 planning / filtering

5. UI 已覆盖关键闭环
   - Start Run 面板支持 Draft / Confirm Plan
   - Review List 支持 matrix item
   - Single Paper Validation 面板支持 Ask Paper
   - 新增 Preference Memory 面板

## 有效性案例

1. claim plan draft
   - 用 research brief 直接生成 claim-mode plan
   - plan 内含 `search_topics -> dimension_label -> query_anchor -> outcome_terms`
   - confirm 后可直接起 run

2. claim run 自动继承 active memory
   - active memory 存在时，`/api/runs` 会把 snapshot 写入 `run.metadata`
   - 后续 matrix generation / card filtering 会读取该 snapshot

3. matrix item review/export
   - accepted matrix item 可在 review list 里显示为独立对象
   - 可直接走 export 生成 markdown / Google Doc artifact

4. paper QA grounded answer
   - 对指定 paper 提问时，接口会返回 answer 和引用的 section ids
   - 避免“只回答，不给出处”的黑盒状态

5. memory closed loop
   - review decisions + comments 可生成 memory draft
   - 用户确认后 activate，下一次 plan 与过滤才会受影响

## 测试结果

- 重点新增测试：
  - `test_research_plan_draft_api_returns_claim_plan`
  - `test_claim_run_uses_confirmed_plan_topics_and_active_memory`
  - `test_review_items_api_supports_matrix_items`
  - `test_matrix_export_endpoint_builds_google_doc_artifact`
  - `test_paper_qa_endpoint_returns_grounded_answer`
  - `test_memory_draft_and_activate_endpoints`
- 主回归：
  - `python3 -m pytest tests/test_app.py -k "not test_figure_asset_endpoint_serves_validated_local_asset and not test_parser_materializes_html_data_uri_figure_assets"`
  - 结果：`110 passed, 1 skipped, 2 deselected`

## 当前剩余风险

- 真实 LLM provider 联调还没在联网环境做 live 验证
- 真实 discovery provider 的 claim-mode 检索效果还需要你本机联网实测
- 真实 `gws` 导出还没在已登录环境做 live create / append
- 未安装 `Pillow` 的环境下，2 条 figure asset 测试仍会失败，这不是本次双模式改动引入的回归

## 建议的本地验收顺序

1. 先在 UI 里用一条 claim brief 做 `Draft Plan -> Confirm Plan -> Start Run`
2. 等 run 完成后检查 review list 是否出现 matrix items
3. 任取一条 matrix item 做 `Accept -> Export`
4. 对其中一篇 paper 走一次 `Ask Paper`
5. 在 review 后执行 `Draft Memory Update -> Activate Draft`
6. 再起第二个 run，确认 active memory snapshot 已进入 metadata
