<!--
This file records handoff-ready project progress for future sessions.
Main sections: completed work, pending work, known issues, and next starting point.
-->
# Progress

## 本次做了什么

- 将 `PRD.md` 落成开发基线。
- 实现了 Phase 0 的最短闭环：
  - 多主题批量输入
  - 本地 PDF 路径输入
  - 并行 topic run
  - 证据段落抽取
  - 候选卡片生成
  - 初始 judgement
  - 内部 review list
  - Google Doc 导出工件
- 搭建了一个可运行的 FastAPI 应用和最小内部操作页。
- 建立了 SQLite 持久层，覆盖 runs、topics、topic_runs、papers、paper_sections、candidate_cards、judgements、review_decisions、access_queue、exports 等核心记录。
- 增加了 Phase 0 API 集成测试，验证了 `本地 PDF -> run -> card -> review -> export` 的完整调用链。
- 处理了一个插队需求：
  - 首页的 `Metadata JSON` 现在预填安全默认值 `{}`
  - 前端会在启动 run 前拦截非法 JSON 或非对象类型 metadata
  - `PRD.md` 已补充 metadata 默认值和填写约束
- 修复了 PDF 乱码出卡的根因链：
  - 远程资产现在根据文件魔数识别真实类型，不再把 `.03314` 这类 PDF 误当 HTML
  - PDF 解析改为只提取 PDF 文本指令中的字面字符串，不再对整个文件跑 `strings`
  - 文本提取进一步收紧为只扫描 `BT ... ET` 文本对象，不再在整个 PDF 二进制里全局搜 `Tj/TJ`
  - 增加 fragment 级噪声过滤，显式拦截 `JFIF`、`MATLAB Handle Graphics`、`endstream` 等图片/对象流伪文本
  - 新增 `MarkItDown` 集成作为 PDF 主提取 fallback；如果本地已安装 `markitdown[pdf]`，会优先把 PDF 转成 Markdown，再走质量闸门和 section 切分
  - Markdown 里的图片引用会被解析成 `figures` 占位对象，并链接回解析出的 sections，避免图片信息污染正文
  - 加入解析质量闸门，检测不到正文或检测到 PDF 结构噪声时直接 `parse_failed/quality_failed`
  - 解析失败的 paper 不再生成 card，且当前 run 会清掉该 paper 的旧坏卡
- 增加了针对性测试：
  - 非 `.pdf` 后缀但实际是 PDF 的文件，仍会按 PDF 解析
  - 损坏/噪声 PDF 会在出卡前被拦截
  - 含合法正文但同时夹带 `JFIF`/非文本流伪 `Tj` 的 PDF，会忽略噪声并保留正文
  - 已增加 `MarkItDown` stub 测试，确认 PDF Markdown 提取和图片引用解析路径生效
  - 真实问题 artifact 现在会被 parser 阻断，不再继续出乱码卡
- 将 card generation / judgement 接成了可插拔 LLM 链路：
  - 新增 OpenAI 兼容 API 客户端和 LLM card engine
  - 默认 `P2B_LLM_MODE=disabled` 时仍可运行
  - 配置 `P2B_LLM_MODE=openai_compatible` 后，可用大模型直接生成 card 和 judgement
  - 现在已改为 `LLM-only`：卡片必须经过 LLM 生成，不再存在 heuristic card fallback
  - 增加了 stub 测试，确认 pipeline 真正会采用 LLM card engine 产出的卡片
  - LLM 请求失败时不再偷偷退回启发式建卡；相反，会把失败原因记录到 paper 级状态里，并显式返回 `0 cards`
  - LLM 网络错误信息现在会带 endpoint / host / HTTP body 上下文，避免出现空的 `LLM request failed:` 报错
  - 对 `openai_compatible`、`anthropic`、`gemini` 增加了合理默认 base URL，减少因 `.env` 少填一项导致的隐性失败
  - 新增 `papers.card_generation_status` / `papers.card_generation_failure_reason`，用于追踪每篇论文为什么没出卡
- 将 LLM 配置扩展为 `.env` 风格和三家原生 provider：
  - `app/config.py` 现在会自动加载项目根目录 `.env`
  - provider 现支持 `openai_compatible`、`anthropic`、`gemini`
  - 各 provider 的响应格式已分别做测试覆盖
  - `PRD.md` 已补充 `.env` 风格和多 provider 配置要求
- 补上了 LLM 联调闭环的 smoke-test 能力：
  - 新增 `/api/llm/smoke` 路由
  - 新增 `scripts/live_llm_smoke_test.py`
  - 增加 stub 测试，确认 smoke test 会产出规范化卡片结果
  - 修复脚本直接运行时的模块搜索路径问题，`python3 scripts/live_llm_smoke_test.py` 不会再因为 `ModuleNotFoundError: app` 失败
- 修复了 `.env` 读取在受限环境里会直接抛 `PermissionError` 的问题，当前会安全降级
- 根据用户提供的优秀卡片案例，补充了 `PRD.md` 的插队需求：
  - 增加 `Aha First, Not Summary First` 原则，明确系统目标是抽取 learner-facing insight，而不是论文摘要
  - 增加 card 内容质量闸门，要求拦截“摘要型、背景型、迁移过远型”伪卡
  - 增加“每篇论文输出被排除内容及理由”的要求，避免只产出通过项、不保留判断过程
  - 增加正样本 / 负样本 / 边界 case 的校准集要求，后续 prompt 和 rubric 改动必须围绕这套校准集验证
- 按新的插队需求完成了一轮代码级主链改造：
  - `LLMCardEngine` 不再只返回 `cards`，而是统一返回 `cards + excluded_content`
  - LLM 输出 schema 现在要求每张卡都包含 `teachable_one_liner` 和逐条 `evidence_analysis`
  - 归一化层会直接拦截缺少可教学一句话或缺少证据分析的弱卡，避免“会总结、不会出 aha 卡”的输出混进系统
  - 新增 `paper_excluded_content` 持久层表，把每篇论文中“明确不出卡的内容及理由”作为正式数据存下来
  - `candidate_cards` 新增 `teachable_one_liner` 字段
  - `Repository` / `PaperPipeline` 已改成同时落库 cards 和 excluded content
  - `/api/cards/{card_id}` 返回中现在会附带同论文同 topic/run 的 `excluded_content`
  - 导出工件现在会带上 `Teach it as` 和每条 evidence 的 `Why it matters`
  - 内部页面的 card detail 现在会显示 evidence analysis 和 excluded content，而不只是原始 JSON
  - 已新增测试，覆盖：
    - excluded content 的存储与读取
    - `teachable_one_liner` 必填的内容质量闸门
    - 新 smoke test 输出结构
    - 现有主流程不回归
- 将“从根上解决卡片内容校准问题”的方案继续写入 `PRD.md` 作为插队需求：
  - 明确要求把正样本 / 负样本 / 边界 case 变成正式 calibration corpus，而不是停留在文档
  - 明确要求将 candidate extraction 与 judgement 拆成两个阶段
  - 明确要求将 excluded content 升级为 first-class review object，而不只是 detail 中可见
  - 明确要求建立 evaluation loop，用评估运行而不是感觉来比较 prompt / rubric / model 变化
  - 已同步到验收标准、Phase 0 interrupt requirement 和详细 TODO 列表

## 还没做什么

- 多源在线检索的真实联网验证
- 真正的 Google Docs 在线写入验证
- 更强的 PDF 图表抽取
- 后续阶段的 calibration / saturation 实现

## 已知的问题和 bug

- 当前环境未确认可直接联网访问论文源。
- 当前环境未安装 `gws`，Google Docs 导出需要先走工件模式或后续补 CLI/API 凭证。
- PDF 图表抽取在 Phase 0 只保留了接口和空结果，不是完整实现。
- `MarkItDown` 解决的是 PDF 正文/Markdown 转换问题，不等于完整 figure asset extraction；当前只保留 Markdown 图片引用和 caption 级信息，尚未把 PDF 内嵌图片导出为独立文件。
- 当前沙箱环境禁止本地端口绑定，因此本次没有完成真实浏览器访问验证；已完成 API 层完整调用链自测。
- 当前 PDF 解析已改为 fail-closed，不会再把明显乱码推进到 card 层；但对复杂压缩 PDF，仍可能因无法可靠提取正文而直接 `parse_failed`。
- 已存在于数据库里的历史坏卡片不会自动消失；新 parser 只影响修复后的新 run，老 run 需要重新跑或手动清理。
- 当前 LLM 路径已支持 OpenAI-compatible、Anthropic、Gemini，但本环境未做真实联网联调；已用 stub 测试验证集成逻辑。
- 当前沙箱限制阻止直接写入新的 `.env` / `.env.example` dotfile；代码已支持 `.env` 加载，但实际密钥文件需要在本地环境落盘。
- 当前沙箱既不能直接读你的 `.env` 内容，也无法从这里连到你本地启动的 `127.0.0.1:8000` 服务，因此真实 provider 的 live smoke call 无法在这个受限执行环境里完成。
- 如果本地调用 `POST /api/llm/smoke` 返回 `404 Not Found`，优先判断为本地 `uvicorn` 仍在运行旧代码；需要重启服务进程才能加载新路由。
- 已确认一次真实本地失败根因是 LLM provider endpoint 的 DNS / base_url 问题，而非 PDF 解析问题；当前已改为 LLM-only，因此 provider 配置不通时会显式 `0 cards`，不会再混入 heuristic 卡。
- 当前工作区不是 git repository，无法按常规读取 `git log`；如果后续需要版本历史，需要确认外层仓库结构。
- 这轮内容校准改造仍然只是第一阶段：
  - 现在已经有了 `cards + excluded_content` 的统一输出契约，但还没有接入真正的正样本 / 负样本 / 边界 case 校准集
  - 当前“挡弱卡”的机制仍以 LLM schema 约束 + 结构化质量闸门为主，距离真正贴合老板审美的 judging 还有一段路
  - review 列表页还没有把 excluded content 做成单独列表或更强筛选器，目前主要在 card detail 中可见
- 当前目录最初不是 git repository；如果要发布到 GitHub，需要先初始化 git、配置 remote、确认忽略规则，再进行首次提交和推送
- 已补 `.gitignore`，新增忽略 `.venv/`，避免首次发布时把本地虚拟环境误提交
- 当前受限执行环境阻止创建 `.git` 目录，因此无法在这里完成 `git init` 和后续推送；如果继续发布，需要在本地终端完成初始化与 push

## 下次开始时应该先做什么

- 先跑一遍 `python3 -m unittest tests/test_app.py` 确认 Phase 0 没坏。
- 如果要验证 LLM-only 主链路，优先在 `.venv` 里启动服务，并确认 `.env` 里的 provider/base_url/model/api_key 可解析、可联网。
- 如果本地 PDF intake、card 生成和 export 仍正常，再接 discovery 的真实联网联调。
- 如果需要老板可直接打开的 Google Doc，再接 `gws` 或官方 Docs API 写入流程。
- 后续如果要提高 PDF 成功率，优先接入正式 PDF 解析库（如 `pypdf` 或 `PyMuPDF`），在保留当前质量闸门的前提下提升正文召回。
- 先在本地补上 `.env`，再用真实 LLM API 对少量 topic 做联调，并根据输出修 prompt/schema。
- 下一次若继续做 live 联调，优先在你本地终端直接执行 `python3 scripts/live_llm_smoke_test.py` 或调用 `/api/llm/smoke`。
- 下一次如果继续提高出卡质量，优先做内容层校准：先整理 5-8 张强正样本、5-8 张强负样本、5-10 个边界 case，再据此改 LLM prompt 和评审 rubric，而不是继续盲调模型。
- 下一次如果继续做内容校准，先跑一遍当前 20 个测试，再补：
  - 正样本 / 负样本 / 边界 case 的持久化结构
  - prompt 变更前后的校准集回归测试
  - review UI 对 excluded content 的筛选与批量审阅能力
- 如果下一次继续做 GitHub 发布之后的开发，先确认远端主分支和默认分支名称，再按 `[module] description` 规范继续提交
