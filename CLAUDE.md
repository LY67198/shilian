# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**试炼（MockPilot）** — 企业级 AI 模拟面试平台。围绕"简历解析 → 岗位匹配 Agent → RAG 题库出题 → 多轮 LangGraph 面试 → AI 评分 → 结构化报告"构建的完整求职训练闭环。站在候选人立场，强调"反复练习锻造面试能力"。

技术栈：FastAPI + LangGraph + LangChain + Milvus + PostgreSQL + Redis + DeepSeek + DashScope。

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI 0.115 + SQLAlchemy 2.x (async) + Alembic |
| 业务数据库 | PostgreSQL 16（结构化数据 + JSONB） |
| 向量数据库 | **Milvus 2.4.10** standalone（embedded etcd + 外部 MinIO），embedding 维度 1024 |
| 缓存/队列 | Redis 7 + Celery 5.5（异步任务、Beat 调度） |
| 对象存储（Milvus 内部） | MinIO（standalone dev） |
| 监控 | Flower 5555（profiles: monitoring） |
| 反向代理 | Nginx (8080/8443) — 仅生产环境 |
| AI 模型 | DeepSeek（OpenAI 兼容 SDK 2.x）、DashScope Embedding (text-embedding-v3, 已归一化) |
| Agent 框架 | LangChain 1.3 + **LangGraph 1.2.8**（Postgres checkpoint 3.1.0 已装） |
| 测试 | pytest 8.3 + pytest-asyncio + httpx |
| 前端 | Vue 3.4 + Vite 5.4 + Pinia 2.1 + Vue Router 4.3 |
| 部署 | Docker Compose，外部端口 5434 / 6382 / 19530 / 9091 |

## 目录结构

```
ai-interview-agent/
├── ai-interview-backend/          # FastAPI 后端（容器化）
│   ├── app/
│   │   ├── api/                   # 接口模块（client/backoffice 拆 v1）
│   │   ├── route/                 # 路由注册中心（router_registry + create_app）
│   │   ├── core/                  # config / celery_app / security / log_config
│   │   ├── configs/               # 双 Swagger 应用拆分
│   │   ├── db/                    # SQLAlchemy async session
│   │   ├── models/                # SQLAlchemy 模型（**embedding 列已迁出**）
│   │   ├── schemas/               # Pydantic 请求/响应模型
│   │   ├── services/              # 业务逻辑（auth/email/redis_verification 等）
│   │   ├── llm/                   # 🆕 LangChain 原子能力（LLM/Embedding/Prompt 工厂）
│   │   ├── prompts/               # 🆕 PromptTemplate YAML 集中管理（8 个文件）
│   │   ├── repositories/          # 🆕 Repository Pattern 数据访问层
│   │   ├── workflows/             # 🆕 LangGraph 编排层
│   │   │   └── _shared/           # 共享基础设施（checkpointer/state/tracing/llm/tools/sse）
│   │   │       └── sse.py         # astream_to_sse() LangGraph → SSE 封装
│   │   │   ├── interview/         # ✅ Phase 2 面试评估（HITL StateGraph）
│   │   │   │   ├── nodes/         # fetch_context / retrieve_knowledge / evaluate / check_finished / ask_question / generate_report
│   │   │   │   ├── state.py       # InterviewState + ScoreResult
│   │   │   │   ├── graph.py       # build_interview_graph() + get_compiled_graph()
│   │   │   │   └── service.py     # InterviewGraphService.submit_answer() 单一入口
│   │   ├── vector_db/             # 🆕 Milvus 客户端 + collections schema + CRUD
│   │   │   ├── client.py          # MilvusClient 单例 + health_check
│   │   │   ├── index.py           # HNSW + L2 索引配置
│   │   │   └── collections/
│   │   │       ├── knowledge.py   # knowledge_chunks collection
│   │   │       └── question_bank.py # question_bank collection
│   │   ├── common/                # language / log_consumer / release
│   │   ├── exceptions/            # APIException / 错误码
│   │   ├── utils/                 # 通用工具
│   │   └── schedule/              # Celery 定时任务
│   ├── tests/                     # pytest 测试
│   ├── migrations/                # Alembic（含 drop_pgvector_embedding_columns）
│   ├── scripts/
│   │   ├── create_first_admin.py
│   │   ├── seed_position_templates.py
│   │   ├── init_milvus.py         # 🆕 建 Milvus collections + 索引
│   │   └── deploy.sh
│   └── docker-compose*.yml        # 含 minio + milvus service
├── ai-interview-frontend/         # 用户端 (localhost:3000 → :8006)
└── ai-interview-admin/            # 管理端 (localhost:3001 → :8006)
```

## 分支策略

| 分支 | 用途 | 谁推 |
|------|------|------|
| `main` | **发布版**（已发布、公开可见） | 仅在显式指定时推 |
| `dev` | **重构 / 新功能开发**（默认） | 日常开发默认推这里 |

**约定**：
- 重构 / Phase 1-4 实施 / bug 修复 → 全部推 `dev`
- 推 `main` 必须显式说"推 main"（或 `git push main` 显式命令）
- 默认 `git push`（无参数）只推当前分支

`.git/config` 已配置：
- `branch.dev.remote = github`（默认 push 目标）
- `push.default = current`（只推当前分支到同名远端分支）

**未公开的内部文档**（CLAUDE.md / `docs/` / 部署教程 / 各种规划 md）通过 `git add -f` 强制提交到 dev，但**永远不进 main**。



## 核心约束

- **必须** 有 DeepSeek API Key 和 DashScope API Key 才能运行
- **必须** 通过 Docker Desktop 启动后端（postgres + redis + minio + milvus + app + celery-worker + celery-beat）
- 后端跑在 Docker 容器中，前端跑在 Windows 本地
- 两个前端 `vite.config.js` proxy 都指向 `http://localhost:8006`
- **embedding 维度 1024 不可随意修改**（与 Milvus 索引绑定）
- **embedding 存 Milvus，不存 PostgreSQL**（已迁移；旧 embedding 列已 drop）
- **JWT 区分 scope**：client / backoffice 两套 token
- **Swagger 双套**：`/client/docs`（无需认证）、`/backoffice/docs`（需 JWT）
- **路由统一注册**：所有路由通过 `app/route/router_registry.py` 集中管理
- **测试在容器内跑**：dev compose 把 `./tests` 和 `./pytest.ini` 挂进 `ai-interview-app`

## LangChain / LangGraph 分工（硬性规则）

| 层级 | 职责 | 位置 |
|------|------|------|
| **LangChain**（原子能力） | LLM / Embedding / @tool / Message / PromptTemplate | `app/llm/`, `app/prompts/` |
| **LangGraph**（编排） | 多步骤流程、StateGraph、Agent 决策、条件分支、Checkpointer | `app/workflows/` |
| **Node 纯函数** | 业务逻辑，可独立测试，调用 repo + llm | `app/workflows/<name>/nodes/` |
| **Repository** | 纯 DB CRUD，不调 LLM / Milvus / HTTPException | `app/repositories/` |

**禁止**：
- ❌ 业务代码里直接 `from openai import OpenAI` — 用 `app.llm.get_chat_llm()`
- ❌ Prompt 写在 Python 代码里 — 用 `app.prompts/*.yaml` + `load_prompt()`
- ❌ 业务 service 里手动管 LangGraph state — 让 `app/workflows/_shared/` 接管
- ❌ 节点函数里直接写 SQL — 用 `app/repositories/`

**新功能必须**写在 `app/workflows/<name>/`（graph + nodes + service），不写在 `app/services/`（除 `auth/`、`email/` 等非 AI 业务）。

**Phase 2 重构硬性规则**：
- ❌ 禁止新增 `@staticmethod` — 新 service 用实例方法 + FastAPI Depends 注入
- ❌ 禁止手写 `yield f"data: {json.dumps(...)}\n\n"` — SSE 用 `sse-starlette` 或 `StreamingResponse` 封装
- ❌ 禁止 `re.search(r'\{.*"score".*\}', text)` 从 LLM 输出抠 JSON — 用 `with_structured_output(PydanticModel)` 拿强类型结果
- ❌ 禁止 `submit_answer(stream=False)` 和 `submit_answer_stream` 分两个方法 — 合并为 `submit_answer(stream: bool = False)`
- ❌ 禁止手写 `yield f"data: {json.dumps(...)}\n\n"` — SSE 用 `app/workflows/_shared/sse.py:astream_to_sse()` 封装
- ❌ 禁止 `isinstance(llm_output, AIMessage)` 后取 `.content` 再 `json.loads()` — 评分节点用 `with_structured_output(PydanticModel)` 拿强类型结果
- ❌ 禁止 `_extract_json` 抛 `ValueError` — 已改为返回带 `parse_failed: True` 的 fallback dict

## Milvus 向量库使用规则

- **唯一入口**：`app/vector_db/collections/{knowledge, question_bank}.py`
- **距离度量：L2**（不用 COSINE — Milvus 2.4.10 的 HNSW+COSINE 有 bug）
- **DashScope embedding 已归一化**：L2 distance 对归一化向量等价于 cosine 距离
  - `similarity = 1 - distance/√2`（distance=0 → sim=1，distance=√2 → sim=0）
  - 实现见 `app/vector_db/index.py:l2_distance_to_similarity()`
- **insert 必须 flush**：Milvus 默认 insert 后数据在写缓冲，必须 `client.flush(collection_name)` 才能被 search 命中。`insert_chunks/insert_questions` 已内置
- **必须 load_collection**：search 前 collection 要在内存，已在 `create_collection` 里处理
- **ID 与 PostgreSQL 同源**：`knowledge_chunks.id` / `question_bank.id` 在 PG 拿 auto-increment，再写入 Milvus 同 id；保证召回结果可追溯到 PG 元数据

## 常用命令

> ⚠️ 容器名从 `ai-interview-*` 重命名为 `shilian-*`（2026-07-10 品牌升级）

```bash
# 后端：构建并启动所有容器（minio + milvus + app + 3 workers）
cd ai-interview-backend
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build

# 启动 Flower 监控（可选）
docker compose --profile monitoring up -d flower   # → http://localhost:5555

# 数据库初始化
docker exec shilian-app alembic upgrade head
docker exec shilian-app python scripts/create_first_admin.py
docker exec shilian-app python scripts/seed_position_templates.py

# Milvus 初始化（建 collections + HNSW L2 索引 + load 到内存）
docker exec shilian-app python scripts/init_milvus.py

# 验证后端健康
curl http://localhost:8006/api/v1/config/health

# Swagger 文档
# 用户端：http://localhost:8006/client/docs
# 管理端：http://localhost:8006/backoffice/docs
# OpenAPI JSON：/api-docs/client.json  /api-docs/backoffice.json

# Milvus 健康（standalone 自带 etcd + MinIO）
curl http://localhost:9091/healthz   # Milvus metrics
curl http://localhost:9000/minio/health/live   # MinIO

# 测试（容器内）
docker exec shilian-app pytest -m "unit"          # 仅单元测试
docker exec shilian-app pytest -m "smoke"         # 冒烟测试（需后端运行）
docker exec shilian-app pytest tests/test_xxx.py  # 单文件
docker exec shilian-app pytest -m "integration"   # 集成测试
RUN_E2E=1 docker exec -e RUN_E2E=1 shilian-app pytest -m "e2e"
# markers：pytest.ini → unit / integration / slow / e2e / smoke

# 前端
cd ai-interview-frontend && npm install && npm run dev   # → localhost:3000
cd ai-interview-admin && npm install && npm run dev       # → localhost:3001

# 管理端登录：admin@ai-interview.com / ai-interview&admin
```

## 当前状态

### 已完成
- 项目可在 Windows + Docker Desktop 上完整运行
- **7 个容器全部启动**（`shilian-app`, `shilian-postgres`, `shilian-redis`, `shilian-minio`, `shilian-milvus`, `shilian-celery-worker`, `shilian-celery-beat`）
- 数据库迁移完成，管理员和 8 个岗位模板已初始化
- **Milvus 已上线**：`knowledge_chunks`（7 字段）+ `question_bank`（10 字段），HNSW L2 索引就绪
- **embedding 已从 PostgreSQL 迁移到 Milvus**（Alembic `drop_pgvector_embedding_columns`）
- **Service 层全部迁 Milvus**（knowledge_service / question_bank_service / interview_service）
- 端到端验证：创建题目 → Milvus 召回命中 ✅
- 两套 RAG 可用：题库 RAG（已接入面试流程）+ 知识库 RAG（技术链路已实现）
- 岗位匹配 Agent 5 个工具链：`get_parsed_resume → build_candidate_profile → match_positions → get_position_interview_focus → start_mock_interview`
- LangChain 升级完成 (0.3.x → 1.x)：`create_agent()` 重写，`openai` 升级 2.x
- **LangGraph 1.2.8 + langgraph-checkpoint-postgres 3.1.0 已装**（PostgresSaver 持久化 LangGraph state）
- 路由重构为注册中心：`router_registry.py` 集中管理所有 RouteConfig
- Swagger 双套：client/backoffice 独立，OpenAPI JSON 可导出
- 测试体系：pytest 8.3 + 5 个测试文件，unit/smoke/integration/e2e/slow 五类 marker
- 生产 compose 已含 Nginx + Flower（profiles: monitoring）
- 验证码服务 (`redis_verification.py`) + 候补名单 (`waiting_list`)，客户端/后台均挂路由
- 面试会话消息独立建模（`interview_message`），支撑多轮上下文
- 品牌升级："AI Interview" → "试炼 (MockPilot)"，容器名 / PROJECT_NAME / 前端 title 全部更新
- **Phase 1 基础设施完成**：LLM 工厂 / Prompt YAML / Repository 骨架 / workflows/_shared 全部就位（详见下方"Phase 1 实施记录"）
- **Phase 2 实施完成**（2026-07-10）：核心面试 LangGraph 化 + 4 个 P0 bug 修复 + YAML prompt 激活 + JSON 解析兜底

### 重构路线（4 个 Phase）

> 路线图于 2026-07-10 重排，详见 `docs/superpowers/specs/2026-07-10-phase-2-4-roadmap-redesign.md`

- [x] **Phase 0** — Milvus 迁 service + 导出名修正（P0-1/P0-7 已完成；P0-2~P0-6 4 个 bug 遗留至 Phase 2，见下方"P0 遗留"）
- [x] **Phase 1** — 基础设施：prompt 外置 yaml + LLM 工厂 + repository 拆分 + workflows/_shared（已完成）
- [x] **Phase 2** — 核心面试 LangGraph 化（**已完成 2026-07-10**）：HITL StateGraph + 去重 submit_answer + 去正则 score + 基于结构化输出 + SSE 标准化 + ai_service 切 YAML + JSON 解析兜底 + P0-2/3/4/6 已修
- [ ] **Phase 3** — RAG 管线升级：Golden set + RAGAS baseline → BM25 + RRF + qwen3-rerank + 自检循环 LangGraph
- [ ] **Phase 4** — 多 Agent + 可观测性：LangSmith tracing + 3 Agent 拆分（出题/评分/报告）+ Repository 补全 + RAGAS 持续评估

### Phase 1 实施记录

**新增目录**：
- `app/llm/` — LangChain 原子能力层
  - `client.py` — `get_chat_llm()` 工厂 + `chat_completion()` 带重试
  - `embedding.py` — DashScope embedding（从 services/common 迁来）
  - `prompts.py` — YAML PromptTemplate 加载器
- `app/prompts/*.yaml` — 8 个 prompt 外置（position_agent_system / resume_parse / resume_analyze / question_generate / question_select / question_seed / evaluate_answer / generate_report）
- `app/repositories/` — Repository Pattern 骨架
  - `base.py` — `BaseRepository[T]` 泛型基类
  - `interview_repo.py` — 首个 repo（list_by_user / list_messages / get_active_question）
- `app/workflows/_shared/` — LangGraph 编排共享设施
  - `llm.py` / `checkpointer.py`（AsyncPostgresSaver 单例）/ `state_base.py` / `tracing.py` / `format_exception.py` / `tools.py`

**修改**：
- `position_agent_service.get_llm()` 委托 `app.llm.get_chat_llm`
- `ai_service._chat / _chat_stream` 委托 `app.llm.chat_completion`
- `services/common/embedding.py` 改为 deprecation stub
- CLAUDE.md 新增"LangChain / LangGraph 分工（硬性规则）"段

### Phase 2 实施记录

**P0 Bug 修复**：
| Bug | 修改 |
|-----|------|
| P0-2 | `core/security.py` — token 哈希 bcrypt → SHA-256 + hmac.compare_digest；新增 `get_password_hash` / `verify_password` 方法；Alembic 清空旧 token |
| P0-3 | `models/admin.py` — `Admin.role` String(20) → `SAEnum(UserRole)` + Alembic 迁移 |
| P0-4 | `admin.py`/`user.py`/`security.py` — 三份 CryptContext 合并到 `security.py` 全局 `pwd_context` 单例 |
| P0-6 | `schedule/celery_job.py` — 顶部标记 DEPRECATED |

**核心重构**：
- `_extract_json` — 解析失败不抛 ValueError，返回 `{"score": 5.0, "parse_failed": True}` 兜底
- `app/workflows/_shared/sse.py` — `astream_to_sse()` 封装，映射 LangGraph `astream_events` → SSE 标准事件
- `app/workflows/interview/` — 6 个 nodes（fetch_context / retrieve_knowledge / evaluate / check_finished / ask_question / generate_report）+ StateGraph HITL 模式（`interrupt_after=["ask_question"]`）+ `InterviewGraphService.submit_answer(stream=True/False)` 单一入口
- `app/api/client/v1/interview.py` — `/answer` 和 `/answer/stream` 端点全部走 graph service
- `ai_service.py` — 5 个出题方法（parse_resume / analyze_resume / generate_questions / select_and_adapt / generate_with_seeds）切 YAML `load_prompt()`；删除 evaluate_answer / evaluate_answer_stream / generate_report（已迁到 graph nodes）
- `interview_service.py` — 删除 submit_answer / submit_answer_stream（~320 行），保留 start / get_report / get_messages / get_interviews / delete

**代码量**：`ai_service.py` 485→230 行，`interview_service.py` 663→320 行，新增 `workflows/interview/` ~350 行。

**P1 遗留更新**（增量改进，不阻塞）：
- 5 个 repo 只建了 1 个示例（interview_repo）— 其他 4 个（user / admin / question_bank / knowledge）— **Phase 4 做**

### 已知技术债（不阻塞，按 Phase 解决）

- `interview_service`、`ai_service` 全 `@staticmethod`，无 DI，不可 mock — **Phase 3+ 逐步去静态**
- position_agent 的 SYSTEM_PROMPT 仍是 Python 字符串常量，未切 YAML — **Phase 4 统一**
- Graph nodes 通过 `state.custom.db` 传 DB session，非标准 DI — **Phase 3+ 改用 FastAPI Depends 注入**

### Milvus 关键 bug 已修（Phase 0-1）
- ✅ service 层不再调已删除的 `KnowledgeChunk.embedding.cosine_distance` 列
- ✅ 创建/更新/删除都双写 PG + Milvus
- ✅ `pgvector` import 已从 model 移除
- ✅ API 层 `_to_response` 不再访问 `q.embedding`
- ✅ Milvus filter 字符串注入已修（`question_bank._escape_filter_string` 转义，2026-07-10）
- ✅ `vector_db/index.py` ensure_index 日志消息 COSINE → L2 修正（2026-07-10）
