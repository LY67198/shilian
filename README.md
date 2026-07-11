# 试炼 (MockPilot)

基于 FastAPI + LangGraph + Milvus 的企业级 AI 模拟面试平台，集成简历解析、RAG 题库、多轮面试编排、AI 评分与结构化报告生成。

站在候选人立场，反复训练锻造面试能力 — 不是单次对话，是可循环的练习场。

## 功能特性

简历解析 — PDF / DOCX / TXT 上传，自动结构化抽取教育、技能、项目经历
岗位匹配 — LangChain ReAct Agent，5 工具链串联：简历读取 → 画像构建 → 岗位匹配 → 面试方向 → 启动面试
题库 RAG — Milvus 向量召回 + 倒排索引 + RRF 融合 + Cross-Encoder 重排 + 自检循环
多轮面试 — LangGraph StateGraph 编排，断点续传，PostgresSaver 状态持久化
AI 评分 — 参考答案 + 知识库片段注入评分 Prompt，减少幻觉
面试报告 — 综合评估 + 分项得分 + 优势/不足 + 改进建议
岗位模板 — 8 种预设岗位（Python / Go / Java / Vue 等），开箱即用
题库管理 — 增删改查、批量导入、按难度 / 岗位 / 分类筛选（管理员）
知识库管理 — PDF / MD 文档上传、向量化、检索测试（管理员）
账号体系 — 用户端邮箱注册 + 验证码；后台管理员 JWT 鉴权
登录限流 — Redis 滑动窗口防暴力破解（5 分钟最多 5 次）
RAG 质量评估 — RAGAS 离线指标（faithfulness / answer_relevancy / context_precision / context_recall）+ golden set 回归
全链路追踪 — LangSmith 节点级 trace（可选接入）

## 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| 后端框架 | FastAPI 0.115 | RESTful API + SSE 流式 |
| 数据校验 | Pydantic 2.11 | 请求 / 响应模型 + Settings |
| 数据库 | PostgreSQL 16 | 主业务数据 + LangGraph Checkpoint |
| 向量数据库 | Milvus 2.4.10 | RAG 文档 / 题库嵌入与检索 |
| 向量索引 | HNSW + L2 | 1024 维向量近似最近邻（DashScope 已归一化，L2 等价 COSINE） |
| 对象存储 | MinIO | Milvus standalone 内部存储 |
| 缓存 / 锁 | Redis 7 | 登录限流 + Celery Broker |
| 异步任务 | Celery 5.5 | 邮件 / 验证码发送 |
| Agent | LangChain 1.3 | 工具调用 + create_agent |
| 编排 | LangGraph 1.2.8 | 多轮面试 StateGraph |
| 检索 | 向量召回 + RRF + qwen3-rerank | 题库 + 知识库四路召回精排 |
| 状态持久化 | langgraph-checkpoint-postgres 3.1.0 | AsyncPostgresSaver 存图状态 |
| 嵌入 | DashScope text-embedding-v3 | 1024 维向量 |
| 重排 | DashScope qwen3-rerank | 召回结果精排 |
| LLM | DeepSeek | 推理（OpenAI 兼容 SDK 2.x） |
| 认证 | python-jose + passlib (bcrypt) | JWT 签发 / 校验 / 密码哈希 |
| 评估 | RAGAS | RAG 质量量化（faithfulness / answer_relevancy / context_precision / context_recall） |
| 可观测 | LangSmith | LangGraph 节点级 trace（可选） |
| 前端 | Vue 3 + Vite 5 + Pinia | 用户端 + 管理端 |
| 构建 | Vite | 前端工程化 |
| 部署 | Docker Compose | 7 容器一键启动 |

## 快速开始

### 前置要求

- Python 3.12+
- Node.js 18+
- Docker Desktop
- DeepSeek API Key
- DashScope API Key

### 1. 启动基础设施

```bash
cd ai-interview-backend
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

启动 7 个容器：`shilian-app` / `shilian-postgres` / `shilian-redis` / `shilian-minio` / `shilian-milvus` / `shilian-celery-worker` / `shilian-celery-beat`。

### 2. 后端初始化

```bash
cp .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY 和 DASHSCOPE_API_KEY

docker exec shilian-app alembic upgrade head
docker exec shilian-app python scripts/create_first_admin.py
docker exec shilian-app python scripts/seed_position_templates.py
docker exec shilian-app python scripts/init_milvus.py
```

### 3. 前端启动

```bash
# 用户端
cd ai-interview-frontend
npm install
npm run dev
# → http://localhost:3000

# 管理端
cd ai-interview-admin
npm install
npm run dev
# → http://localhost:3001
```

### 4. 验证

```bash
# 后端健康
curl http://localhost:8006/api/v1/config/health

# Swagger 文档
# 用户端：http://localhost:8006/client/docs
# 管理端：http://localhost:8006/backoffice/docs
```

### 5. RAG 质量评估（可选）

```bash
# 一次性跑 baseline
docker exec shilian-app python scripts/eval_ragas.py \
  --golden-set eval/golden_set.json \
  --output eval/reports/baseline-$(date +%Y%m%d).json

# 报告包含 4 个指标：faithfulness / answer_relevancy /
# context_precision / context_recall
```

## 项目结构

```
ai-interview-agent/
├── ai-interview-backend/             # FastAPI 后端（容器化）
│   ├── app/
│   │   ├── api/                      # 接口层（client / backoffice）
│   │   │   ├── client/v1/            # 候选用户端
│   │   │   └── backoffice/v1/        # 后台管理端
│   │   ├── workflows/                # LangGraph 工作流（主栈）
│   │   │   ├── interview/            # 多轮面试 StateGraph（规划中）
│   │   │   ├── question_gen/         # 题目召回自检循环（规划中）
│   │   │   └── _shared/              # 共享基础设施
│   │   ├── vector_db/                # Milvus 客户端 + collections
│   │   ├── retrieval/                # RAG 检索引擎（规划中）
│   │   ├── llm/                      # LangChain 域（LLM / embedding）
│   │   ├── repositories/             # 数据访问层
│   │   ├── prompts/                  # PromptTemplate 集中管理
│   │   ├── services/                 # 业务服务层
│   │   ├── models/                   # SQLAlchemy 实体
│   │   ├── schemas/                  # Pydantic 模型
│   │   ├── core/                     # 配置 / 安全 / Celery
│   │   ├── db/                       # async session
│   │   ├── route/                    # 路由注册中心
│   │   ├── configs/                  # Swagger 拆分
│   │   ├── schedule/                 # Celery 定时任务
│   │   ├── common/                   # i18n / log
│   │   ├── exceptions/               # 业务异常
│   │   └── utils/                    # 工具函数
│   ├── tests/                        # pytest
│   ├── migrations/                   # Alembic
│   ├── scripts/                      # 初始化脚本
│   ├── eval/                         # RAGAS 离线评估（golden set + 报告）
│   └── docker-compose*.yml
├── ai-interview-frontend/            # 用户端 (Vue 3 + Vite)
├── ai-interview-admin/               # 管理端 (Vue 3 + Element Plus)
└── docs/                             # 内部规划文档（不公开）
```

## API 概览

| 方法 | 路径 | 说明 |
|------|------|------|
| **用户端 — 认证** |  |  |
| POST | /api/v1/client/auth/register | 用户注册 |
| POST | /api/v1/client/auth/login | 用户登录 |
| POST | /api/v1/client/auth/refresh | 刷新令牌 |
| GET  | /api/v1/client/auth/me | 当前用户信息 |
| POST | /api/v1/client/auth/send-verification-code | 发送验证码 |
| **用户端 — 简历** |  |  |
| POST | /api/v1/client/resume/upload | 上传简历 |
| GET  | /api/v1/client/resume | 简历列表 |
| GET  | /api/v1/client/resume/{id} | 简历详情 |
| DELETE | /api/v1/client/resume/{id} | 删除简历 |
| **用户端 — 面试** |  |  |
| POST | /api/v1/client/interview/start | 开始面试 |
| POST | /api/v1/client/interview/{id}/answer | 提交答案 |
| POST | /api/v1/client/interview/{id}/answer/stream | 流式评分（SSE） |
| GET  | /api/v1/client/interview/{id}/report | 面试报告 |
| GET  | /api/v1/client/interview/{id}/messages | 对话历史 |
| GET  | /api/v1/client/interview | 面试列表 |
| **用户端 — Agent** |  |  |
| POST | /api/v1/client/position-agent/match | 岗位匹配 |
| POST | /api/v1/client/position-agent/start-interview | 一键启动面试 |
| **后台 — 题库** |  |  |
| GET / POST | /api/v1/backoffice/question-bank | 题库 CRUD |
| POST | /api/v1/backoffice/question-bank/test-retrieve | 检索测试 |
| POST | /api/v1/backoffice/question-bank/batch-import | 批量导入 |
| **后台 — 知识库** |  |  |
| GET / POST | /api/v1/backoffice/knowledge | 知识库 CRUD |
| POST | /api/v1/backoffice/knowledge/{id}/upload | 上传文档 |
| **后台 — 模板 / 用户** |  |  |
| GET / POST | /api/v1/backoffice/position-template | 岗位模板 CRUD |
| GET / POST | /api/v1/backoffice/admins | 管理员 CRUD |
| GET / POST | /api/v1/backoffice/users | 读者管理 |
| GET | /api/v1/backoffice/interviews | 面试记录 |

## 亮点设计

### 混合检索与重排序

向量召回（Milvus HNSW）+ 倒排索引（BM25）+ RRF 倒数秩融合 + Cross-Encoder 重排序（qwen3-rerank）。所有召回结果按相似度统一排序，知识库与题库使用同一管线，支持 self-check 循环自动改写 query 提升召回质量。

### LangGraph 多轮面试编排

采用 StateGraph 编排多轮对话：`generate_question → wait_for_answer → evaluate_answer → should_continue → next | report`，使用 PostgresSaver 按 `interview_id` 持久化状态。断点续传 / 失败重试 / 中断恢复零成本，状态字段最小化（只存跨节点需要的）。

### 端到端可观测

LangSmith 全链路 trace、SSE 实时推送节点切换事件、structured output 替代正则解析 score。LLM 调用、向量召回、节点耗时全程可追溯，调试 production 召回质量时不用猜。

### RAG 质量评估体系

基于 [RAGAS](https://github.com/explodinggradients/ragas) 框架的离线评估流水线，4 个核心指标：

- **faithfulness** — 答案相对召回内容的忠实度（防幻觉）
- **answer_relevancy** — 答案与 query 的相关度
- **context_precision** — 召回结果中真正相关的占比
- **context_recall** — 理想答案被召回到的比例

配套 `eval/golden_set.json` 手工标注集（20-30 条覆盖各 difficulty × position_tag），每次改 RAG 相关代码（retrieval / vector_db / prompt）跑一次评估，diff > 5% 阻断 merge。所有 baseline 数字存 `eval/reports/baseline-{date}.json`，后续 Phase 2 加 hybrid + RRF + rerank 后可对比看提升。

### 工业级工程化

7 个服务（App + Postgres + Redis + MinIO + Milvus + Celery）Docker Compose 一键启动；embedding 与业务数据分离存储（PG 存元数据，Milvus 存向量）；Alembic 数据库迁移；Pytest 分层标记（unit / smoke / integration / e2e）；CI 友好。

## 许可

本项目以 [LICENSE](./LICENSE) 协议开源。
