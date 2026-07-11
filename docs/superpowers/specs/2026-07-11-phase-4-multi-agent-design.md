# Phase 4 — 多 Agent + 可观测性 设计文档

日期：2026-07-11
状态：已批准

## 背景

Phase 0-3 已完成后端核心链路：Milvus 上线 → LangGraph 面试编排 → RAG 混合检索 + 自检循环。Phase 4 的定位是**代码质量 + 可观测性**升级，不改产品行为，只改善内部架构和生产调试能力。

原路线图 4 个子系统全部保留，但根据讨论做了具体化：

1. Agent 类提取（不改图拓扑）
2. Repository 补全（question_bank + knowledge + interview）
3. RAG 链路 Trace
4. RAGAS + LangSmith 持续评估
5. Service 全量去 @staticmethod

## 1. Agent 类提取

### 1.1 总体方案

保持当前 6-node HITL 线性 StateGraph 不变，在每个 node 内部将 LLM 调用委托给独立的 Agent 类。Agent 类位于 `app/agents/`（LangChain 层），被 workflow nodes 调用。

```
app/agents/
├── __init__.py
├── base.py              # BaseAgent — LLM + prompt 公共逻辑
├── question_agent.py    # QuestionAgent — 出题
├── evaluator_agent.py   # EvaluatorAgent — 评分
└── report_agent.py      # ReportAgent — 报告
```

### 1.2 BaseAgent

```python
class BaseAgent:
    def __init__(self, prompt_name: str, temperature: float, tools: list = None):
        self.llm = get_chat_llm(temperature)
        self.prompt = load_prompt(prompt_name)
        self.tools = tools

    async def invoke(self, vars: dict) -> str:
        """prompt | llm → content"""

    async def invoke_structured(self, vars: dict, schema: type) -> BaseModel:
        """prompt | llm.with_structured_output(schema) → obj"""
```

### 1.3 三个 Agent

| Agent | 方法 | Tools | Temp | 结构化输出 | 对应 Node |
|-------|------|-------|------|-----------|-----------|
| QuestionAgent | `.generate(position, difficulty, count, resume) → list[dict]` | question_bank_search | 0.9 | list[QuestionItem] | ask_question |
| EvaluatorAgent | `.evaluate(answer, question, ref_answer, knowledge) → ScoreResult` | kb_search | 0.3 | ScoreResult (已有) | evaluate |
| ReportAgent | `.generate(qa_data, resume, position) → Report` | (无) | 0.5 | extract_json 兜底 | generate_report |

### 1.4 QuestionAgent 职责边界

一个类，内部两条路径：
- `_select_from_bank()` — 题库 RAG 召回 + 选择
- `_fallback_generate()` — LLM 兜底生成新题
- 对外统一接口：`generate()` → list[dict]

调用方（ask_question node / position_agent）不关心题目来源。

### 1.5 Node 调用变化

Before（evaluate_node 直接调 LLM）:
```python
prompt = load_prompt("evaluate_answer")
llm = get_chat_llm(temperature=0.5)
chain = prompt | llm.with_structured_output(ScoreResult)
result = await chain.ainvoke({...})
```

After（node 委托 Agent）:
```python
agent: EvaluatorAgent = state["custom"]["evaluator_agent"]
result = await agent.evaluate(
    answer=state["answer"],
    question=state["current_question"],
    reference=state["reference_answer"],
    knowledge=state["knowledge_context"]
)
```

### 1.6 Prompt YAML

新增 3 个 YAML prompt 文件（每个 Agent 独立 prompt）：
- `app/prompts/question_agent.yaml`
- `app/prompts/evaluator_agent.yaml`
- `app/prompts/report_agent.yaml`

现有 `evaluate_answer.yaml` 和 `generate_report.yaml` 内容迁移到新文件后删除。

### 1.7 注入方式

通过 `app/deps.py` 声明 FastAPI Depends:

```python
def get_question_agent() -> QuestionAgent:
    return QuestionAgent()

def get_evaluator_agent() -> EvaluatorAgent:
    return EvaluatorAgent()

def get_report_agent() -> ReportAgent:
    return ReportAgent()
```

InterviewGraphService 在构建 state.custom 时注入 Agent 实例。

---

## 2. Repository 补全

### 2.1 question_bank_repo.py（新建）

```python
class QuestionBankRepository(BaseRepository[QuestionBank]):
    model = QuestionBank

    async def search_by_position(db, position_tag, difficulty, limit) -> list[QuestionBank]
    async def list_by_position(db, position_tag, page, page_size) -> tuple[list, int]
    async def increment_use_count(db, question_id) -> None
```

QuestionAgent 用 `search_by_position` 按岗位 + 难度查题库。

### 2.2 knowledge_repo.py（新建）

```python
class KnowledgeRepository(BaseRepository[KnowledgeChunk]):
    model = KnowledgeChunk

    async def list_by_document(db, doc_id, page, page_size) -> tuple[list, int]
    async def get_by_chunk_id(db, chunk_id) -> Optional[KnowledgeChunk]
    async def full_text_search(db, query, limit) -> list[KnowledgeChunk]
```

EvaluatorAgent 用 `full_text_search` 做 PG 全文搜索辅助。

### 2.3 interview_repo.py（补全 2 方法）

```python
async def get_by_id_with_messages(db, interview_id) -> Optional[Interview]
    # eager load interview + messages + resume

async def delete_cascade(db, interview_id) -> bool
    # 删 interview + 关联 messages，事务内完成
```

### 2.4 Scope 说明

- user_repo 和 admin_repo **不在 Phase 4 范围**
- 新建 2 个 repo + 修改 1 个 repo
- question_bank_service / knowledge_service / interview_service 中直接写 SQL 的地方切 repo

---

## 3. RAG 链路 Tracing

### 3.1 目标

在 LangSmith 中可以看到检索每个步骤的耗时：

```
retrieve_knowledge (480ms)
  ├── vector_recall   210ms (Milvus HNSW)
  ├── bm25_search      85ms (rank-bm25)
  ├── rrf_fusion        2ms (RRF merge)
  └── rerank          180ms (DashScope gte-rerank)
```

### 3.2 实现

在 `app/workflows/_shared/tracing.py` 新增 `trace_span` context manager：

```python
@contextmanager
def trace_span(name: str, metadata: dict = None):
    if not is_tracing_enabled():
        yield
        return
    span = RunTree(name=name, run_type="retriever", ...)
    try:
        yield span
        span.end(outputs={...})
    except Exception as e:
        span.end(error=str(e))
```

在 `app/retrieval/pipeline.py` 的 `retrieve()` 方法里给 4 个步骤各包一层 `trace_span`。

Tracing 关闭时（缺 LANGSMITH_API_KEY）零开销。

### 3.3 变更文件

- `app/workflows/_shared/tracing.py` — 新增 trace_span
- `app/retrieval/pipeline.py` — 4 步各包 span

---

## 4. RAGAS + LangSmith 持续评估

### 4.1 数据流

```
eval/golden_set.json → upload_golden_set.py → LangSmith Dataset "shilian-golden-set"
    ↓
eval_ragas.py --upload → LangSmith Experiment "ragas-{date}"
    ↓ (多次运行)
LangSmith UI: Experiment A vs Experiment B diff 对比
```

### 4.2 新文件/修改

- `eval/scripts/upload_golden_set.py`（新建）— golden_set.json 同步到 LangSmith dataset
- `eval/scripts/eval_ragas.py`（修改）— 新增 `--upload` flag 和 `--experiment-name` 参数

### 4.3 使用方式

```bash
docker exec shilian-app python eval/scripts/upload_golden_set.py
docker exec shilian-app python eval/scripts/eval_ragas.py --golden-set eval/golden_set.json --upload
```

不改 CI 配置，仍然手动触发。LangSmith 提供历史 experiment 统一对比视图。

---

## 5. Service 全量去 @staticmethod

### 5.1 范围

全部 14 个 service 文件：

- client: ai_service.py, auth.py, interview_service.py, position_agent_service.py, redis_verification.py, resume_service.py, waiting_list.py, email_templates.py
- backoffice: admin.py, auth.py, knowledge_service.py, position_template_service.py, question_bank_service.py, waiting_list.py

### 5.2 改造规则

1. 去掉 `@staticmethod` + 加 `self` → 方法签名变，方法体不变
2. 不拆类、不合类，保持现有文件边界
3. 调用方从 `ClassName.method(db)` 改为 `Depends → instance.method()`
4. 不影响 Phase 4 其他工作，可并行

### 5.3 依赖注入

`app/deps.py` 集中声明：

```python
def get_ai_service() -> AiService:
    return AiService()

def get_interview_service() -> InterviewService:
    return InterviewService()

# ... 其他 service
```

API 层通过 `Depends(get_ai_service)` 注入。

---

## 变更量估算

| 子系统 | 新建文件 | 修改文件 | 删除行/新增行 |
|--------|---------|---------|-------------|
| Agent 类 | 5 (agents/* + deps.py) | 3 nodes | ~200 / ~350 |
| Repository | 2 | 1 repo + 3 services | ~100 / ~250 |
| Tracing | 0 | 2 | ~10 / ~40 |
| RAGAS eval | 1 | 1 | ~20 / ~80 |
| 去 static | 0 | 14+ 所有调用方 | ~100 / ~120 |
| **合计** | **8** | **~24** | **~430 / ~840** |

---

## 不做什么

- 不接入 LangFuse（保持 LangSmith）
- 不引入 Supervisor 编排
- 不建 user_repo / admin_repo
- 不拆 QuestionAgent 为两个类
- 不改 graph 拓扑结构
- 不新增 Phase 4 之外的 CI 配置
