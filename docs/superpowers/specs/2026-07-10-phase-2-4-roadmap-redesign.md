# Phase 2-4 路线图重设计

日期：2026-07-10
状态：已批准

## 背景

Phase 1（基础设施）完成后，对原 Phase 2-4 路线进行全量架构审查，发现三个问题：

1. **Phase 顺序不当**：Phase 3 要修的"正则解析 score"和"SSE 流式"是当前代码的实时质量风险，不应排在 RAG 升级后面
2. **评估缺失**：Phase 2 的 RAG 改进（BM25 + RRF + rerank）在没有 golden set 和 RAGAS 基线的情况下无法验证效果
3. **可观测性延后**：Phase 4 才接 tracing，Phase 2-3 建的 LangGraph workflow 全程黑盒

修订原则：**先修核心流程质量问题 → 用评估驱动 RAG 改进 → 最后加复杂度和可观测性**。

---

## 修订路线图

```
Phase 1 ✅  基础设施（LLM 工厂 + YAML prompt + repo 骨架 + workflows/_shared）

Phase 2     核心面试 LangGraph 化
            去重 submit_answer → 面试 StateGraph → 去正则 score
            → 基于结构化输出 → SSE 标准化 → ai_service 切 YAML
            → JSON 解析兜底

Phase 3     RAG 管线升级
            Golden set → RAGAS baseline → BM25 → RRF
            → qwen3-rerank → 自检循环 LangGraph

Phase 4     多 Agent + 可观测性
            LangFuse tracing → 3 Agent 拆分
            → Repository 补全 → RAGAS 持续评估
```

---

## Phase 2：核心面试 LangGraph 化

### 2.1 去重 submit_answer

合并 `interview_service.py` 中的 `submit_answer` 和 `submit_answer_stream` 为单一入口。

**当前问题**：两个方法（190-347 和 350-511）共享 ~80% 代码，差异仅在 `evaluate_answer` vs `evaluate_answer_stream` 调用和流式版的正则分数提取。

**方案**：提取共享逻辑为私有方法，保留单一入口 `submit_answer(stream=False)`。

```
submit_answer(db, user_id, interview_id, answer, stream=False)
  ├── _get_interview_context()        # 共享
  ├── _save_candidate_answer()        # 共享
  ├── _retrieve_knowledge_context()   # 共享
  └── _evaluate_and_proceed(stream)   # 分流
```

### 2.2 面试流程 StateGraph

新建 `app/workflows/interview/`，将线性面试流程建模为 LangGraph StateGraph。

Graph 结构：
```
START → fetch_context → retrieve_knowledge → evaluate → check_finished
                                                          ├── no → next_question → END
                                                          └── yes → generate_report → END
```

Nodes 文件：
- `nodes/fetch_context.py` — DB 查询面试、简历、对话历史、当前题目
- `nodes/retrieve_knowledge.py` — Milvus 知识库检索 + 题库 reference_answer
- `nodes/evaluate.py` — LLM 评分（YAML prompt + 结构化输出）
- `nodes/next_question.py` — 推进 index，取下一题
- `nodes/generate_report.py` — 汇总评分，生成报告

State 类型继承 `BaseWorkflowState`，使用 AsyncPostgresSaver checkpoint。

### 2.3 去正则解析 score

用 LangChain `with_structured_output(ScoreResult)` 替代 `re.search` 从流式文本提取分数。

两步调用：
1. 流式：LLM 输出纯自然语言评语，SSE 推前端
2. 非流式：同一 prompt + Pydantic 解析 → `ScoreResult(score=8.5, follow_up=False)`

ScoreResult schema：
```python
class ScoreResult(BaseModel):
    score: float
    feedback: str
    follow_up: bool
```

### 2.4 SSE 标准化

不再手写 `yield f"data: {json.dumps(...)}\n\n"`，改用 `sse-starlette` 或 FastAPI `StreamingResponse` 封装。

### 2.5 ai_service 切 YAML

`ai_service.py` 的 6 个方法（parse_resume / analyze_resume / generate_questions / evaluate_answer / select_and_adapt / generate_with_seeds / generate_report）将硬编码 prompt 替换为 `load_prompt()`，验证变量名正确性。

### 2.6 JSON 解析兜底

`_extract_json` 不再抛 ValueError，改为返回带 fallback 的 dict：
```python
{"score": 5.0, "feedback": raw_text[:200], "parse_failed": True}
```

---

## Phase 3：RAG 管线升级

### 3.1 Golden Set 构建

50-100 条标注数据，覆盖 8 个岗位模板的典型 query。标注格式：
```json
{
  "id": "golden_001",
  "position_tag": "python_backend",
  "query": "Python 协程和异步编程的区别",
  "relevant_chunk_ids": ["chunk_42", "chunk_87"],
  "relevance": [3, 2]
}
```

### 3.2 RAGAS Baseline

用 golden set 跑当前纯向量管线：
- context_precision — chunk 排位精度
- context_recall — 相关 chunk 召回率
- faithfulness — LLM 回答对 context 的忠实度
- answer_relevancy — 回答与 query 的相关度

### 3.3 BM25 关键词检索

引入 `rank-bm25` 库，对 Milvus collection 的 content 字段建 BM25 索引。与向量检索并行，各召回 2k 条候选。

### 3.4 RRF 融合

Reciprocal Rank Fusion：`score = 1/(k + rank + 1)`，k=60。向量排名和 BM25 排名取调和平均后排序。

### 3.5 qwen3-rerank

用 DashScope `gte-rerank` 对融合后 top 10-20 做 Cross-encoder 精排。同平台，免额外 API Key。

备选：本地 `BAAI/bge-reranker-v2-m3`（如果 DashScope 延迟高）。

### 3.6 自检循环 LangGraph

```
START → retrieve → check_sufficiency
                      ├── sufficient → END
                      └── insufficient (retry < 3)
                              → rewrite_query → retrieve → check_sufficiency
```

- `check_sufficiency`：top-1 score < MIN_SCORE 或 < min_results → insufficient
- `rewrite_query`：LLM 缩写 query、去限定词、同义词替换
- 最多 2 次 rewrite，最终不足时 graceful degradation

---

## Phase 4：多 Agent + 可观测性

### 4.1 Tracing

接入 LangSmith（LangChain 官方可观测平台），通过环境变量 `LANGSMITH_API_KEY` + `LANGSMITH_PROJECT` 控制。当前 `workflows/_shared/tracing.py` 已实现基础接入，Phase 4 补全节点级 trace 集成。

配置：缺 `LANGSMITH_API_KEY` 时自动跳过，零副作用。

### 4.2 多 Agent 拆分

将当前单体 LLM 拆为 3 个独立 Agent，supervisor 编排：

```
InterviewOrchestrator (supervisor)
  ├── QuestionAgent      — 出题（题库 RAG + 兜底生成）
  ├── EvaluatorAgent     — 单题评分（知识库 + 参考答案作 tool）
  └── ReportAgent        — 最终报告汇总
```

每个 Agent 独立 tools + YAML prompt，通过 supervisor state 传递上下文。

### 4.3 Repository 补全

5 个 repo 全部实现：

| Repo | 核心方法 |
|------|----------|
| user_repo | get_by_id, get_by_email, create, update, list_active |
| admin_repo | get_by_id, get_by_email, list_all |
| question_bank_repo | CRUD + increment_use_count |
| knowledge_repo | CRUD + full_text_search |
| interview_repo | 已有 + get_by_id_with_messages, delete_cascade |

`BaseRepository[T]` 提供泛型 CRUD，各 repo 只加特殊查询。Service 层不再直写 SQL。

### 4.4 RAGAS 持续评估

每次 RAG pipeline 改动后跑 `pytest -m "ragas"`，输出对比表，结果写 LangSmith。
