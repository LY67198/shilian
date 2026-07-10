"""面试 workflow 节点单元测试"""
import pytest

from app.workflows.interview.nodes.check_finished import check_finished_node


@pytest.mark.unit
class TestCheckFinished:
    """check_finished_node — 纯函数节点，无外部依赖"""

    async def test_not_finished(self):
        state = {"current_index": 2, "total_questions": 5}
        result = await check_finished_node(state)
        assert result == {"is_finished": False}

    async def test_finished_last_question(self):
        state = {"current_index": 4, "total_questions": 5}
        result = await check_finished_node(state)
        assert result == {"is_finished": True}

    async def test_finished_beyond_last(self):
        state = {"current_index": 5, "total_questions": 5}
        result = await check_finished_node(state)
        assert result == {"is_finished": True}

    async def test_empty_state_defaults(self):
        state = {}
        result = await check_finished_node(state)
        assert result == {"is_finished": True}  # 0 + 1 >= 0 → True


@pytest.mark.unit
class TestScoreResult:
    """ScoreResult Pydantic 模型验证"""

    def test_valid_score(self):
        from app.workflows.interview.state import ScoreResult
        r = ScoreResult(score=8.5, feedback="不错", follow_up=False)
        assert r.score == 8.5
        assert r.feedback == "不错"
        assert r.follow_up is False

    def test_default_values(self):
        from app.workflows.interview.state import ScoreResult
        r = ScoreResult()
        assert r.score == 5.0
        assert r.feedback == ""
        assert r.follow_up is False

    def test_score_out_of_range_clamped(self):
        from app.workflows.interview.state import ScoreResult
        with pytest.raises(ValueError):
            ScoreResult(score=11.0)
        with pytest.raises(ValueError):
            ScoreResult(score=-1.0)


@pytest.mark.unit
class TestInterviewState:
    """InterviewState TypedDict 字段设置"""

    def test_minimal_state(self):
        from app.workflows.interview.state import InterviewState
        state: InterviewState = {
            "interview_id": 1,
            "user_id": 42,
        }
        assert state["interview_id"] == 1
        assert state["user_id"] == 42
        # 可选字段
        assert state.get("resume_context") is None
        assert state.get("is_finished") is None


@pytest.mark.unit
class TestEvaluateNode:
    """evaluate_node — 需 mock LLM"""

    async def test_returns_score_and_feedback(self, monkeypatch):
        """验证 evaluate_node 调用 with_structured_output 后返回 score + feedback"""
        from app.workflows.interview.nodes.evaluate import evaluate_node

        class MockScoreResult:
            score = 7.5
            feedback = "回答良好"

        class MockChain:
            async def ainvoke(self, *args, **kwargs):
                return MockScoreResult()

        mock_llm = type("MockLLM", (), {
            "with_structured_output": lambda self, model: MockChain(),
        })()

        monkeypatch.setattr(
            "app.workflows.interview.nodes.evaluate.get_chat_llm",
            lambda **kw: mock_llm,
        )
        class MockPrompt:
            def __or__(self, other):
                return other

        monkeypatch.setattr(
            "app.workflows.interview.nodes.evaluate.load_prompt",
            lambda name: MockPrompt(),
        )

        state = {
            "current_question": "请介绍 Python 的 GIL",
            "answer": "GIL 是全局解释器锁...",
            "resume_context": {},
            "chat_history": [],
            "knowledge_context": [],
            "interview_id": 1,
            "user_id": 42,
        }
        result = await evaluate_node(state)
        assert result["score"] == 7.5
        assert result["feedback"] == "回答良好"

    async def test_fallback_on_llm_error(self, monkeypatch):
        """LLM 异常时返回兜底分数"""
        from app.workflows.interview.nodes.evaluate import evaluate_node

        monkeypatch.setattr(
            "app.workflows.interview.nodes.evaluate.get_chat_llm",
            lambda **kw: (_ for _ in ()).throw(RuntimeError("LLM 超时")),
        )

        state = {
            "current_question": "Q",
            "answer": "A",
            "resume_context": {},
            "chat_history": [],
            "knowledge_context": [],
            "interview_id": 1,
            "user_id": 42,
        }
        result = await evaluate_node(state)
        assert result["score"] == 5.0
        assert "评分异常" in result["feedback"]


@pytest.mark.unit
class TestExtractJson:
    """extract_json utility（app.common.json_utils）"""

    def test_fallback_on_empty(self):
        from app.common.json_utils import extract_json
        result = extract_json("")
        assert result["score"] == 5.0
        assert result["parse_failed"] is True

    def test_fallback_on_plain_text(self):
        from app.common.json_utils import extract_json
        result = extract_json("这是普通文字")
        assert result["score"] == 5.0
        assert result["parse_failed"] is True

    def test_extract_nested_json(self):
        from app.common.json_utils import extract_json
        text = '```json\n{"score": 8.5, "feedback": "优秀"}\n```'
        result = extract_json(text)
        assert result["score"] == 8.5
        assert result["feedback"] == "优秀"
