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
    """evaluate_node — 委托 EvaluatorAgent，需 mock agent"""

    @staticmethod
    def _mock_agent(score=7.5, feedback="回答良好"):
        """创建 mock EvaluatorAgent，返回指定 ScoreResult"""
        from app.workflows.interview.state import ScoreResult

        class MockAgent:
            async def evaluate(self, **kwargs):
                return ScoreResult(score=score, feedback=feedback)
        return MockAgent()

    async def test_returns_score_and_feedback(self):
        """验证 evaluate_node 通过 config.configurable.evaluator_agent 获取 agent 并返回 score + feedback"""
        from app.workflows.interview.nodes.evaluate import evaluate_node

        config = {
            "configurable": {
                "evaluator_agent": self._mock_agent(score=7.5, feedback="回答良好"),
            },
        }
        state = {
            "current_question": "请介绍 Python 的 GIL",
            "answer": "GIL 是全局解释器锁...",
            "resume_context": {},
            "chat_history": [],
            "knowledge_context": [],
            "interview_id": 1,
            "user_id": 42,
        }
        result = await evaluate_node(state, config)
        assert result["score"] == 7.5
        assert result["feedback"] == "回答良好"

    async def test_fallback_when_agent_not_in_config(self):
        """agent 不在 config 时使用默认 EvaluatorAgent（无外部依赖则不测 LLM 调用）"""
        from app.workflows.interview.nodes.evaluate import evaluate_node

        state = {
            "current_question": "Q",
            "answer": "A",
            "resume_context": {},
            "chat_history": [],
            "knowledge_context": [],
            "interview_id": 1,
            "user_id": 42,
            # 未注入 evaluator_agent — 节点应创建默认实例
        }
        # 默认 EvaluatorAgent 会尝试调用 LLM，此处只验证 agent 可成功创建并传入参数
        # 因此我们 mock agent.evaluate 避免真实 LLM 调用
        from unittest.mock import AsyncMock, patch
        from app.workflows.interview.state import ScoreResult

        with patch.object(
            __import__("app.agents.evaluator_agent", fromlist=["EvaluatorAgent"]).EvaluatorAgent,
            "evaluate",
            new_callable=AsyncMock,
            return_value=ScoreResult(score=5.0, feedback="评分异常，已记录"),
        ):
            result = await evaluate_node(state, {"configurable": {}})
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
