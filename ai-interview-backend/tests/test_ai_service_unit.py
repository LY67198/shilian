"""
单元测试 — 测试 AIService._extract_json，无任何外部依赖

运行：
  pytest tests/test_ai_service_unit.py -m "unit"
"""
import pytest
from app.services.client.ai_service import AIService


class TestExtractJson:
    """验证从 DeepSeek 返回的自由文本中提取 JSON 的逻辑"""

    def test_pure_json_object(self):
        text = '{"score": 7.5, "feedback": "ok"}'
        result = AIService._extract_json(text)
        assert result == {"score": 7.5, "feedback": "ok"}

    def test_pure_json_array(self):
        text = '[{"index": 0, "question": "Q1"}]'
        result = AIService._extract_json(text)
        assert result == [{"index": 0, "question": "Q1"}]

    def test_markdown_fenced_json(self):
        text = '好的，这是评分：\n```json\n{"score": 8.0}\n```\n以上是评分。'
        result = AIService._extract_json(text)
        assert result == {"score": 8.0}

    def test_json_with_leading_explanation(self):
        text = '评价如下：这是一个不错的回答。{"score": 6.5, "feedback": "good"}'
        result = AIService._extract_json(text)
        assert result["score"] == 6.5
        assert result["feedback"] == "good"

    def test_json_with_trailing_explanation(self):
        text = '{"score": 9.0, "follow_up": false} —— 评分完成'
        result = AIService._extract_json(text)
        assert result == {"score": 9.0, "follow_up": False}

    def test_empty_text_fallback(self):
        result = AIService._extract_json("")
        assert result["score"] == 5.0
        assert result["parse_failed"] is True

    def test_whitespace_text_fallback(self):
        result = AIService._extract_json("   \n  \t  ")
        assert result["score"] == 5.0
        assert result["parse_failed"] is True

    def test_no_json_in_text_fallback(self):
        result = AIService._extract_json("这是普通文字，没有任何 JSON")
        assert result["score"] == 5.0
        assert result["parse_failed"] is True

    def test_pass_through_dict(self):
        """已是 dict/list 的直接透传"""
        d = {"score": 10}
        assert AIService._extract_json(d) is d

    def test_nested_object(self):
        text = '```json\n{"score": 7.5, "extras": {"follow_up": true, "tags": ["a", "b"]}}\n```'
        result = AIService._extract_json(text)
        assert result["score"] == 7.5
        assert result["extras"]["follow_up"] is True
        assert result["extras"]["tags"] == ["a", "b"]