import pytest
from common.evaluation.ai_judge import AIJudge

class DummyLLM:
    def __init__(self, fixed_response):
        self.fixed_response = fixed_response

    def invoke(self, prompt):
        class Response:
            def __init__(self, content):
                self.content = content
            @property
            def content(self):
                return self._content
            @content.setter
            def content(self, value):
                self._content = value
        return Response(self.fixed_response)

def test_evaluate_with_reference():
    judge = AIJudge(llm=DummyLLM("0.8"))
    scores = judge.evaluate("Paris is in Germany.", "Paris is in France.")
    assert "accuracy" in scores
    assert 0.0 <= scores["accuracy"] <= 1.0

def test_evaluate_without_reference():
    judge = AIJudge(llm=DummyLLM("0.9"))
    scores = judge.evaluate("Paris is in Germany.")
    assert "clarity" in scores
    assert 0.0 <= scores["clarity"] <= 1.0

def test_add_metric():
    judge = AIJudge(llm=DummyLLM("0.7"))
    judge.add_metric("conciseness", "Rate how concise the response is", False)
    assert "conciseness" in judge.available_metrics()
    scores = judge.evaluate("Short and clear.")
    assert "conciseness" in scores
    assert 0.0 <= scores["conciseness"] <= 1.0

def test_weighted_score():
    judge = AIJudge()
    scores = {"accuracy": 0.6, "clarity": 0.8}
    weights = {"accuracy": 2.0, "clarity": 1.0}
    weighted = judge._weighted(scores, weights)
    assert abs(weighted - (0.6*2 + 0.8*1)/3) < 1e-6

def test_parse_weights():
    judge = AIJudge()
    parsed = judge._parse_weights(["accuracy=0.7", "clarity=0.3"])
    assert parsed == {"accuracy": 0.7, "clarity": 0.3}
