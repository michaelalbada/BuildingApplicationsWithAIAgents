import pytest
from common.evaluation import memory_evaluation

def test_precision_recall_f1_perfect_match():
    pred = ["apple", "banana"]
    expected = ["banana", "apple"]
    p, r, f1 = memory_evaluation.precision_recall_f1(pred, expected)
    assert p == r == f1 == 1.0

def test_precision_recall_f1_no_overlap():
    pred = ["apple"]
    expected = ["banana"]
    p, r, f1 = memory_evaluation.precision_recall_f1(pred, expected)
    assert p == r == f1 == 0.0

def test_evaluate_memory_updates_partial_match():
    pred = ["a", "b", "c"]
    expected = ["b", "c", "d"]
    result = memory_evaluation.evaluate_memory_updates(pred, expected)
    assert result["memory_precision"] == 2/3
    assert result["memory_recall"] == 2/3
    assert round(result["memory_f1"], 2) == 0.67

def test_evaluate_memory_retrieval_basic():
    def dummy_retriever(query, k):
        return ["memory1", "memory2"] if "x" in query else ["memory3"]

    queries = ["x1", "x2", "y"]
    expected = [["memory1"], ["memory2"], ["none"]]
    result = memory_evaluation.evaluate_memory_retrieval(dummy_retriever, queries, expected, top_k=2)
    assert abs((result["retrieval_accuracy@2"]) - 2./3) < 1e-5

def test_aggregate_metrics():
    inputs = [
        {"memory_precision": 0.5, "memory_recall": 0.8},
        {"memory_precision": 0.7, "memory_recall": 0.6}
    ]
    agg = memory_evaluation.aggregate_metrics(inputs)
    assert round(agg["memory_precision"], 2) == 0.6
    assert round(agg["memory_recall"], 2) == 0.7
