# memory_evaluation.py

from typing import List, Tuple, Dict, Any
import statistics

def precision_recall_f1(predicted: List[Any], expected: List[Any]) -> Tuple[float, float, float]:
    """
    Compute precision, recall and F1 between two lists of items.
    Items are compared for equality. Order does not matter.
    """
    pred_set = set(predicted)
    exp_set = set(expected)
    if not pred_set and not exp_set:
        return 1.0, 1.0, 1.0
    if not pred_set:
        return 0.0, 0.0, 0.0

    tp = len(pred_set & exp_set)
    precision = tp / len(pred_set)
    recall = tp / len(exp_set) if exp_set else 1.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1

def evaluate_memory_updates(
    predicted_updates: List[Any],
    expected_updates: List[Any]
) -> Dict[str, float]:
    """
    Evaluate how well the agent's memory *updates* match the expected updates.
    Returns precision, recall and F1.
    """
    p, r, f1 = precision_recall_f1(predicted_updates, expected_updates)
    return {"memory_precision": p, "memory_recall": r, "memory_f1": f1}

def evaluate_memory_retrieval(
    retrieve_fn: Any,
    queries: List[str],
    expected_results: List[List[Any]],
    top_k: int = 1
) -> Dict[str, float]:
    """
    Given a retrieval function `retrieve_fn(query, k)` that returns a list of
    k memory items, evaluate over multiple queries.
    Returns:
      - `retrieval_accuracy@k`: fraction of queries for which at least one
        expected item appears in the top‐k.
    """
    hits = 0
    for query, expect in zip(queries, expected_results):
        results = retrieve_fn(query, top_k)
        # did we retrieve any expected item?
        if set(results) & set(expect):
            hits += 1
    accuracy = hits / len(queries) if queries else 1.0
    return {f"retrieval_accuracy@{top_k}": accuracy}

def aggregate_metrics(list_of_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Given a list of metric‐dicts (e.g. outputs from evaluate_*), compute
    the mean for each metric.
    """
    if not list_of_dicts:
        return {}
    aggregated: Dict[str, float] = {}
    keys = list_of_dicts[0].keys()
    for k in keys:
        vals = [d[k] for d in list_of_dicts if k in d]
        aggregated[k] = statistics.mean(vals) if vals else 0.0
    return aggregated
