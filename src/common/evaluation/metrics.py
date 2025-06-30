import json, pathlib, importlib, argparse, statistics as stats
from typing import Dict, List
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util

st_model = SentenceTransformer("all-MiniLM-L6-v2")

def phrase_recall(pred_reply: str, phrases: List[str]) -> float:
    if not phrases:
        return 1.0
    found = sum(1 for p in phrases if p.lower() in pred_reply.lower())
    return found / len(phrases)

def tool_metrics(pred_tools: List[str], expected_calls: List[dict]) -> Dict[str, float]:
    expected_names = [c.get("tool") for c in expected_calls]
    if not expected_names:
        return {"tool_recall": 1.0, "tool_precision": 1.0}
    pred_set = set(pred_tools)
    exp_set = set(expected_names)
    tp = len(exp_set & pred_set)
    recall = tp / len(exp_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    return {"tool_recall": recall, "tool_precision": precision}

def param_accuracy(pred_calls: List[dict], expected_calls: List[dict]) -> float:
    if not expected_calls:
        return 1.0
    matched = 0
    for exp in expected_calls:
        for pred in pred_calls:
            if pred.get("tool") == exp.get("tool") and pred.get("params") == exp.get("params"):
                matched += 1
                break
    return matched / len(expected_calls)

def task_success(pred_reply: str, pred_tools: List[str], expected: dict) -> float:
    pr = phrase_recall(pred_reply, expected.get("customer_msg_contains", []))
    tr = tool_metrics(pred_tools, expected.get("tool_calls", [])).get("tool_recall", 0.0)
    return (pr + tr) / 2.0

def parse_weights(w_pairs: List[str]) -> Dict[str, float]:
    out = {}
    for pair in w_pairs:
        k, v = pair.split("=", 1)
        out[k.lower()] = float(v)
    return out

def accuracy(match, exp): 
    return float(match.strip() == exp.strip())

def bert(pred, ref):
    P, R, F = bert_score([pred], [ref], lang="en", rescale_with_baseline=True)
    return F.mean().item()          # value between 0-1

def cosine(pred, ref):
    # returns a value between –1 … 1
    # Can also be used with any embedding model
    emb = st_model.encode([pred, ref], convert_to_tensor=True, normalize_embeddings=True)
    return util.cos_sim(emb[0], emb[1]).item()

METRICS = {"accuracy": accuracy, "bertscore": bert, "cosine": cosine}
