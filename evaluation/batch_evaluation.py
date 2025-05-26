"""
batch_evaluation.py – outcome + diagnostic metrics
=================================================
Handles multi-turn conversations and surfaces diagnostics, including parameter accuracy.
"""
from __future__ import annotations
import builtins
from langchain.schema import BaseMessage
import operator
from typing import Dict, List, Annotated, Sequence
builtins.Annotated = Annotated
builtins.BaseMessage = BaseMessage
builtins.operator = operator
builtins.Sequence = Sequence
builtins.List = List
import argparse
import importlib.util
import json
import pathlib
import statistics as stats

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

def to_lc_message(turn: dict):
    role = (turn.get("role") or "").lower()
    txt = turn.get("content", "")
    if role in {"customer", "user", "human"}:
        return HumanMessage(content=txt)
    if role in {"assistant", "agent", "ai"}:
        return AIMessage(content=txt)
    if role == "system":
        return SystemMessage(content=txt)
    if role == "tool":
        return ToolMessage(content=txt, tool_call_id=turn.get("tool_call_id", "unknown"))
    return HumanMessage(content=txt)

def load_graph(path: str):
    spec = importlib.util.spec_from_file_location("user_graph", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(mod)                 # type: ignore
    if hasattr(mod, "graph"):
        return mod.graph
    if hasattr(mod, "construct_graph"):
        return mod.construct_graph()  # type: ignore
    raise AttributeError(f"{path} exposes neither `graph` nor `construct_graph()`")

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
    phrase_ok = phrase_recall(pred_reply, expected.get("customer_msg_contains", [])) == 1.0
    tool_ok = tool_metrics(pred_tools, expected.get("tool_calls", [])).get("tool_recall", 0.0) == 1.0
    return float(phrase_ok and tool_ok)

def parse_weights(pairs: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for pair in pairs:
        if "=" in pair:
            k, v = pair.split("=", 1)
            try:
                out[k.strip().lower()] = float(v)
            except ValueError:
                pass
    return out

def run_evaluation(graph, args, weights: Dict[str, float], metrics: Dict[str, List[float]], verbose: bool = False):
    for raw in pathlib.Path(args.dataset).read_text().splitlines():
        if not raw.strip():
            continue
        ex = json.loads(raw)
        order = ex["order"]
        messages = [to_lc_message(t) for t in ex["conversation"]]
        exp_final = ex["expected"]["final_state"]

        result = graph.invoke({"order": order, "messages": messages})
        # find last assistant message without tool calls
        final_reply = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                calls = msg.additional_kwargs.get("tool_calls", [])
                if not calls:
                    final_reply = msg.content or ""
                    break
        if verbose:
            print(f"\n=== Example: {ex.get('id', 'unknown')} ===")
            print("Conversation:")
            for m in result["messages"]:
                if isinstance(m, HumanMessage):
                    print(f"Customer: {m.content}")
                elif isinstance(m, AIMessage):
                    print(f"Agent: {m.content}")
                elif isinstance(m, ToolMessage):
                    print(f"Tool call: {m.tool_call_id} ({m.tool_name})")
            print("Final reply:", final_reply)

        pred_tool_names = []
        pred_call_objs  = []
        for m in result["messages"]:
            if not isinstance(m, AIMessage):
                continue
            for tc in m.additional_kwargs.get("tool_calls", []):
                if "function" in tc:
                    name   = tc["function"]["name"]
                    params = json.loads(tc["function"]["arguments"])
                else:
                    name   = tc.get("name")
                    params = tc.get("args", {})
                pred_tool_names.append(name)
                pred_call_objs.append({"tool": name, "params": params})

        pr = phrase_recall(final_reply, exp_final.get("customer_msg_contains", []))
        metrics["phrase_recall"].append(pr)
        tm = tool_metrics(pred_tool_names, exp_final.get("tool_calls", []))
        metrics["tool_recall"].append(tm["tool_recall"])
        metrics["tool_precision"].append(tm["tool_precision"])
        metrics["param_accuracy"].append(
            param_accuracy(pred_call_objs, exp_final.get("tool_calls", []))
        )
        metrics["task_success"].append(
            task_success(final_reply, pred_tool_names, exp_final)
        )

    print("\n=== Aggregate scores ===")
    for k, vals in metrics.items():
        print(f"{k:15s}: {stats.mean(vals):.3f} (n={len(vals)})")

    active = {m: weights.get(m, 1.0) for m in metrics}
    total = sum(active.values())
    overall = sum(stats.mean(metrics[m]) * w for m, w in active.items()) / total
    print(f"\nWeighted overall score: {overall:.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph_py")
    ap.add_argument("dataset")
    ap.add_argument("--weights", nargs="*", default=[])
    ap.add_argument("--verbose", default=False)
    args = ap.parse_args()

    graph = load_graph(args.graph_py)
    weights = parse_weights(args.weights) or {"task_success": 1.0}

    metrics: Dict[str, List[float]] = {
        "task_success": [],
        "phrase_recall": [],
        "tool_recall": [],
        "tool_precision": [],
        "param_accuracy": [],
    }

    run_evaluation(graph, args, weights, metrics, verbose=args.verbose)


if __name__ == "__main__":
    main()
