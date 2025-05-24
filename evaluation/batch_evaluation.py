from __future__ import annotations
"""
batch_evaluation.py – outcome + diagnostic metrics
=================================================
Evaluates end-to-end customer-support scenarios **and** surfaces
intermediate diagnostics so you can see *why* a case failed.

Dataset schema
--------------
Same as before (order, conversation, expected.final_state).

Metrics produced
----------------
* **task_success**      1 if all tools + phrases matched, else 0
* **phrase_recall**     fraction of required substrings present in final reply
* **tool_recall**       fraction of required tools actually invoked
* **tool_precision**    |{expected ∩ used}| / |used| (0 if no tools used)

You can weight any subset with `--weights metric=weight`.

Example
~~~~~~~
```bash
python evaluation/batch_evaluation.py \
       langgraph/.../customer_support_agent.py \
       evaluation_sets/ecommerce_customer_support.json \
       --weights task_success=0.6 tool_recall=0.2 phrase_recall=0.2
```
"""

import argparse
import importlib.util
import json
import pathlib
import statistics as stats
from typing import Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

# ----------------------------------------------------------------------------
# util: convert JSON turn -> LangChain message
# ----------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------
# load graph from python file
# ----------------------------------------------------------------------------

def load_graph(path: str):
    spec = importlib.util.spec_from_file_location("user_graph", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(mod)                 # type: ignore
    if hasattr(mod, "graph"):
        return mod.graph
    if hasattr(mod, "construct_graph"):
        return mod.construct_graph()  # type: ignore
    raise AttributeError(f"{path} exposes neither `graph` nor `construct_graph()`")

# ----------------------------------------------------------------------------
# diagnostic sub-metrics
# ----------------------------------------------------------------------------

def phrase_recall(pred_reply: str, phrases: List[str]) -> float:
    if not phrases:
        return 1.0
    found = sum(1 for p in phrases if p.lower() in pred_reply.lower())
    return found / len(phrases)


def tool_metrics(pred_tools: List[str], expected_tools: List[str]):
    if not expected_tools:
        return {"tool_recall": 1.0, "tool_precision": 1.0}
    exp_set = set(expected_tools)
    pred_set = set(pred_tools)
    tp = len(exp_set & pred_set)
    recall = tp / len(exp_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    return {"tool_recall": recall, "tool_precision": precision}


def task_success(pred_reply: str, pred_tools: List[str], expected: dict) -> float:
    phrases_ok = phrase_recall(pred_reply, expected.get("customer_msg_contains", [])) == 1.0
    tools_ok = tool_metrics(pred_tools, expected.get("tool_calls", []))["tool_recall"] == 1.0
    return float(phrases_ok and tools_ok)

# ----------------------------------------------------------------------------
# main evaluation loop
# ----------------------------------------------------------------------------

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph_py")
    ap.add_argument("dataset")
    ap.add_argument("--weights", nargs="*", default=[])
    args = ap.parse_args()

    graph = load_graph(args.graph_py)
    weights = parse_weights(args.weights) or {"task_success": 1.0}

    # collect metric vectors
    metrics: Dict[str, List[float]] = {
        "task_success": [],
        "phrase_recall": [],
        "tool_recall": [],
        "tool_precision": [],
    }

    for line in pathlib.Path(args.dataset).read_text().splitlines():
        ex = json.loads(line)
        order = ex["order"]
        messages = [to_lc_message(t) for t in ex["conversation"]]
        exp_final = ex["expected"]["final_state"]

        result = graph.invoke({"order": order, "messages": messages})
        final_msg = result["messages"][-1].content

        # extract tool names used anywhere in run
        used_tools: List[str] = []
        for m in result["messages"]:
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                used_tools.extend(tc["name"] for tc in m.tool_calls)

        # compute metrics
        metrics["phrase_recall"].append(
            phrase_recall(final_msg, exp_final.get("customer_msg_contains", []))
        )
        tm = tool_metrics(used_tools, exp_final.get("tool_calls", []))
        metrics["tool_recall"].append(tm["tool_recall"])
        metrics["tool_precision"].append(tm["tool_precision"])
        metrics["task_success"].append(
            task_success(final_msg, used_tools, exp_final)
        )

    # print aggregate
    print("\n=== Aggregate scores ===")
    for k, vals in metrics.items():
        if vals:  # avoid division by zero if list empty
            print(f"{k:15s}: {stats.mean(vals):.3f} (n={len(vals)})")

    # weighted overall
    active = {m: weights.get(m, 1.0) for m in metrics}
    total = sum(active.values())
    overall = sum(stats.mean(metrics[m]) * w for m, w in active.items()) / total
    print(f"\nWeighted overall score: {overall:.3f}")


if __name__ == "__main__":
    main()
