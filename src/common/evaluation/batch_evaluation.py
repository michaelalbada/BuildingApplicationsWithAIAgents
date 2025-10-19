"""
batch_evaluation.py
Handles multi-turn conversations and surfaces diagnostics, including parameter accuracy.
"""
from __future__ import annotations
import builtins
from langchain.schema import BaseMessage
import operator
from typing import Dict, List, Annotated, Sequence, Optional
builtins.Annotated = Annotated
builtins.BaseMessage = BaseMessage
builtins.operator = operator
builtins.Sequence = Sequence
builtins.List = List
builtins.Optional = Optional
import argparse
import importlib.util
import json
import pathlib
import statistics as stats
from src.common.observability.loki_logger import log_to_loki
from src.common.evaluation.metrics import phrase_recall, tool_metrics, param_accuracy, task_success
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

def evaluate_single_instance(raw: str, graph) -> Optional[Dict[str, float]]:
    """
    Parse one JSONL line, run the graph, compute metrics, or return None if skipped.
    """
    if not raw.strip():
        return None
    try:
        ex = json.loads(raw)
        print(f"[DEBUG] Parsed JSON keys: {list(ex.keys())}")
        
        if "input" in ex and "expected_function_call" in ex:
            messages = [to_lc_message(t) for t in ex["input"]]
            expected_call = ex["expected_function_call"]
            exp_final = {
                "tool_calls": [{
                    "tool": expected_call["name"],
                    "params": expected_call["arguments"]
                }],
                "customer_msg_contains": []
            }

            # Extract relevant IDs from expected function call arguments
            order_id = None
            patient_id = None
            customer_id = None
            account_id = None
            
            args = expected_call.get("arguments", {})
            if "order_id" in args:
                order_id = args["order_id"]
            if "patient_id" in args:
                patient_id = args["patient_id"]
            if "customer_id" in args:
                customer_id = args["customer_id"]
            if "account_id" in args:
                account_id = args["account_id"]

            # Create appropriate dummy data based on the expected function call
            initial_state = {"messages": messages}
            
            if order_id or expected_call["name"] in ["issue_refund", "cancel_order", "modify_order"]:
                # E-commerce scenario
                initial_state["order"] = {
                    "order_id": order_id or "UNKNOWN",
                    "customer_id": "EVAL_CUSTOMER",
                    "items": [],
                    "total": 0.0,
                    "status": "pending"
                }
            elif patient_id or expected_call["name"] in ["assess_symptoms", "register_patient", "schedule_appointment"]:
                # Healthcare scenario
                initial_state["patient"] = {
                    "patient_id": patient_id or "UNKNOWN",
                    "name": "Eval Patient",
                    "status": "active"
                }
            elif customer_id or account_id or expected_call["name"] in ["investigate_transaction", "freeze_account", "process_loan_application"]:
                # Financial services scenario
                initial_state["account"] = {
                    "account_id": account_id or customer_id or "UNKNOWN",
                    "customer_id": customer_id or "UNKNOWN",
                    "status": "active"
                }
            elif expected_call["name"] in ["provision_user_access", "troubleshoot_network", "diagnose_system_issue", 
                                           "deploy_software", "contain_security_incident", "troubleshoot_hardware"]:
                # IT Help Desk scenario
                initial_state["ticket"] = {
                    "ticket_id": "EVAL_TICKET",
                    "user_id": customer_id or "UNKNOWN",
                    "priority": "medium",
                    "status": "open"
                }
            elif expected_call["name"] in ["review_contract", "research_case_law", "client_intake", "assess_compliance", 
                                           "manage_discovery", "calculate_damages", "track_deadlines"]:
                # Legal Document Review scenario
                initial_state["matter"] = {
                    "matter_id": "EVAL_MATTER",
                    "client_id": args.get("client_name", "UNKNOWN"),
                    "matter_type": args.get("matter_type", "general"),
                    "status": "active"
                }
            elif expected_call["name"] in ["lookup_threat_intel", "query_logs", "triage_incident", "isolate_host"]:
                # Security Operations Center scenario
                initial_state["incident"] = {
                    "incident_id": args.get("incident_id", "EVAL_INCIDENT"),
                    "severity": "medium",
                    "status": "investigating",
                    "analyst": "SOC_EVAL"
                }
            else:
                # Supply Chain & Logistics scenario (default fallback)
                initial_state["operation"] = {
                    "operation_id": "EVAL_OPERATION",
                    "type": "general",
                    "priority": "medium",
                    "status": "active"
                }
            
        elif "order" in ex and "conversation" in ex:
            # Legacy format
            order = ex["order"]
            messages = [to_lc_message(t) for t in ex["conversation"]]
            exp_final = ex["expected"]["final_state"]
            initial_state = {"order": order, "messages": messages}
        else:
            print(f"[SKIPPED] Unrecognized format: {list(ex.keys())}")
            return None

        result = graph.invoke(initial_state)

        routing_pred = None
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and msg.content.strip().lower() in ["inventory", "transportation", "supplier"]:
                routing_pred = msg.content.strip().lower()
                break

        final_reply = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                calls = msg.additional_kwargs.get("tool_calls", [])
                if not calls:
                    final_reply = msg.content or ""
                    break

        pred_tool_names: List[str] = []
        pred_call_objs: List[dict] = []
        for m in result["messages"]:
            if not isinstance(m, AIMessage):
                continue
            for tc in m.additional_kwargs.get("tool_calls", []):
                if "function" in tc:
                    name = tc["function"]["name"]
                    params = json.loads(tc["function"]["arguments"])
                else:
                    name = tc.get("name")
                    params = tc.get("args", {})
                pred_tool_names.append(name)
                pred_call_objs.append({"tool": name, "params": params})

        expected_routing = ex.get("expected_routing", "").lower()
        routing_acc = 1.0 if routing_pred == expected_routing else 0.0

        tm = tool_metrics(pred_tool_names, exp_final.get("tool_calls", []))
        
        return {
            "phrase_recall": phrase_recall(final_reply, exp_final.get("customer_msg_contains", [])),
            "tool_recall": tm["tool_recall"],
            "tool_precision": tm["tool_precision"],
            "param_accuracy": param_accuracy(pred_call_objs, exp_final.get("tool_calls", [])),
            "task_success": task_success(final_reply, pred_tool_names, exp_final),
            "routing_accuracy": routing_acc,
        }

    except Exception as e:
        print(f"[SKIPPED] example failed with error: {e!r}")
        import traceback
        traceback.print_exc()
        return None

def run_evaluation(
    graph, args, weights: Dict[str, float],
    metrics: Dict[str, List[float]], verbose: bool = False
):
    for raw in pathlib.Path(args.dataset).read_text().splitlines():
        result = evaluate_single_instance(raw, graph)
        if result is None:
            try:
                log_to_loki("batch_eval", "[SKIPPED] evaluation failed or malformed entry.")
            except:
                print("[SKIPPED] evaluation failed or malformed entry.")
            continue
        else:
            try:
                log_to_loki("batch_eval", f"[METRICS] {json.dumps(result)}")
            except:
                print(f"[METRICS] {json.dumps(result)}")
        # Append to global metrics
        for k, v in result.items():
            if k not in metrics:
                metrics[k] = []
            metrics[k].append(v)

    print("\n=== Aggregate scores ===")
    if any(len(vals) > 0 for vals in metrics.values()):
        for k, vals in metrics.items():
            if len(vals) > 0:
                print(f"{k:15s}: {stats.mean(vals):.3f} (n={len(vals)})")
            else:
                print(f"{k:15s}: No data")

        active_metrics = {k: vals for k, vals in metrics.items() if len(vals) > 0}
        if active_metrics:
            active = {m: weights.get(m, 1.0) for m in active_metrics}
            total = sum(active.values())
            overall = sum(stats.mean(active_metrics[m]) * w for m, w in active.items()) / total
            print(f"\nWeighted overall score: {overall:.3f}")
    else:
        print("No successful evaluations completed.")
    try:
        if any(len(vals) > 0 for vals in metrics.values()):
            log_lines = [f"{k}: {stats.mean(vals):.3f}" for k, vals in metrics.items() if len(vals) > 0]
            summary = "Evaluation Summary - " + " | ".join(log_lines)
            log_to_loki("batch_eval", summary)
    except:
        pass  # Don't crash if logging fails

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_py")
    ap.add_argument("--dataset")
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

'''
Example usage
python -m src.common.evaluation.batch_evaluation \
  --dataset src/common/evaluation/scenarios/ecommerce_customer_support_evaluation_set.json \
  --graph_py src/frameworks/langgraph_agents/ecommerce_customer_support/customer_support_agent.py
'''

if __name__ == "__main__":
    main()
