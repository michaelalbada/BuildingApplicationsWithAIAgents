"""customer_support_agent.py
Enhanced LangGraph workflow for an e‑commerce customer‑support agent.

Key features
~~~~~~~~~~~~
* Order‑aware: passes `order` JSON as a system message.
* Dummy tool stubs (replace with real APIs).
* Exported module‑level **`graph`** for easy import.
"""

from __future__ import annotations
import json
import operator
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

def send_customer_message(text: str) -> str:
    print(f"[TOOL] send_customer_message → {text}")
    return "OK"

def issue_refund(order_id: str, amount: float) -> str:
    print(f"[TOOL] issue_refund(order={order_id}, amount={amount})")
    return "refund_queued"

def cancel_order(order_id: str) -> str:
    print(f"[TOOL] cancel_order(order={order_id})")
    return "cancelled"

def modify_order(order_id: str, shipping_address: dict) -> str:
    print(f"[TOOL] modify_order(order={order_id}, to={shipping_address})")
    return "address_updated"

tools = {
    "send_customer_message": send_customer_message,
    "issue_refund": issue_refund,
    "cancel_order": cancel_order,
    "modify_order": modify_order,
}

class AgentState(TypedDict):
    order: dict
    messages: Annotated[Sequence[BaseMessage], operator.add]

def call_llm(state: AgentState, *, config=None):
    """Invoke the LLM with order context + conversation."""

    order_json = json.dumps(state["order"], ensure_ascii=False, separators=(',', ':'))
    system_prompt = (
        "You are a helpful e‑commerce customer support agent. "
        "Here are the order details in JSON: " + order_json
    )
    history = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = llm.invoke(history, config=config)
    return {"messages": [response]}

def should_continue(state: AgentState):
    last = state["messages"][-1]
    return "do_tools" if getattr(last, "tool_calls", None) else "done"

def run_tools(state: AgentState):
    ai_msg: AIMessage = state["messages"][-1]  # type: ignore
    results: list[ToolMessage] = []
    for call in ai_msg.tool_calls:
        fn = tools.get(call["name"])
        if fn is None:
            results.append(ToolMessage(content="ERROR: unknown tool", tool_call_id=call["id"]))
            continue
        tool_result = fn(**call["args"])
        results.append(ToolMessage(content=str(tool_result), tool_call_id=call["id"]))
    return {"messages": results}

def construct_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_llm)
    workflow.add_node("action", run_tools)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"do_tools": "action", "done": END},
    )
    workflow.add_edge("action", "agent")
    return workflow.compile()

if __name__ == "__main__":
    graph = construct_graph()
    order_example = {
        "order_id": "A12345",
        "status": "Delivered",
        "total": 39.99,
        "currency": "USD",
        "items": [
            {"sku": "MUG-001", "name": "Ceramic Coffee Mug", "qty": 1, "unit_price": 19.99}
        ],
        "delivered_at": "2025-05-15",
    }
    convo = [HumanMessage(content="Hi, my coffee mug arrived cracked. Can I get a refund?")]
    result = graph.invoke({"order": order_example, "messages": convo})
    for m in result["messages"]:
        print(f"{m.type}: {m.content}")
