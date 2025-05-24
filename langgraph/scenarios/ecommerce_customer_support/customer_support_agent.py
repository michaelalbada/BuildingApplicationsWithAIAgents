from __future__ import annotations
"""
customer_support_agent.py
LangGraph workflow for an e-commerce customer-support agent,
using LangGraph's built-in tool-calling via @tool decorators.
"""
import json
import operator
from typing import Annotated, Sequence, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain.tools import tool
from langgraph.graph import StateGraph, END

@tool
def send_customer_message(text: str) -> str:
    """Send a plain response to the customer."""
    print(f"[TOOL] send_customer_message → {text}")
    return "sent"

@tool
def issue_refund(order_id: str, amount: float) -> str:
    """Issue a refund for the given order."""
    print(f"[TOOL] issue_refund(order_id={order_id}, amount={amount})")
    return "refund_queued"

@tool
def cancel_order(order_id: str) -> str:
    """Cancel an order that hasn't shipped."""
    print(f"[TOOL] cancel_order(order_id={order_id})")
    return "cancelled"

@tool
def modify_order(order_id: str, shipping_address: dict) -> str:
    """Change the shipping address for a pending order."""
    print(f"[TOOL] modify_order(order_id={order_id}, address={shipping_address})")
    return "address_updated"

TOOLS = [send_customer_message, issue_refund, cancel_order, modify_order]

llm = ChatOpenAI(model="gpt-4o", temperature=0.1).bind_tools(TOOLS)

class AgentState(TypedDict):
    order: dict
    messages: Annotated[Sequence[BaseMessage], operator.add]

def call_model(state: AgentState):
    # Remove any previous ToolMessage before calling the LLM
    prior = [m for m in state["messages"] if not isinstance(m, ToolMessage)]

    order_json = json.dumps(state["order"], ensure_ascii=False)
    system_prompt = (
        "You are a helpful e-commerce support agent.\n"
        "POLICY:\n"
        "- Ask for a photo before refunding damaged items.\n"
        "- Cancel only if status is Processing or Pending Shipment.\n"
        "- Modify address only if status is Pending Shipment.\n\n"
        "When you choose to act, call exactly one tool. Afterwards, send one plain-text reply "
        "including key phrases like 'full refund', 'cancelled', 'updated', '3-5 business days'."
        f"ORDER: {order_json}"
    )
    history = [SystemMessage(content=system_prompt)] + prior
    response = llm.invoke(history)
    return {"messages": [response]}

def execute_tool(state: AgentState):
    ai_msg: AIMessage = state["messages"][-1]  # last assistant with tool_calls
    outputs: list[ToolMessage] = []
    for call in ai_msg.tool_calls:
        fn = {t.name: t for t in TOOLS}[call["name"]]
        res = fn.invoke(call["args"])
        outputs.append(ToolMessage(content=str(res), tool_call_id=call["id"]))
    return {"messages": outputs}

def continue_or_end(state: AgentState):
    last = state["messages"][-1]
    return "tool_step" if getattr(last, "tool_calls", None) else "end"

def construct_graph():
    g = StateGraph(AgentState)
    g.add_node("assistant", call_model)
    g.add_node("tool_step", execute_tool)
    g.set_entry_point("assistant")
    g.add_conditional_edges(
        "assistant",
        continue_or_end,
        {"tool_step": "tool_step", "end": END},
    )
    g.add_edge("tool_step", "assistant")
    return g.compile()

graph = construct_graph()

if __name__ == "__main__":
    example = {"order_id":"A12345","status":"Delivered","total":19.99}
    convo = [HumanMessage(content="My mug arrived broken. Refund?")]
    result = graph.invoke({"order": example, "messages": convo})
    for m in result["messages"]:
        print(f"{m.type}: {m.content}")
