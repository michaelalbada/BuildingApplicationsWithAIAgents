from __future__ import annotations
"""
customer_support_agent.py
LangGraph workflow for an e-commerce customer-support agent,
using LangGraph's built-in tool-calling via @tool decorators.
"""
import json
import operator
import builtins
from typing import Annotated, Sequence, TypedDict

from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.tool import ToolMessage

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

llm = ChatOpenAI(model="gpt-4o", temperature=0.0).bind_tools(TOOLS)

class AgentState(TypedDict):
    order: dict
    messages: Annotated[Sequence[BaseMessage], operator.add]

def call_model(state: AgentState):
    history = state["messages"]
    order_json = json.dumps(state["order"], ensure_ascii=False)
    system_prompt = (
        "You are a helpful e-commerce support agent. "
        "When you act, you MUST do exactly TWO steps in order:\n"
        "  1) call one business tool (issue_refund / cancel_order / modify_order)\n"
        "  2) call send_customer_message with confirmation text\n"
        "Then STOP.\n\n"
        "Example 1 (before photo):\n"
        "  User: 'My mug arrived broken. Refund please.'\n"
        "  Assistant (tool_calls): {'name':'send_customer_message','arguments':{'text':'Please send us a quick photo of the damage so we can process a full refund.'}}\n"
        "  Assistant: 'Please send us a quick photo of the damage so we can process a full refund.'\n\n"
        "Example 2 (after photo):\n"
        "  User: 'Sure, here's the photo. *[customer uploads image]*'\n"
        "  Assistant (tool_calls): {'name':'issue_refund','arguments':{'order_id':'<order_id>','amount':<amount>}}\n"
        "  Assistant (tool_calls): {'name':'send_customer_message','arguments':{'text':'Your refund of <amount> has been processed! You’ll see it in 3-5 business days.'}}\n"
        "  Assistant: 'Your refund of <amount> has been processed! You'll see it in 3-5 business days.'\n\n"
        f"ORDER: {order_json}"
    )
    full = [SystemMessage(content=system_prompt)] + history
    # this single invoke will handle both the function call and the follow-up message
    final_reply = llm.invoke(full)
    return {"messages": [final_reply]}

def construct_graph():
    g = StateGraph(AgentState)
    g.add_node("assistant", call_model)
    g.set_entry_point("assistant")
    return g.compile()

graph = construct_graph()

if __name__ == "__main__":
    example = {"order_id":"A12345","status":"Delivered","total":19.99}
    convo = [HumanMessage(content="My mug arrived broken. Refund?")]
    result = graph.invoke({"order": example, "messages": convo})
    for m in result["messages"]:
        print(f"{m.type}: {m.content}")
