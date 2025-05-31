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
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.tools import tool
from langgraph.graph import StateGraph, END

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

@workflow(name='send_customer_message')
@tool
def send_customer_message(order_id: str, text: str) -> str:
    """Send a plain response to the customer."""
    print(f"[TOOL] send_customer_message → {text}")
    return "sent"

@workflow(name='issue_refund')
@tool
def issue_refund(order_id: str, amount: float) -> str:
    """Issue a refund for the given order."""
    print(f"[TOOL] issue_refund(order_id={order_id}, amount={amount})")
    return "refund_queued"

@workflow(name='cancel_order')
@tool
def cancel_order(order_id: str) -> str:
    """Cancel an order that hasn't shipped."""
    print(f"[TOOL] cancel_order(order_id={order_id})")
    return "cancelled"

@workflow(name='modify_order')
@tool
def modify_order(order_id: str, shipping_address: dict) -> str:
    """Change the shipping address for a pending order."""
    print(f"[TOOL] modify_order(order_id={order_id}, address={shipping_address})")
    return "address_updated"

TOOLS = [send_customer_message, issue_refund, cancel_order, modify_order]

Traceloop.init(disable_batch=True)
llm = ChatOpenAI(model="gpt-4o", temperature=0.0, callbacks=[StreamingStdOutCallbackHandler()],  
    verbose=True).bind_tools(TOOLS)

class AgentState(TypedDict):
    order: dict
    messages: Annotated[Sequence[BaseMessage], operator.add]

@workflow(name='call_model')
def call_model(state: AgentState):
    history = state["messages"]
    order_json = json.dumps(state["order"], ensure_ascii=False)
    system_prompt = (
        "You are a helpful e-commerce support agent.\n"
        "When you act, you MUST do exactly TWO steps in order:\n"
        "  1) call one business tool (issue_refund / cancel_order / modify_order)\n"
        "  2) call send_customer_message with confirmation text\n"
        "Then STOP.\n\n"
        f"ORDER: {order_json}"
    )

    full = [SystemMessage(content=system_prompt)] + history

    first: ToolMessage | BaseMessage = llm.invoke(full)
    messages = [first]

    if getattr(first, "tool_calls", None):
        for tc in first.tool_calls:
            print(first)
            print(tc['name'])
            fn = next(t for t in TOOLS if t.name == tc['name'])
            out = fn.invoke(tc["args"])
            messages.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))

        second = llm.invoke(full + messages)
        messages.append(second)

    return {"messages": messages}

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
