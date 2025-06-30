from __future__ import annotations
"""
customer_support_agent.py
LangGraph workflow for an e-commerce customer-support agent,
using LangGraph's built-in tool-calling via @tool decorators.
"""
import os
import json
import operator
import builtins
from typing import Annotated, Sequence, TypedDict, Optional

from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.tool import ToolMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.tools import tool
from langgraph.graph import StateGraph, END

from traceloop.sdk import Traceloop
from src.common.observability.loki_logger import log_to_loki



os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4317"
os.environ["OTEL_EXPORTER_OTLP_INSECURE"] = "true"

@tool
def send_customer_message(order_id: str, text: str) -> str:
    """Send a plain response to the customer."""
    print(f"[TOOL] send_customer_message → {text}")
    log_to_loki("tool.send_customer_message", f"order_id={order_id}, text={text}")
    return "sent"

@tool
def issue_refund(order_id: str, amount: float) -> str:
    """Issue a refund for the given order."""
    print(f"[TOOL] issue_refund(order_id={order_id}, amount={amount})")
    log_to_loki("tool.issue_refund", f"order_id={order_id}, amount={amount}")
    return "refund_queued"

@tool
def cancel_order(order_id: str) -> str:
    """Cancel an order that hasn't shipped."""
    print(f"[TOOL] cancel_order(order_id={order_id})")
    log_to_loki("tool.cancel_order", f"order_id={order_id}")
    return "cancelled"

@tool
def update_address_for_order(order_id: str, shipping_address: dict) -> str:
    """Change the shipping address for a pending order."""
    print(f"[TOOL] update_address_for_order(order_id={order_id}, address={shipping_address})")
    log_to_loki("tool.update_address_for_order", f"order_id={order_id}, address={shipping_address}")
    return "address_updated"

TOOLS = [send_customer_message, issue_refund, cancel_order, update_address_for_order]

Traceloop.init(disable_batch=True, app_name="customer_support_agent")
llm = ChatOpenAI(model="gpt-4o", temperature=0.0, callbacks=[StreamingStdOutCallbackHandler()],  
    verbose=True).bind_tools(TOOLS)

class AgentState(TypedDict):
    order: Optional[dict]  # Make order optional
    messages: Annotated[Sequence[BaseMessage], operator.add]

def call_model(state: AgentState):
    history = state["messages"]
    
    # Handle missing or incomplete order data gracefully
    order = state.get("order", {})
    if not order:
        order = {"order_id": "UNKNOWN", "status": "unknown", "total": 0.0}
    
    order_json = json.dumps(order, ensure_ascii=False)
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
