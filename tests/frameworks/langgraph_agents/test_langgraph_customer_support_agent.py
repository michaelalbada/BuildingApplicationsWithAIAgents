import pytest
from langchain.schema import HumanMessage
from frameworks.langgraph_agents.ecommerce_customer_support.customer_support_agent import graph

def extract_tool_names(messages):
    return [getattr(m, "name", None) for m in messages if hasattr(m, "name")]

def test_refund_flow():
    order = {"order_id": "A12345", "status": "Delivered", "total": 19.99}
    user_msg = [HumanMessage(content="My mug arrived broken. Refund?")]

    result = graph.invoke({"order": order, "messages": user_msg})
    messages = result["messages"]

    assert len(messages) >= 3, "Should return at least 3 messages"

    assert any("refund" in m.content.lower() for m in messages), "Should confirm refund in user message"

def test_cancel_flow():
    order = {"order_id": "B54321", "status": "Processing", "total": 59.99}
    user_msg = [HumanMessage(content="Please cancel my order, I don't need it anymore.")]

    result = graph.invoke({"order": order, "messages": user_msg})
    messages = result["messages"]

    assert any("cancel" in m.content.lower() for m in messages), "Should confirm cancellation"

if __name__ == "__main__":
    pytest.main([__file__])
