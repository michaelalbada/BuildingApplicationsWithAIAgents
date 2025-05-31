import asyncio
import json
import os

from langchain.schema import HumanMessage
from langchain.tools import Tool
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.graph import StateGraph, END
from typing import TypedDict, Sequence, Any

class AgentState(TypedDict):
    messages: Sequence[Any]  # A list of BaseMessage/HumanMessage/…


mcp_client = MultiServerMCPClient(
    {
        "math": {
            "command": "python3",
            "args": ["src/common/mcp/MCP_weather_server.py"],
            "transport": "stdio",
        },
        "weather": {
            # Ensure weather_server.py is running on port 8000
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        },
    }
)

async def get_mcp_tools() -> list[Tool]:
    return await mcp_client.get_tools()

async def call_mcp_tools(state: AgentState) -> dict[str, Any]:
    """
    Given AgentState with state["messages"], decide which MCP tool to call.
    For demonstration, if the user message mentions 'weather', call weather tool;
    if it contains a math expression (digits/operators), call math tool.
    Otherwise, return a fallback message.
    """
    messages = state["messages"]
    last_msg = messages[-1].content.lower()

    # Load tools from the client if not already cached
    global MCP_TOOLS
    if "MCP_TOOLS" not in globals():
        MCP_TOOLS = await mcp_client.get_tools()

    if any(token in last_msg for token in ["+", "-", "*", "/", "(", ")"]):
        tool_name = "math"
    elif "weather" in last_msg:
        tool_name = "weather"
    else:
        # No matching tool: return a default text response
        return {"messages": [{"role": "assistant", "content": "Sorry, I can only answer math or weather queries."}]}

    tool_obj = next(t for t in MCP_TOOLS if t.name == tool_name)

    user_input = messages[-1].content
    mcp_result: str = await tool_obj.arun(user_input)

    return {
        "messages": [
            {"role": "assistant", "content": mcp_result}
        ]
    }

def construct_graph():
    g = StateGraph(AgentState)
    g.add_node("assistant", call_mcp_tools)
    g.set_entry_point("assistant")
    return g.compile()

GRAPH = construct_graph()

async def run_math_query():
    initial_state = {
        "messages": [
            HumanMessage(content="What is (3 + 5) * 12?")
        ]
    }
    result = await GRAPH.ainvoke(initial_state)
    assistant_msg = result["messages"][-1]
    print("Math answer:", assistant_msg.content)


async def run_weather_query():
    initial_state = {
        "messages": [
            HumanMessage(content="What is the weather in NYC?")
        ]
    }
    result = await GRAPH.ainvoke(initial_state)
    assistant_msg = result["messages"][-1]
    print("Weather answer:", assistant_msg.content)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_math_query())
    loop.run_until_complete(run_weather_query())
