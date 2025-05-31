import asyncio
import json
import os

from langchain.schema import HumanMessage
from langchain.tools import Tool
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.graph import StateGraph, END
from typing import TypedDict, Sequence, Any

# 1) Define the AgentState TypedDict
class AgentState(TypedDict):
    messages: Sequence[Any]  # A list of BaseMessage/HumanMessage/…


# 2) Initialize MultiServerMCPClient
mcp_client = MultiServerMCPClient(
    {
        "math": {
            "command": "python3",
            # Replace with the absolute path to math_server.py
            "args": ["/absolute/path/to/math_server.py"],
            "transport": "stdio",
        },
        "weather": {
            # Ensure weather_server.py is running on port 8000
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        },
    }
)

# 3) Fetch MCP tools asynchronously
async def get_mcp_tools() -> list[Tool]:
    return await mcp_client.get_tools()

# 4) Define an async node function that picks a tool based on the last message
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

    # Decide tool name
    if any(token in last_msg for token in ["+", "-", "*", "/", "(", ")"]):
        tool_name = "math"
    elif "weather" in last_msg:
        tool_name = "weather"
    else:
        # No matching tool: return a default text response
        return {"messages": [{"role": "assistant", "content": "Sorry, I can only answer math or weather queries."}]}

    # Find the Tool object by name
    tool_obj = next(t for t in MCP_TOOLS if t.name == tool_name)

    # Build the MCPRequest payload (LangChain Tool.run expects just the raw user content)
    user_input = messages[-1].content
    # The `Tool.run()` call under the hood will wrap this as an MCPRequest using default context
    mcp_result: str = await tool_obj.arun(user_input)

    # Return a single assistant message with the raw tool output
    return {
        "messages": [
            {"role": "assistant", "content": mcp_result}
        ]
    }


# 5) Construct and compile the LangGraph
def construct_graph():
    g = StateGraph(AgentState)
    g.add_node("assistant", call_mcp_tools)
    g.set_entry_point("assistant")
    return g.compile()

GRAPH = construct_graph()


# 6) Example: Invoke the graph with a math question
async def run_math_query():
    initial_state = {
        "messages": [
            HumanMessage(content="What is (3 + 5) * 12?")
        ]
    }
    result = await GRAPH.ainvoke(initial_state)
    assistant_msg = result["messages"][-1]
    print("Math answer:", assistant_msg.content)


# 7) Example: Invoke the graph with a weather question
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
    # For demonstration, run both queries sequentially
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_math_query())
    loop.run_until_complete(run_weather_query())
