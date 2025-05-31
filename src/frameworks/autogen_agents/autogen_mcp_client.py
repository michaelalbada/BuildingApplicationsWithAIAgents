#!/usr/bin/env python3
"""
Example: Autogen agent that uses MCP tools (math and weather).

 - Starts by configuring a MultiServerMCPClient with:
     • "math"  → stdio transport pointing at math_server.py
     • "weather" → HTTP transport pointing at http://localhost:8000/mcp

 - Wraps each LangChain Tool in a simple async function so Autogen can call it.

 - Spins up a basic Autogen AssistantAgent with those two tools. 
"""

import asyncio
from typing import List

# 1) Import the MCP client and the base Tool type from LangChain
from langchain.tools import Tool
from langchain_mcp_adapters.client import MultiServerMCPClient

# 2) Import Autogen primitives
from openai_autogen import (
    AIConfig,
    AssistantAgent,
    UserProxyAgent,
)

# ────────────────────────────────────────────────────────────────────────────────
# STEP A: Configure the MultiServerMCPClient
# ────────────────────────────────────────────────────────────────────────────────

# NOTE: Adjust the paths/URLs below to match where math_server.py and weather_server.py live in your filesystem.
MCP_CONFIG = {
    "math": {
        "command": "python3",
        "args": ["/absolute/path/to/math_server.py"],
        "transport": "stdio",
    },
    "weather": {
        "url": "http://localhost:8000/mcp",
        "transport": "streamable_http",
    },
}


# ────────────────────────────────────────────────────────────────────────────────
# STEP B: Fetch the LangChain Tools from MCP
# ────────────────────────────────────────────────────────────────────────────────

async def get_mcp_tools() -> List[Tool]:
    """
    Instantiate a MultiServerMCPClient, fetch its Tools,
    and return them as a list of LangChain Tool objects.
    """
    client = MultiServerMCPClient(MCP_CONFIG)
    return await client.get_tools()


# ────────────────────────────────────────────────────────────────────────────────
# STEP C: Convert each LangChain Tool into an Autogen-usable async function
# ────────────────────────────────────────────────────────────────────────────────

def wrap_tool_for_autogen(lc_tool: Tool):
    """
    Given a LangChain Tool, produce a callable async function that Autogen can call.
    Autogen expects each tool to be an async function taking a single string
    (the user’s input) and returning a string (the tool’s output).
    """
    async def _fn(user_input: str) -> str:
        # Under the hood, lc_tool.arun(...) will wrap `user_input` in an MCPRequest,
        # send it to the corresponding server, and return the raw text from MCPResponse.
        return await lc_tool.arun(user_input)

    # Return a tuple: (tool_name, the async function, tool_description)
    return lc_tool.name, _fn, lc_tool.description


# ────────────────────────────────────────────────────────────────────────────────
# STEP D: Create a simple Autogen assistant that “knows” these two tools
# ────────────────────────────────────────────────────────────────────────────────

async def main():
    # 1) Fetch MCP tools from LangChainMCPClient
    lc_tools = await get_mcp_tools()  # e.g. [Tool(name="math", ...), Tool(name="weather", ...)]

    # 2) Wrap each one into an Autogen‐compatible Tool descriptor
    autogen_tools = []
    for lc_tool in lc_tools:
        name, func, description = wrap_tool_for_autogen(lc_tool)
        autogen_tools.append(
            {
                "name":        name,
                "function":    func,
                "description": description or f"Invoke the `{name}` MCP server",
            }
        )

    # 3) Configure your OpenAI LLM (or Anthropic, etc.)
    #    Here we assume you have OPENAI_API_KEY set in your environment.
    ai_config = AIConfig(
        model="gpt-4o", 
        temperature=0.0, 
        openai_api_key=None  # None→Autogen will pull from OPENAI_API_KEY env var
    )

    # 4) Create the AssistantAgent with MCP Tools
    assistant = AssistantAgent(
        name="mcp_assistant",
        ai_config=ai_config,
        tools=autogen_tools
    )

    # 5) Create a “User” proxy so you can simulate a conversation
    user = UserProxyAgent(assistant)

    # ─────────────────────────────────────────────────────────────────────────────
    # Now you can ask the assistant questions. Internally, Autogen will:
    #    • Package your question into a single string
    #    • Detect which tool to call (based on how you configure the agent prompt)
    #    • Invoke the corresponding TCP/HTTP/stdio transport for math or weather
    #    • Unwrap the MCPResponse and deliver the result back to you as an assistant message
    # ─────────────────────────────────────────────────────────────────────────────

    # Example 1: A math question
    math_query = "What's (3 + 5) * 12?"
    print("User asks:", math_query)
    math_answer = await user.send_message(math_query)
    print("Assistant replies:", math_answer, "\n")

    # Example 2: A weather question
    weather_query = "What is the weather in NYC?"
    print("User asks:", weather_query)
    weather_answer = await user.send_message(weather_query)
    print("Assistant replies:", weather_answer)

    # Clean up: gracefully shut down any open transports
    await assistant.shutdown()
    await user.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
