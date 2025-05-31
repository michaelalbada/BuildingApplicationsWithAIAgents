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

from langchain.tools import Tool
from langchain_mcp_adapters.client import MultiServerMCPClient

from openai_autogen import (
    AIConfig,
    AssistantAgent,
    UserProxyAgent,
)

MCP_CONFIG = {
    "math": {
        "command": "python3",
        "args": ["src/common/mcp/MCP_weather_server.py"],
        "transport": "stdio",
    },
    "weather": {
        "url": "http://localhost:8000/mcp",
        "transport": "streamable_http",
    },
}

async def get_mcp_tools() -> List[Tool]:
    """
    Instantiate a MultiServerMCPClient, fetch its Tools,
    and return them as a list of LangChain Tool objects.
    """
    client = MultiServerMCPClient(MCP_CONFIG)
    return await client.get_tools()


def wrap_tool_for_autogen(lc_tool: Tool):
    """
    Given a LangChain Tool, produce a callable async function that Autogen can call.
    Autogen expects each tool to be an async function taking a single string
    (the user’s input) and returning a string (the tool’s output).
    """
    async def _fn(user_input: str) -> str:
        return await lc_tool.arun(user_input)

    return lc_tool.name, _fn, lc_tool.description



async def main():
    lc_tools = await get_mcp_tools()  # e.g. [Tool(name="math", ...), Tool(name="weather", ...)]

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

    ai_config = AIConfig(
        model="gpt-4o", 
        temperature=0.0, 
        openai_api_key=None
    )

    assistant = AssistantAgent(
        name="mcp_assistant",
        ai_config=ai_config,
        tools=autogen_tools
    )

    user = UserProxyAgent(assistant)

    math_query = "What's (3 + 5) * 12?"
    print("User asks:", math_query)
    math_answer = await user.send_message(math_query)
    print("Assistant replies:", math_answer, "\n")

    weather_query = "What is the weather in NYC?"
    print("User asks:", weather_query)
    weather_answer = await user.send_message(weather_query)
    print("Assistant replies:", weather_answer)

    await assistant.shutdown()
    await user.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
