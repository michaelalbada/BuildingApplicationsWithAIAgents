import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def should_continue(state):
    return "continue" if state["messages"][-1].tool_calls else "end"


def call_model(state, config):
    return {"messages": [llm_with_tools.invoke(state["messages"], config=config)]}


def _invoke_tool(tool_call):
    tool = {tool.name: tool for tool in tools}[tool_call["name"]]
    return ToolMessage(tool.invoke(tool_call["args"]), tool_call_id=tool_call["id"])


tool_executor = RunnableLambda(_invoke_tool)


def call_tools(state):
    last_message = state["messages"][-1]
    return {"messages": tool_executor.batch(last_message.tool_calls)}


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tools)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)
workflow.add_edge("action", "agent")
graph = workflow.compile()

result = graph.invoke(
    {
        "messages": [
            HumanMessage(
                "what's 3 plus 5 raised to the 2.743. also what's 17.24 - 918.1241"
            )
        ]
    }
)

print("Conversation Messages:")
for idx, message in enumerate(result["messages"], start=1):
    if isinstance(message, HumanMessage):
        print(f"{idx}. Human: {message.content}")
    elif isinstance(message, AIMessage):
        print(f"{idx}. AI: {message.content}")
    elif isinstance(message, ToolMessage):
        print(f"{idx}. Tool ({message.tool_call_id}): {message.content}")
    else:
        print(f"{idx}. Unknown Message Type: {message}")