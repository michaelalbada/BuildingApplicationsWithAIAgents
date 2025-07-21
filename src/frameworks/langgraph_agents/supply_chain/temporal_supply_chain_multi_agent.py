from __future__ import annotations
"""
supply_chain_logistics_agent_temporal.py
LangGraph workflow for a multi-agent Supply Chain & Logistics Management system, revised to use Temporal for durable orchestration.
Handles inventory management, shipping operations, supplier relations, and warehouse optimization through specialized agents orchestrated via Temporal workflows.
The workflow sequences agent steps with retries, persistent state, and failure recovery, ideal for long-running supply chain processes.
"""

import os
import json
from datetime import timedelta
from typing import Annotated, Sequence, TypedDict, Optional, Dict, Any

from temporalio import workflow, activity
from temporalio.common import RetryPolicy

from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.tool import ToolMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.tools import tool
from temporalio.client import Client
from temporalio.worker import Worker

from traceloop.sdk import Traceloop
from src.common.observability.loki_logger import log_to_loki

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4317"
os.environ["OTEL_EXPORTER_OTLP_INSECURE"] = "true"

# Shared tool for all specialists
@tool
def send_logistics_response(operation_id: str = None, message: str = None) -> str:
    """Send logistics updates, recommendations, or status reports to stakeholders."""
    print(f"[TOOL] send_logistics_response → {message}")
    log_to_loki("tool.send_logistics_response", f"operation_id={operation_id}, message={message}")
    return "logistics_response_sent"

# Inventory & Warehouse Specialist Tools (same as original)
@tool
def manage_inventory(sku: str = None, **kwargs) -> str:
    """Manage inventory levels, stock replenishment, audits, and optimization strategies."""
    print(f"[TOOL] manage_inventory(sku={sku}, kwargs={kwargs})")
    log_to_loki("tool.manage_inventory", f"sku={sku}")
    return "inventory_management_initiated"

# ... (omit other inventory tools for brevity; include all from original)

INVENTORY_TOOLS = [manage_inventory, ... , send_logistics_response]  # Full list

# Transportation Tools (omit for brevity)
TRANSPORTATION_TOOLS = [...]  # Full list

# Supplier Tools (omit for brevity)
SUPPLIER_TOOLS = [...]  # Full list

Traceloop.init(disable_batch=True, app_name="supply_chain_logistics_agent_temporal")
llm = ChatOpenAI(model="gpt-4o", temperature=0.0, callbacks=[StreamingStdOutCallbackHandler()], verbose=True)

# Bind tools to specialized LLMs
inventory_llm = llm.bind_tools(INVENTORY_TOOLS)
transportation_llm = llm.bind_tools(TRANSPORTATION_TOOLS)
supplier_llm = llm.bind_tools(SUPPLIER_TOOLS)

class AgentState(TypedDict):
    operation: Optional[dict]  # Supply chain operation information
    messages: Annotated[Sequence[BaseMessage], "add"]

# Temporal Activities (wrap specialist logic)
@activity.defn
async def supervisor_activity(operation: Dict[str, Any], messages: list) -> Dict[str, Any]:
    """Activity to determine specialist via supervisor."""
    operation_json = json.dumps(operation, ensure_ascii=False)
    
    supervisor_prompt = (
        "You are a supervisor coordinating a team of supply chain specialists.\n"
        "Team members:\n"
        "- inventory: Handles inventory levels, forecasting, quality, warehouse optimization, scaling, and costs.\n"
        "- transportation: Handles shipping tracking, arrangements, operations coordination, special handling, returns, delivery optimization, and disruptions.\n"
        "- supplier: Handles supplier evaluation and compliance.\n"
        "\n"
        "Based on the user query, select ONE team member to handle it.\n"
        "Output ONLY the selected member's name (inventory, transportation, or supplier), nothing else.\n\n"
        f"OPERATION: {operation_json}"
    )

    full = [SystemMessage(content=supervisor_prompt)] + [HumanMessage(**m) if isinstance(m, dict) else m for m in messages]
    response = llm.invoke(full)
    agent_name = response.content.strip().lower()
    return {"agent_name": agent_name, "messages": [response.dict()]}

@activity.defn
async def specialist_activity(agent_name: str, operation: Dict[str, Any], messages: list, prompts: Dict[str, str], llms: Dict[str, Any], tools_dict: Dict[str, list]) -> Dict[str, Any]:
    """Activity for specialist processing (inventory, transportation, supplier)."""
    if agent_name not in prompts:
        raise ValueError(f"Unknown agent: {agent_name}")
    
    specialist_llm = llms[agent_name]
    tools = {t.name: t for t in tools_dict[agent_name]}
    system_prompt = prompts[agent_name]
    
    operation_json = json.dumps(operation, ensure_ascii=False)
    full_prompt = system_prompt + f"\n\nOPERATION: {operation_json}"
    
    full = [SystemMessage(content=full_prompt)] + [HumanMessage(**m) if isinstance(m, dict) else m for m in messages]

    first = specialist_llm.invoke(full)
    result_messages = [first.dict()]

    if hasattr(first, "tool_calls"):
        for tc in first.tool_calls:
            fn = tools.get(tc['name'])
            if fn:
                out = fn.invoke(tc["args"])
                result_messages.append(ToolMessage(content=str(out), tool_call_id=tc["id"]).dict())

        second = specialist_llm.invoke(full + [ToolMessage(**msg) if isinstance(msg, dict) else msg for msg in result_messages])
        result_messages.append(second.dict())

    return {"messages": result_messages}

# Temporal Workflow
@workflow.defn(name="SupplyChainWorkflow")
class SupplyChainWorkflow:
    async def run(self, operation: Dict[str, Any], initial_messages: list, prompts: Dict[str, str], llms: Dict[str, Any], tools_dict: Dict[str, list]) -> Dict[str, Any]:
        # Step 1: Supervisor to route
        supervisor_result = await workflow.execute_activity(
            supervisor_activity,
            args=[operation, initial_messages],
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(maximum_attempts=3)
        )
        agent_name = supervisor_result["agent_name"]
        updated_messages = initial_messages + supervisor_result["messages"]
        
        # Step 2: Specialist processing
        specialist_result = await workflow.execute_activity(
            specialist_activity,
            args=[agent_name, operation, updated_messages, prompts, llms, tools_dict],
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(maximum_attempts=3)
        )
        
        # Compile results (extend for multi-step if needed)
        final_messages = updated_messages + specialist_result["messages"]
        return {
            "agent_name": agent_name,
            "final_messages": final_messages,
            "operation": operation
        }

# Prompts (as in original)
inventory_prompt = "You are an inventory and warehouse management specialist...\n"  # Full prompt
transportation_prompt = "You are a transportation and logistics specialist...\n"  # Full prompt
supplier_prompt = "You are a supplier relations and compliance specialist...\n"  # Full prompt

prompts = {
    "inventory": inventory_prompt,
    "transportation": transportation_prompt,
    "supplier": supplier_prompt
}

llms_dict = {
    "inventory": inventory_llm,
    "transportation": transportation_llm,
    "supplier": supplier_llm
}

tools_dict = {
    "inventory": INVENTORY_TOOLS,
    "transportation": TRANSPORTATION_TOOLS,
    "supplier": SUPPLIER_TOOLS
}

async def main():
    client = await Client.connect("localhost:7233")
    # Start worker
    async with Worker(client, task_queue="supply-chain-queue", workflows=[SupplyChainWorkflow], activities=[supervisor_activity, specialist_activity]):
        # Example execution
        example_operation = {"operation_id": "OP-12345", "type": "inventory_management", "priority": "high", "location": "Warehouse A"}
        example_messages = [{"content": "We're running critically low on SKU-12345. Current stock is 50 units but we have 200 units on backorder. What's our reorder strategy?", "type": "human"}]

        result = await client.execute_workflow(
            SupplyChainWorkflow.run,
            {"operation": example_operation, "initial_messages": example_messages, "prompts": prompts, "llms": llms_dict, "tools_dict": tools_dict},
            id="supply-chain-workflow",
            task_queue="supply-chain-queue"
        )
        print("Workflow result:")
        for m in result["final_messages"]:
            print(m)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
