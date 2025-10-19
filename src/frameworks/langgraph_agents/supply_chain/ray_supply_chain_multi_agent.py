from __future__ import annotations
"""
supply_chain_logistics_agent_ray_per_session.py
LangGraph workflow for a multi-agent Supply Chain & Logistics Management system, revised to use Ray actors for decoupling with per-session isolation.
Handles inventory management, shipping operations, supplier relations, and warehouse optimization through specialized agents as Ray actors.
Each session (identified by operation_id) gets its own actor instances per specialist type, ensuring isolated state and sequential execution per session.
The supervisor determines the specialist and invokes the session-specific actor remotely via a SessionManager.
"""

import os
import json
import time
from typing import Annotated, Sequence, TypedDict, Optional, Dict

import ray
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

# Shared tool for all specialists
@tool
def send_logistics_response(operation_id: str = None, message: str = None) -> str:
    """Send logistics updates, recommendations, or status reports to stakeholders."""
    print(f"[TOOL] send_logistics_response → {message}")
    log_to_loki("tool.send_logistics_response", f"operation_id={operation_id}, message={message}")
    return "logistics_response_sent"

# Inventory & Warehouse Specialist Tools
@tool
def manage_inventory(sku: str = None, **kwargs) -> str:
    """Manage inventory levels, stock replenishment, audits, and optimization strategies."""
    print(f"[TOOL] manage_inventory(sku={sku}, kwargs={kwargs})")
    log_to_loki("tool.manage_inventory", f"sku={sku}")
    return "inventory_management_initiated"

@tool
def optimize_warehouse(operation_type: str = None, **kwargs) -> str:
    """Optimize warehouse operations, layout, capacity, and storage efficiency."""
    print(f"[TOOL] optimize_warehouse(operation_type={operation_type}, kwargs={kwargs})")
    log_to_loki("tool.optimize_warehouse", f"operation_type={operation_type}")
    return "warehouse_optimization_initiated"

@tool
def forecast_demand(season: str = None, **kwargs) -> str:
    """Analyze demand patterns, seasonal trends, and create forecasting models."""
    print(f"[TOOL] forecast_demand(season={season}, kwargs={kwargs})")
    log_to_loki("tool.forecast_demand", f"season={season}")
    return "demand_forecast_generated"

@tool
def manage_quality(supplier: str = None, **kwargs) -> str:
    """Manage quality control, defect tracking, and supplier quality standards."""
    print(f"[TOOL] manage_quality(supplier={supplier}, kwargs={kwargs})")
    log_to_loki("tool.manage_quality", f"supplier={supplier}")
    return "quality_management_initiated"

@tool
def scale_operations(scaling_type: str = None, **kwargs) -> str:
    """Scale operations for peak seasons, capacity planning, and workforce management."""
    print(f"[TOOL] scale_operations(scaling_type={scaling_type}, kwargs={kwargs})")
    log_to_loki("tool.scale_operations", f"scaling_type={scaling_type}")
    return "operations_scaled"

@tool
def optimize_costs(cost_type: str = None, **kwargs) -> str:
    """Analyze and optimize transportation, storage, and operational costs."""
    print(f"[TOOL] optimize_costs(cost_type={cost_type}, kwargs={kwargs})")
    log_to_loki("tool.optimize_costs", f"cost_type={cost_type}")
    return "cost_optimization_initiated"

INVENTORY_TOOLS = [manage_inventory, optimize_warehouse, forecast_demand, manage_quality, scale_operations, optimize_costs, send_logistics_response]

# Transportation & Logistics Specialist Tools
@tool
def track_shipments(origin: str = None, **kwargs) -> str:
    """Track shipment status, delays, and coordinate delivery logistics."""
    print(f"[TOOL] track_shipments(origin={origin}, kwargs={kwargs})")
    log_to_loki("tool.track_shipments", f"origin={origin}")
    return "shipment_tracking_updated"

@tool
def arrange_shipping(shipping_type: str = None, **kwargs) -> str:
    """Arrange shipping methods, expedited delivery, and multi-modal transportation."""
    print(f"[TOOL] arrange_shipping(shipping_type={shipping_type}, kwargs={kwargs})")
    log_to_loki("tool.arrange_shipping", f"shipping_type={shipping_type}")
    return "shipping_arranged"

@tool
def coordinate_operations(operation_type: str = None, **kwargs) -> str:
    """Coordinate complex operations like cross-docking, consolidation, and transfers."""
    print(f"[TOOL] coordinate_operations(operation_type={operation_type}, kwargs={kwargs})")
    log_to_loki("tool.coordinate_operations", f"operation_type={operation_type}")
    return "operations_coordinated"

@tool
def manage_special_handling(product_type: str = None, **kwargs) -> str:
    """Handle special requirements for hazmat, cold chain, and sensitive products."""
    print(f"[TOOL] manage_special_handling(product_type={product_type}, kwargs={kwargs})")
    log_to_loki("tool.manage_special_handling", f"product_type={product_type}")
    return "special_handling_managed"

@tool
def process_returns(returned_quantity: str = None, **kwargs) -> str:
    """Process returns, reverse logistics, and product disposition."""
    print(f"[TOOL] process_returns(returned_quantity={returned_quantity}, kwargs={kwargs})")
    log_to_loki("tool.process_returns", f"returned_quantity={returned_quantity}")
    return "returns_processed"

@tool
def optimize_delivery(delivery_type: str = None, **kwargs) -> str:
    """Optimize delivery routes, last-mile logistics, and sustainability initiatives."""
    print(f"[TOOL] optimize_delivery(delivery_type={delivery_type}, kwargs={kwargs})")
    log_to_loki("tool.optimize_delivery", f"delivery_type={delivery_type}")
    return "delivery_optimization_complete"

@tool
def manage_disruption(disruption_type: str = None, **kwargs) -> str:
    """Manage supply chain disruptions, contingency planning, and risk mitigation."""
    print(f"[TOOL] manage_disruption(disruption_type={disruption_type}, kwargs={kwargs})")
    log_to_loki("tool.manage_disruption", f"disruption_type={disruption_type}")
    return "disruption_managed"

TRANSPORTATION_TOOLS = [track_shipments, arrange_shipping, coordinate_operations, manage_special_handling, process_returns, optimize_delivery, manage_disruption, send_logistics_response]

# Supplier & Compliance Specialist Tools
@tool
def evaluate_suppliers(supplier_name: str = None, **kwargs) -> str:
    """Evaluate supplier performance, conduct audits, and manage supplier relationships."""
    print(f"[TOOL] evaluate_suppliers(supplier_name={supplier_name}, kwargs={kwargs})")
    log_to_loki("tool.evaluate_suppliers", f"supplier_name={supplier_name}")
    return "supplier_evaluation_complete"

@tool
def handle_compliance(compliance_type: str = None, **kwargs) -> str:
    """Manage regulatory compliance, customs, documentation, and certifications."""
    print(f"[TOOL] handle_compliance(compliance_type={compliance_type}, kwargs={kwargs})")
    log_to_loki("tool.handle_compliance", f"compliance_type={compliance_type}")
    return "compliance_handled"

SUPPLIER_TOOLS = [evaluate_suppliers, handle_compliance, send_logistics_response]

Traceloop.init(disable_batch=True, app_name="supply_chain_logistics_agent_ray_per_session")
llm = ChatOpenAI(model="gpt-4o", temperature=0.0, callbacks=[StreamingStdOutCallbackHandler()], verbose=True)

# Bind tools to specialized LLMs
inventory_llm = llm.bind_tools(INVENTORY_TOOLS)
transportation_llm = llm.bind_tools(TRANSPORTATION_TOOLS)
supplier_llm = llm.bind_tools(SUPPLIER_TOOLS)

class AgentState(TypedDict):
    operation: Optional[dict]  # Supply chain operation information
    messages: Annotated[Sequence[BaseMessage], operator.add]

# Ray Actor for Specialists (per-session isolation)
@ray.remote
class SpecialistActor:
    def __init__(self, name: str, specialist_llm, tools: list, system_prompt: str):
        self.name = name
        self.llm = specialist_llm
        self.tools = {t.name: t for t in tools}
        self.prompt = system_prompt
        self.internal_state = {}  # Isolated per-session state, e.g., for tracking within the session

    def process_task(self, operation: dict, messages: Sequence[BaseMessage]):
        if not operation:
            operation = {"operation_id": "UNKNOWN", "type": "general", "priority": "medium", "status": "active"}
        operation_json = json.dumps(operation, ensure_ascii=False)
        full_prompt = self.prompt + f"\n\nOPERATION: {operation_json}"
        
        full = [SystemMessage(content=full_prompt)] + messages

        first = self.llm.invoke(full)
        result_messages = [first]

        if hasattr(first, "tool_calls"):
            for tc in first.tool_calls:
                print(first)
                print(tc['name'])
                fn = self.tools.get(tc['name'])
                if fn:
                    out = fn.invoke(tc["args"])
                    result_messages.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))

            second = self.llm.invoke(full + result_messages)
            result_messages.append(second)

        # Update internal state (example: track processed steps within session)
        step_key = str(len(self.internal_state) + 1)  # Or use a more specific key
        self.internal_state[step_key] = {"status": "processed", "timestamp": time.time()}

        return {"messages": result_messages}

    def get_state(self):
        return self.internal_state  # Return entire session state

# Session Manager Actor: Tracks per-session specialist actors
@ray.remote
class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, ray.actor.ActorHandle]] = {}  # session_id -> {agent_name: actor}

    def get_or_create_actor(self, session_id: str, agent_name: str, llm, tools: list, prompt: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = {}
        if agent_name not in self.sessions[session_id]:
            actor = SpecialistActor.remote(agent_name, llm, tools, prompt)
            self.sessions[session_id][agent_name] = actor
        return self.sessions[session_id][agent_name]

    def get_session_state(self, session_id: str, agent_name: str):
        if session_id in self.sessions and agent_name in self.sessions[session_id]:
            actor = self.sessions[session_id][agent_name]
            return actor.get_state.remote()  # Returns future
        return None

# Supervisor: Determines specialist and invokes session-specific Ray actor remotely via manager
def supervisor_invoke(operation: dict, messages: Sequence[BaseMessage], manager: ray.actor.ActorHandle, llms: dict, tools_dict: dict, prompts: dict):
    session_id = operation.get("operation_id", "UNKNOWN")
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

    full = [SystemMessage(content=supervisor_prompt)] + messages
    response = llm.invoke(full)
    agent_name = response.content.strip().lower()
    
    if agent_name not in llms:
        raise ValueError(f"Unknown agent: {agent_name}")
    
    # Get or create session-specific actor
    actor_ref = manager.get_or_create_actor.remote(
        session_id, agent_name, llms[agent_name], tools_dict[agent_name], prompts[agent_name]
    )
    actor = ray.get(actor_ref)  # Get the actor handle
    
    # Invoke remotely
    result_ref = actor.process_task.remote(operation, messages)
    result = ray.get(result_ref)
    return result

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)  # Local cluster for demo; configure for distributed

    # Define prompts (as in original)
    inventory_prompt = (
        "You are an inventory and warehouse management specialist.\n"
        "When managing:\n"
        "  1) Analyze the inventory/warehouse challenge\n"
        "  2) Call the appropriate tool\n"
        "  3) Follow up with send_logistics_response\n"
        "Consider cost, efficiency, and scalability."
    )
    transportation_prompt = (
        "You are a transportation and logistics specialist.\n"
        "When managing:\n"
        "  1) Analyze the shipping/delivery challenge\n"
        "  2) Call the appropriate tool\n"
        "  3) Follow up with send_logistics_response\n"
        "Consider efficiency, sustainability, and risk mitigation."
    )
    supplier_prompt = (
        "You are a supplier relations and compliance specialist.\n"
        "When managing:\n"
        "  1) Analyze the supplier/compliance issue\n"
        "  2) Call the appropriate tool\n"
        "  3) Follow up with send_logistics_response\n"
        "Consider performance, regulations, and relationships."
    )

    prompts = {
        "inventory": inventory_prompt,
        "transportation": transportation_prompt,
        "supplier": supplier_prompt
    }

    llms = {
        "inventory": inventory_llm,
        "transportation": transportation_llm,
        "supplier": supplier_llm
    }

    tools_dict = {
        "inventory": INVENTORY_TOOLS,
        "transportation": TRANSPORTATION_TOOLS,
        "supplier": SUPPLIER_TOOLS
    }

    # Create session manager
    manager = SessionManager.remote()

    # Example invocation
    example = {"operation_id": "OP-12345", "type": "inventory_management", "priority": "high", "location": "Warehouse A"}
    convo = [HumanMessage(content="We're running critically low on SKU-12345. Current stock is 50 units but we have 200 units on backorder. What's our reorder strategy?")]

    result = supervisor_invoke(example, convo, manager, llms, tools_dict, prompts)
    for m in result["messages"]:
        print(f"{m.type}: {m.content}")

    # Optional: Query session-specific actor state
    state_ref = manager.get_session_state.remote("OP-12345", "inventory")
    if state_ref:
        state = ray.get(ray.get(state_ref))  # Resolve nested future
        print("Session actor state:", state)

    # Shutdown Ray
    ray.shutdown()
