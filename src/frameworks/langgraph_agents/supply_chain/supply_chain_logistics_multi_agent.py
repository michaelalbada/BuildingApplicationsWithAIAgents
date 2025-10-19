from __future__ import annotations
"""
supply_chain_logistics_agent.py
LangGraph workflow for a multi-agent Supply Chain & Logistics Management system.
Handles inventory management, shipping operations, supplier relations, and warehouse optimization through specialized agents coordinated by a supervisor.
"""
import os
import json
import operator
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

Traceloop.init(disable_batch=True, app_name="supply_chain_logistics_agent")
llm = ChatOpenAI(model="gpt-4o", temperature=0.0, callbacks=[StreamingStdOutCallbackHandler()], verbose=True)

# Bind tools to specialized LLMs
inventory_llm = llm.bind_tools(INVENTORY_TOOLS)
transportation_llm = llm.bind_tools(TRANSPORTATION_TOOLS)
supplier_llm = llm.bind_tools(SUPPLIER_TOOLS)

class AgentState(TypedDict):
    operation: Optional[dict]  # Supply chain operation information
    messages: Annotated[Sequence[BaseMessage], operator.add]

# Supervisor (Manager) Node: Routes to the appropriate specialist
def supervisor_node(state: AgentState):
    history = state["messages"]
    operation = state.get("operation", {})
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

    full = [SystemMessage(content=supervisor_prompt)] + history
    response = llm.invoke(full)
    return {"messages": [response]}

# Specialist Node Template
def specialist_node(state: AgentState, specialist_llm, system_prompt: str):
    history = state["messages"]
    operation = state.get("operation", {})
    if not operation:
        operation = {"operation_id": "UNKNOWN", "type": "general", "priority": "medium", "status": "active"}
    operation_json = json.dumps(operation, ensure_ascii=False)
    full_prompt = system_prompt + f"\n\nOPERATION: {operation_json}"
    
    full = [SystemMessage(content=full_prompt)] + history

    first: ToolMessage | BaseMessage = specialist_llm.invoke(full)
    messages = [first]

    if getattr(first, "tool_calls", None):
        for tc in first.tool_calls:
            print(first)
            print(tc['name'])
            # Find the tool (assuming tools are unique by name across all)
            all_tools = INVENTORY_TOOLS + TRANSPORTATION_TOOLS + SUPPLIER_TOOLS
            fn = next(t for t in all_tools if t.name == tc['name'])
            out = fn.invoke(tc["args"])
            messages.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))

        second = specialist_llm.invoke(full + messages)
        messages.append(second)

    return {"messages": messages}

# Inventory Specialist Node
def inventory_node(state: AgentState):
    inventory_prompt = (
        "You are an inventory and warehouse management specialist.\n"
        "When managing:\n"
        "  1) Analyze the inventory/warehouse challenge\n"
        "  2) Call the appropriate tool\n"
        "  3) Follow up with send_logistics_response\n"
        "Consider cost, efficiency, and scalability."
    )
    return specialist_node(state, inventory_llm, inventory_prompt)

# Transportation Specialist Node
def transportation_node(state: AgentState):
    transportation_prompt = (
        "You are a transportation and logistics specialist.\n"
        "When managing:\n"
        "  1) Analyze the shipping/delivery challenge\n"
        "  2) Call the appropriate tool\n"
        "  3) Follow up with send_logistics_response\n"
        "Consider efficiency, sustainability, and risk mitigation."
    )
    return specialist_node(state, transportation_llm, transportation_prompt)

# Supplier Specialist Node
def supplier_node(state: AgentState):
    supplier_prompt = (
        "You are a supplier relations and compliance specialist.\n"
        "When managing:\n"
        "  1) Analyze the supplier/compliance issue\n"
        "  2) Call the appropriate tool\n"
        "  3) Follow up with send_logistics_response\n"
        "Consider performance, regulations, and relationships."
    )
    return specialist_node(state, supplier_llm, supplier_prompt)

# Routing function for conditional edges
def route_to_specialist(state: AgentState):
    last_message = state["messages"][-1]
    agent_name = last_message.content.strip().lower()
    if agent_name == "inventory":
        return "inventory"
    elif agent_name == "transportation":
        return "transportation"
    elif agent_name == "supplier":
        return "supplier"
    else:
        # Fallback if no match
        return END

def construct_graph():
    g = StateGraph(AgentState)
    g.add_node("supervisor", supervisor_node)
    g.add_node("inventory", inventory_node)
    g.add_node("transportation", transportation_node)
    g.add_node("supplier", supplier_node)
    
    g.set_entry_point("supervisor")
    g.add_conditional_edges("supervisor", route_to_specialist, {"inventory": "inventory", "transportation": "transportation", "supplier": "supplier"})
    
    g.add_edge("inventory", END)
    g.add_edge("transportation", END)
    g.add_edge("supplier", END)
    
    return g.compile()

graph = construct_graph()

# Actor Node: Generates candidate plans
def actor_node(state: AgentState):
    history = state["messages"]
    actor_prompt = "Generate 3 candidate supply chain plans as JSON list: [{'plan': 'description', 'tools': [...]}]"
    response = llm.invoke([SystemMessage(content=actor_prompt)] + history)
    state["candidates"] = json.loads(response.content)
    return state

# Critic Node: Evaluates and selects/iterates
def critic_node(state: AgentState):
    candidates = state["candidates"]
    history = state["messages"]
    critic_prompt = f"Score candidates {candidates} on scale 1-10 for feasibility, cost, risk. Select best if >8, else request regeneration."
    response = llm.invoke([SystemMessage(content=critic_prompt)] + history)
    eval = json.loads(response.content)
    if eval['best_score'] > 8:
        winning_plan = eval['selected']
        # Execute winning plan's tools (similar to specialist execution)
        messages = []
        for tool_info in winning_plan['tools']:
            tc = {'name': tool_info['tool'], 'args': tool_info['args'], 'id': 'dummy'}
            fn = next(t for t in all_tools if t.name == tc['name'])
            out = fn.invoke(tc["args"])
            messages.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
        # Send response
        send_fn.invoke({"message": winning_plan['plan']})
        return {"messages": history + messages}
    else:
        # Iterate: Add feedback to history for actor
        return {"messages": history + [AIMessage(content="Regenerate with improvements: " + eval['feedback'])]}

def construct_actor_critic_graph():
    g = StateGraph(AgentState)
    g.add_node("actor", actor_node)
    g.add_node("critic", critic_node)
    
    g.set_entry_point("actor")
    g.add_edge("actor", "critic")
    # Loop back if not approved (conditional)
    g.add_conditional_edges("critic", lambda s: "actor" if "regenerate" in s["messages"][-1].content.lower() else END)
    
    return g.compile()

if __name__ == "__main__":
    example = {"operation_id": "OP-12345", "type": "inventory_management", "priority": "high", "location": "Warehouse A"}
    convo = [HumanMessage(content="We're running critically low on SKU-12345. Current stock is 50 units but we have 200 units on backorder. What's our reorder strategy?")]
    result = graph.invoke({"operation": example, "messages": convo})
    for m in result["messages"]:
        print(f"{m.type}: {m.content}")
