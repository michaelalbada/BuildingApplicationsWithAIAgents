from __future__ import annotations
"""
it_helpdesk_agent.py
LangGraph workflow for an IT Help Desk & System Administration agent,
handling user access management, system troubleshooting, security incidents, and infrastructure support.
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
def provision_user_access(user_id: str = None, action: str = "grant_access", **kwargs) -> str:
    """Manage user access including account creation, password resets, permissions, and account termination."""
    print(f"[TOOL] provision_user_access(user_id={user_id}, action={action}, kwargs={kwargs})")
    log_to_loki("tool.provision_user_access", f"user_id={user_id}, action={action}")
    return "user_access_updated"

@tool
def troubleshoot_network(issue: str = None, location: str = None, **kwargs) -> str:
    """Diagnose and resolve network connectivity issues including WiFi, VPN, internet, and firewall problems."""
    print(f"[TOOL] troubleshoot_network(issue={issue}, location={location}, kwargs={kwargs})")
    log_to_loki("tool.troubleshoot_network", f"issue={issue}, location={location}")
    return "network_issue_diagnosed"

@tool
def diagnose_system_issue(system: str = None, issue: str = None, service: str = None, **kwargs) -> str:
    """Diagnose server, database, application, and system performance issues."""
    print(f"[TOOL] diagnose_system_issue(system={system}, issue={issue}, service={service}, kwargs={kwargs})")
    log_to_loki("tool.diagnose_system_issue", f"system={system}, issue={issue}")
    return "system_diagnosis_complete"

@tool
def deploy_software(software: str = None, action: str = "install", **kwargs) -> str:
    """Handle software installation, updates, license management, and deployment."""
    print(f"[TOOL] deploy_software(software={software}, action={action}, kwargs={kwargs})")
    log_to_loki("tool.deploy_software", f"software={software}, action={action}")
    return "software_deployment_initiated"

@tool
def contain_security_incident(incident_type: str = None, affected_system: str = None, **kwargs) -> str:
    """Respond to security incidents including malware, ransomware, phishing, and breaches."""
    print(f"[TOOL] contain_security_incident(incident_type={incident_type}, affected_system={affected_system}, kwargs={kwargs})")
    log_to_loki("tool.contain_security_incident", f"incident_type={incident_type}, affected_system={affected_system}")
    return "security_incident_contained"

@tool
def troubleshoot_hardware(device: str = None, location: str = None, issue: str = None, **kwargs) -> str:
    """Diagnose and resolve hardware issues with printers, projectors, computers, and peripherals."""
    print(f"[TOOL] troubleshoot_hardware(device={device}, location={location}, issue={issue}, kwargs={kwargs})")
    log_to_loki("tool.troubleshoot_hardware", f"device={device}, issue={issue}")
    return "hardware_troubleshooting_initiated"

@tool
def assign_roles(user_id: str = None, new_role: str = None, **kwargs) -> str:
    """Manage user roles, permissions, and security policy enforcement."""
    print(f"[TOOL] assign_roles(user_id={user_id}, new_role={new_role}, kwargs={kwargs})")
    log_to_loki("tool.assign_roles", f"user_id={user_id}, new_role={new_role}")
    return "role_assignment_complete"

@tool
def escalate_incident(incident_id: str = None, escalate_to: str = None, **kwargs) -> str:
    """Escalate complex issues to higher-level support teams or specialists."""
    print(f"[TOOL] escalate_incident(incident_id={incident_id}, escalate_to={escalate_to}, kwargs={kwargs})")
    log_to_loki("tool.escalate_incident", f"incident_id={incident_id}, escalate_to={escalate_to}")
    return "incident_escalated"

@tool
def apply_patches(target_systems: str = None, patch_type: str = None, **kwargs) -> str:
    """Apply system patches, updates, and security fixes to infrastructure."""
    print(f"[TOOL] apply_patches(target_systems={target_systems}, patch_type={patch_type}, kwargs={kwargs})")
    log_to_loki("tool.apply_patches", f"target_systems={target_systems}, patch_type={patch_type}")
    return "patch_deployment_scheduled"

@tool
def send_user_response(user_id: str = None, message: str = None) -> str:
    """Send a response or status update to the user."""
    print(f"[TOOL] send_user_response → {message}")
    log_to_loki("tool.send_user_response", f"user_id={user_id}, message={message}")
    return "response_sent"

TOOLS = [
    provision_user_access, troubleshoot_network, diagnose_system_issue, deploy_software,
    contain_security_incident, troubleshoot_hardware, assign_roles, escalate_incident,
    apply_patches, send_user_response
]

Traceloop.init(disable_batch=True, app_name="it_helpdesk_agent")
llm = ChatOpenAI(model="gpt-4o", temperature=0.0, callbacks=[StreamingStdOutCallbackHandler()],  
    verbose=True).bind_tools(TOOLS)

class AgentState(TypedDict):
    ticket: Optional[dict]  # IT support ticket information
    messages: Annotated[Sequence[BaseMessage], operator.add]

def call_model(state: AgentState):
    history = state["messages"]
    
    # Handle missing or incomplete ticket data gracefully
    ticket = state.get("ticket", {})
    if not ticket:
        ticket = {"ticket_id": "UNKNOWN", "user_id": "UNKNOWN", "priority": "medium", "status": "open"}
    
    ticket_json = json.dumps(ticket, ensure_ascii=False)
    system_prompt = (
        "You are an experienced IT Help Desk technician and system administrator.\n"
        "Your expertise covers:\n"
        "- User access management (accounts, passwords, permissions)\n"
        "- Network troubleshooting (WiFi, VPN, connectivity issues)\n"
        "- System diagnostics (servers, databases, applications)\n"
        "- Software deployment and license management\n"
        "- Security incident response and containment\n"
        "- Hardware troubleshooting and maintenance\n"
        "- Infrastructure monitoring and patch management\n"
        "\n"
        "When helping users:\n"
        "  1) Analyze the technical issue and call the appropriate diagnostic/resolution tool\n"
        "  2) Follow up with send_user_response to explain what actions were taken\n"
        "  3) Escalate complex issues when they exceed your support level\n"
        "\n"
        "Always prioritize security, business continuity, and user productivity.\n\n"
        f"TICKET: {ticket_json}"
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
    example = {"ticket_id": "INC-12345", "user_id": "jsmith", "priority": "high", "category": "access_issue"}
    convo = [HumanMessage(content="Hi, I'm John Smith, the new marketing manager. I need access to the shared marketing drive and Adobe Creative Suite.")]
    result = graph.invoke({"ticket": example, "messages": convo})
    for m in result["messages"]:
        print(f"{m.type}: {m.content}") 