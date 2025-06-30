from __future__ import annotations
"""
soc_analyst_agent.py
LangGraph workflow for a Security Operations Center (SOC) Analyst agent,
handling threat investigation, incident response, log analysis, and security monitoring.
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
def lookup_threat_intel(indicator: str, type: str, **kwargs) -> str:
    """Look up threat intelligence for IP addresses, file hashes, URLs, and domains."""
    print(f"[TOOL] lookup_threat_intel(indicator={indicator}, type={type}, kwargs={kwargs})")
    log_to_loki("tool.lookup_threat_intel", f"indicator={indicator}, type={type}")
    return "threat_intel_retrieved"

@tool
def query_logs(query: str, log_index: str, **kwargs) -> str:
    """Search and analyze security logs across authentication, endpoint, network, firewall, and DNS systems."""
    print(f"[TOOL] query_logs(query={query}, log_index={log_index}, kwargs={kwargs})")
    log_to_loki("tool.query_logs", f"query={query}, log_index={log_index}")
    return "log_query_executed"

@tool
def triage_incident(incident_id: str, decision: str, reason: str, **kwargs) -> str:
    """Classify security incidents as true positive, false positive, or escalate for further investigation."""
    print(f"[TOOL] triage_incident(incident_id={incident_id}, decision={decision}, reason={reason}, kwargs={kwargs})")
    log_to_loki("tool.triage_incident", f"incident_id={incident_id}, decision={decision}")
    return "incident_triaged"

@tool
def isolate_host(host_id: str, reason: str, **kwargs) -> str:
    """Isolate compromised hosts to prevent lateral movement and contain security incidents."""
    print(f"[TOOL] isolate_host(host_id={host_id}, reason={reason}, kwargs={kwargs})")
    log_to_loki("tool.isolate_host", f"host_id={host_id}, reason={reason}")
    return "host_isolated"

@tool
def send_analyst_response(incident_id: str = None, message: str = None) -> str:
    """Send security analysis, incident updates, or recommendations to stakeholders."""
    print(f"[TOOL] send_analyst_response → {message}")
    log_to_loki("tool.send_analyst_response", f"incident_id={incident_id}, message={message}")
    return "analyst_response_sent"

TOOLS = [
    lookup_threat_intel, query_logs, triage_incident, isolate_host, send_analyst_response
]

Traceloop.init(disable_batch=True, app_name="soc_analyst_agent")
llm = ChatOpenAI(model="gpt-4o", temperature=0.0, callbacks=[StreamingStdOutCallbackHandler()],  
    verbose=True).bind_tools(TOOLS)

class AgentState(TypedDict):
    incident: Optional[dict]  # Security incident information
    messages: Annotated[Sequence[BaseMessage], operator.add]

def call_model(state: AgentState):
    history = state["messages"]
    
    # Handle missing or incomplete incident data gracefully
    incident = state.get("incident", {})
    if not incident:
        incident = {"incident_id": "UNKNOWN", "severity": "medium", "status": "investigating", "analyst": "SOC_EVAL"}
    
    incident_json = json.dumps(incident, ensure_ascii=False)
    system_prompt = (
        "You are an experienced Security Operations Center (SOC) analyst specializing in cybersecurity incident response.\n"
        "Your expertise covers:\n"
        "- Threat intelligence analysis and IOC research\n"
        "- Security log analysis and correlation across multiple systems\n"
        "- Incident triage and classification (true positive/false positive)\n"
        "- Malware analysis and threat hunting\n"
        "- Network security monitoring and anomaly detection\n"
        "- Incident containment and response coordination\n"
        "- SIEM/SOAR platform operations\n"
        "\n"
        "Your investigation methodology:\n"
        "  1) Analyze security alerts and gather initial indicators\n"
        "  2) Use lookup_threat_intel to research IPs, hashes, URLs, and domains\n"
        "  3) Use query_logs to search relevant log sources for evidence\n"
        "  4) Use triage_incident to classify findings as true/false positives\n"
        "  5) Use isolate_host when containment is needed to prevent spread\n"
        "  6) Follow up with send_analyst_response to document findings\n"
        "\n"
        "Always prioritize rapid threat containment and accurate incident classification.\n\n"
        f"INCIDENT: {incident_json}"
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
    example = {"incident_id": "INC-12345", "severity": "high", "type": "suspicious_login", "analyst": "J.Smith"}
    convo = [HumanMessage(content='We have an alert: "Suspicious login attempt from IP 203.0.113.45 to admin account." What should we do?')]
    result = graph.invoke({"incident": example, "messages": convo})
    for m in result["messages"]:
        print(f"{m.type}: {m.content}") 