from __future__ import annotations
"""
legal_document_review_agent.py
LangGraph workflow for a Legal Document Review & Case Management agent,
handling contract analysis, case research, client intake, compliance assessment, and litigation support.
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
def review_contract(contract_type: str = None, **kwargs) -> str:
    """Review and analyze contracts, agreements, and legal documents for risks, compliance, and negotiation points."""
    print(f"[TOOL] review_contract(contract_type={contract_type}, kwargs={kwargs})")
    log_to_loki("tool.review_contract", f"contract_type={contract_type}")
    return "contract_review_complete"

@tool
def research_case_law(case_type: str = None, jurisdiction: str = None, **kwargs) -> str:
    """Research legal precedents, statutes, regulations, and case law for litigation and compliance matters."""
    print(f"[TOOL] research_case_law(case_type={case_type}, jurisdiction={jurisdiction}, kwargs={kwargs})")
    log_to_loki("tool.research_case_law", f"case_type={case_type}, jurisdiction={jurisdiction}")
    return "legal_research_complete"

@tool
def client_intake(client_name: str = None, matter_type: str = None, **kwargs) -> str:
    """Process new client intake including conflict checks, case assessment, and matter setup."""
    print(f"[TOOL] client_intake(client_name={client_name}, matter_type={matter_type}, kwargs={kwargs})")
    log_to_loki("tool.client_intake", f"client_name={client_name}, matter_type={matter_type}")
    return "client_intake_complete"

@tool
def assess_compliance(regulations: list = None, client_industry: str = None, **kwargs) -> str:
    """Assess regulatory compliance requirements and analyze adherence to applicable laws and regulations."""
    print(f"[TOOL] assess_compliance(regulations={regulations}, client_industry={client_industry}, kwargs={kwargs})")
    log_to_loki("tool.assess_compliance", f"regulations={regulations}, client_industry={client_industry}")
    return "compliance_assessment_complete"

@tool
def manage_discovery(discovery_type: str = None, **kwargs) -> str:
    """Manage discovery processes including document review, production, privilege review, and e-discovery."""
    print(f"[TOOL] manage_discovery(discovery_type={discovery_type}, kwargs={kwargs})")
    log_to_loki("tool.manage_discovery", f"discovery_type={discovery_type}")
    return "discovery_management_initiated"

@tool
def calculate_damages(case_type: str = None, **kwargs) -> str:
    """Calculate damages, settlements, financial impacts, and economic analysis for litigation and transactions."""
    print(f"[TOOL] calculate_damages(case_type={case_type}, kwargs={kwargs})")
    log_to_loki("tool.calculate_damages", f"case_type={case_type}")
    return "damages_calculation_complete"

@tool
def track_deadlines(case_name: str = None, document_type: str = None, filing_deadline: str = None, **kwargs) -> str:
    """Track legal deadlines, court dates, filing requirements, and case management timelines."""
    print(f"[TOOL] track_deadlines(case_name={case_name}, document_type={document_type}, deadline={filing_deadline}, kwargs={kwargs})")
    log_to_loki("tool.track_deadlines", f"case_name={case_name}, document_type={document_type}")
    return "deadline_tracking_updated"

@tool
def send_legal_response(client_id: str = None, message: str = None) -> str:
    """Send legal advice, status updates, or responses to clients and stakeholders."""
    print(f"[TOOL] send_legal_response → {message}")
    log_to_loki("tool.send_legal_response", f"client_id={client_id}, message={message}")
    return "legal_response_sent"

TOOLS = [
    review_contract, research_case_law, client_intake, assess_compliance,
    manage_discovery, calculate_damages, track_deadlines, send_legal_response
]

Traceloop.init(disable_batch=True, app_name="legal_document_review_agent")
llm = ChatOpenAI(model="gpt-4o", temperature=0.0, callbacks=[StreamingStdOutCallbackHandler()],  
    verbose=True).bind_tools(TOOLS)

class AgentState(TypedDict):
    matter: Optional[dict]  # Legal matter/case information
    messages: Annotated[Sequence[BaseMessage], operator.add]

def call_model(state: AgentState):
    history = state["messages"]
    
    # Handle missing or incomplete matter data gracefully
    matter = state.get("matter", {})
    if not matter:
        matter = {"matter_id": "UNKNOWN", "client_id": "UNKNOWN", "matter_type": "general", "status": "active"}
    
    matter_json = json.dumps(matter, ensure_ascii=False)
    system_prompt = (
        "You are an experienced legal professional specializing in document review and case management.\n"
        "Your expertise covers:\n"
        "- Contract review and analysis (employment, commercial, licensing, etc.)\n"
        "- Legal research and case law analysis\n"
        "- Client intake and conflict checking\n"
        "- Regulatory compliance assessment\n"
        "- Discovery management and document review\n"
        "- Damages calculation and economic analysis\n"
        "- Deadline tracking and case management\n"
        "- Litigation support and trial preparation\n"
        "\n"
        "When assisting with legal matters:\n"
        "  1) Analyze the legal issue and call the appropriate legal tool\n"
        "  2) Follow up with send_legal_response to provide legal advice or status updates\n"
        "  3) Always consider ethical obligations, privilege, and confidentiality\n"
        "  4) Ensure compliance with applicable laws and professional standards\n"
        "\n"
        "Prioritize accuracy, thoroughness, and client protection in all legal work.\n\n"
        f"MATTER: {matter_json}"
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
    example = {"matter_id": "MTR-12345", "client_id": "CLI-789", "matter_type": "contract_review", "attorney": "J. Smith"}
    convo = [HumanMessage(content="I need you to review this employment agreement for red flags. The client is concerned about the non-compete clause and termination provisions.")]
    result = graph.invoke({"matter": example, "messages": convo})
    for m in result["messages"]:
        print(f"{m.type}: {m.content}") 