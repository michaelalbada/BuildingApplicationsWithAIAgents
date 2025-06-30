from __future__ import annotations
"""
healthcare_patient_intake_agent.py
LangGraph workflow for a healthcare patient intake and triage agent,
handling patient registration, symptom assessment, appointment scheduling, and medical history management.
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
def assess_symptoms(patient_id: str, symptoms: list = None, urgency: str = "routine", **kwargs) -> str:
    """Assess patient symptoms and determine urgency level for triage."""
    print(f"[TOOL] assess_symptoms(patient_id={patient_id}, symptoms={symptoms}, urgency={urgency})")
    log_to_loki("tool.assess_symptoms", f"patient_id={patient_id}, urgency={urgency}")
    return "symptoms_assessed"

@tool
def register_patient(name: str, date_of_birth: str = None, insurance_provider: str = None, **kwargs) -> str:
    """Register a new patient in the healthcare system."""
    print(f"[TOOL] register_patient(name={name}, dob={date_of_birth}, insurance={insurance_provider})")
    log_to_loki("tool.register_patient", f"name={name}, insurance={insurance_provider}")
    return "patient_registered"

@tool
def schedule_appointment(patient_id: str, appointment_type: str, provider: str = None, **kwargs) -> str:
    """Schedule appointments for patients with healthcare providers."""
    print(f"[TOOL] schedule_appointment(patient_id={patient_id}, type={appointment_type}, provider={provider})")
    log_to_loki("tool.schedule_appointment", f"patient_id={patient_id}, type={appointment_type}")
    return "appointment_scheduled"

@tool
def verify_insurance(patient_id: str, insurance_provider: str, policy_number: str = None, **kwargs) -> str:
    """Verify patient insurance coverage and eligibility."""
    print(f"[TOOL] verify_insurance(patient_id={patient_id}, provider={insurance_provider}, policy={policy_number})")
    log_to_loki("tool.verify_insurance", f"patient_id={patient_id}, provider={insurance_provider}")
    return "insurance_verified"

@tool
def update_medical_history(patient_id: str, **kwargs) -> str:
    """Update patient medical history including medications, allergies, and family history."""
    print(f"[TOOL] update_medical_history(patient_id={patient_id}, updates={kwargs})")
    log_to_loki("tool.update_medical_history", f"patient_id={patient_id}")
    return "medical_history_updated"

@tool
def refer_specialist(patient_id: str, specialty: str, reason: str = None, **kwargs) -> str:
    """Refer patient to specialist for specialized care."""
    print(f"[TOOL] refer_specialist(patient_id={patient_id}, specialty={specialty}, reason={reason})")
    log_to_loki("tool.refer_specialist", f"patient_id={patient_id}, specialty={specialty}")
    return "referral_created"

@tool
def prescribe_medication(patient_id: str, medication: str, dosage: str = None, **kwargs) -> str:
    """Prescribe or refill medications for patients."""
    print(f"[TOOL] prescribe_medication(patient_id={patient_id}, medication={medication}, dosage={dosage})")
    log_to_loki("tool.prescribe_medication", f"patient_id={patient_id}, medication={medication}")
    return "prescription_sent"

@tool
def send_patient_message(patient_id: str, message: str) -> str:
    """Send a message or response to the patient."""
    print(f"[TOOL] send_patient_message → {message}")
    log_to_loki("tool.send_patient_message", f"patient_id={patient_id}, message={message}")
    return "message_sent"

TOOLS = [
    assess_symptoms, register_patient, schedule_appointment, verify_insurance,
    update_medical_history, refer_specialist, prescribe_medication, send_patient_message
]

Traceloop.init(disable_batch=True, app_name="healthcare_patient_intake_agent")
llm = ChatOpenAI(model="gpt-4o", temperature=0.0, callbacks=[StreamingStdOutCallbackHandler()],  
    verbose=True).bind_tools(TOOLS)

class AgentState(TypedDict):
    patient: Optional[dict]  # Patient information
    messages: Annotated[Sequence[BaseMessage], operator.add]

def call_model(state: AgentState):
    history = state["messages"]
    
    # Handle missing or incomplete patient data gracefully
    patient = state.get("patient", {})
    if not patient:
        patient = {"patient_id": "UNKNOWN", "name": "Unknown Patient", "status": "active"}
    
    patient_json = json.dumps(patient, ensure_ascii=False)
    system_prompt = (
        "You are a professional healthcare patient intake and triage specialist.\n"
        "Your role is to help patients with:\n"
        "- Registering new patients and verifying insurance\n"
        "- Assessing symptoms and determining urgency for proper triage\n"
        "- Scheduling appointments with appropriate providers\n"
        "- Managing medical history, medications, and allergies\n"
        "- Coordinating specialist referrals when needed\n"
        "When assisting patients:\n"
        "  1) Call the appropriate healthcare tool based on their needs\n"
        "  2) Follow up with send_patient_message to confirm actions taken\n"
        "Always prioritize patient safety and ensure urgent cases are handled immediately.\n\n"
        f"PATIENT: {patient_json}"
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
    example = {"patient_id": "P12345", "name": "John Doe", "age": 45, "insurance": "Blue Cross"}
    convo = [HumanMessage(content="Hi, I need to schedule an appointment. I'm having chest pain that started this morning.")]
    result = graph.invoke({"patient": example, "messages": convo})
    for m in result["messages"]:
        print(f"{m.type}: {m.content}") 