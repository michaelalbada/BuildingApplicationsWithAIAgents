#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import uvicorn

# ─── MCP Schemas ────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class Payload(BaseModel):
    model: str
    inputs: List[Message]
    # Optional other fields, e.g., options: Dict[str, Any]

class Context(BaseModel):
    conversation_id: str
    request_id: str
    # You can add metadata here if needed

class MCPRequest(BaseModel):
    context: Context
    payload: Payload

class Choice(BaseModel):
    text: str

class PayloadResponse(BaseModel):
    choices: List[Choice]

class ContextResponse(BaseModel):
    conversation_id: str
    request_id: str
    tokens_used: int
    # Add latency_ms or other fields if needed

class MCPResponse(BaseModel):
    context: ContextResponse
    payload: PayloadResponse

# ─── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(title="Weather MCP Server")

@app.post("/mcp", response_model=MCPResponse)
async def handle_mcp(request: MCPRequest):
    """
    Receives an MCPRequest at /mcp, expects a user question about the weather,
    and returns an MCPResponse with dummy data.
    """
    context = request.context
    payload = request.payload

    # Extract user’s question (e.g., “what is the weather in nyc?”)
    user_msg = None
    for msg in payload.inputs:
        if msg.role == "user":
            user_msg = msg.content
            break

    if user_msg is None:
        raise HTTPException(status_code=400, detail="No user message in payload.")

    # Very naive parsing: look for a city name after "weather in "
    city = "unknown"
    lowered = user_msg.lower()
    if "weather in" in lowered:
        try:
            city = lowered.split("weather in")[1].strip().rstrip("?").strip()
        except Exception:
            city = "unknown"

    # Dummy temperature data; in a real implementation, call a real API
    dummy_temps = {
        "nyc": "58°F",
        "london": "48°F",
        "san francisco": "62°F"
    }
    temp = dummy_temps.get(city, "65°F (approx)")

    # Build response context
    response_context = {
        "conversation_id": context.conversation_id,
        "request_id": context.request_id,
        "tokens_used": 0  # set to 0 or mock value; depends on your billing model
    }

    # Build response payload
    response_payload = {
        "choices": [
            { "text": f"The current temperature in {city.title()} is {temp}." }
        ]
    }

    return {
        "context": response_context,
        "payload": response_payload
    }

if __name__ == "__main__":
    # Run with: python3 weather_server.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
