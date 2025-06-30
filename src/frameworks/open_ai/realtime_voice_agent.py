# realtime_voice_minimal.py
import os, json, base64, asyncio, websockets
from fastapi import FastAPI, WebSocket
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VOICE          = "alloy"                 # GPT-4o voice
PCM_SR         = 16000                   # sample-rate we’ll use client-side
PORT           = 5050

app = FastAPI()

@app.websocket("/voice")
async def voice_bridge(ws: WebSocket) -> None:
    """
    1. Browser opens ws://host:5050/voice
    2. Browser streams base64-encoded 16-bit mono PCM chunks: {"audio": "<b64>"}
    3. We forward chunks to OpenAI Realtime (`input_audio_buffer.append`)
    4. We relay assistant audio deltas back to the browser the same way
    5. We listen for 'speech_started' events and send a truncate if user interrupts
    """
    await ws.accept()

    # --- open downstream connection to OpenAI Realtime -----------------------
    openai_ws = await websockets.connect(
        "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01",
        extra_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta" : "realtime=v1"
        },
        max_size=None, max_queue=None # unbounded for demo simplicity
    )

    # initialise the realtime session
    await openai_ws.send(json.dumps({
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": f"pcm_{PCM_SR}",
            "output_audio_format": f"pcm_{PCM_SR}",
            "voice": VOICE,
            "modalities": ["audio"],
            "instructions": "You are a concise AI assistant."
        }
    }))

    last_assistant_item = None          # track current assistant response
    latest_pcm_ts       = 0             # ms timestamp from client
    pending_marks       = []

    async def from_client() -> None:
        """Relay microphone PCM chunks from browser → OpenAI."""
        nonlocal latest_pcm_ts
        async for msg in ws.iter_text():
            data = json.loads(msg)
            pcm = base64.b64decode(data["audio"])
            latest_pcm_ts += int(len(pcm) / (PCM_SR * 2) * 1000)  # naïve ms counter
            await openai_ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode("ascii")
            }))

    async def to_client() -> None:
        """Relay assistant audio + handle interruptions."""
        nonlocal last_assistant_item, pending_marks
        async for raw in openai_ws:
            msg = json.loads(raw)

            # assistant speaks
            if msg["type"] == "response.audio.delta":
                pcm = base64.b64decode(msg["delta"])
                await ws.send_json({"audio": base64.b64encode(pcm).decode("ascii")})
                last_assistant_item = msg.get("item_id")

            # user started talking → cancel assistant speech
            if msg["type"] == "input_audio_buffer.speech_started" and last_assistant_item:
                await openai_ws.send(json.dumps({
                    "type": "conversation.item.truncate",
                    "item_id": last_assistant_item,
                    "content_index": 0,
                    "audio_end_ms": 0   # stop immediately
                }))
                last_assistant_item = None
                pending_marks.clear()

    try:
        await asyncio.gather(from_client(), to_client())
    finally:
        await openai_ws.close()
        await ws.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("realtime_voice_minimal:app", host="0.0.0.0", port=PORT)
