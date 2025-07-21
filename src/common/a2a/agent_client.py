import requests
import json

# Step 1: Discover the agent via registry or known URL (mocked as direct access)
card_url = 'http://localhost:8000/.well-known/agent.json'
response = requests.get(card_url)
if response.status_code != 200:
    print("Failed to retrieve Agent Card")
    exit()

agent_card = response.json()
print("Discovered Agent Card:", json.dumps(agent_card, indent=2))

# Step 2: Handshake/Negotiation (mocked: check version and capabilities)
if agent_card['version'] != '1.0':
    print("Incompatible protocol version")
    exit()
if "summarizeText" not in agent_card['capabilities']:
    print("Required capability not supported")
    exit()
print("Handshake successful: Agent is compatible.")

# Step 3: Issue a structured JSON-RPC request
rpc_url = agent_card['endpoint']
rpc_request = {
    "jsonrpc": "2.0",
    "method": "summarizeText",
    "params": {"text": "This is a long example text that needs summarization. It discusses multi-agent systems, communication protocols, and how agents can collaborate autonomously using standards like A2A."},
    "id": 123  # Unique request ID
}

response = requests.post(rpc_url, json=rpc_request)
if response.status_code == 200:
    rpc_response = response.json()
    print("RPC Response:", json.dumps(rpc_response, indent=2))
else:
    print("Error:", response.status_code, response.text)
