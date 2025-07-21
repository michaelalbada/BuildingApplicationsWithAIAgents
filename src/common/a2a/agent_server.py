import json
from http.server import BaseHTTPRequestHandler, HTTPServer

#  Agent Card (JSON descriptor for discovery)
agent_card = {
    "identity": "SummarizerAgent",
    "capabilities": ["summarizeText"],
    "schemas": {
        "summarizeText": {
            "input": {"text": "string"},
            "output": {"summary": "string"}
        }
    },
    "endpoint": "http://localhost:8000/api",
    "auth_methods": ["none"],  # In production, use OAuth2, API keys, etc.
    "version": "1.0"
}

class AgentHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/.well-known/agent.json':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(agent_card).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/api':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            rpc_request = json.loads(post_data)
            
            # Handle JSON-RPC request (core of A2A)
            if rpc_request.get('jsonrpc') == '2.0' and rpc_request['method'] == 'summarizeText':
                text = rpc_request['params']['text']
                # Mock summarization (in real use, call an LLM here)
                summary = text[:100] + "..." if len(text) > 100 else text
                response = {
                    "jsonrpc": "2.0",
                    "result": {"summary": summary},
                    "id": rpc_request['id']
                }
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            else:
                # Error handling as per JSON-RPC
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": "Method not found"},
                    "id": rpc_request.get('id')
                }
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(error_response).encode())
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == '__main__':
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, AgentHandler)
    print("Starting A2A agent server on http://localhost:8000")
    httpd.serve_forever()