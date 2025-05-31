#!/usr/bin/env python3
import sys
import json
import ast
import operator
from typing import Any, Dict

# ─── Safe Expression Evaluation ────────────────────────────────────────────────
# Restrict nodes to only arithmetic operators (+, -, *, /, **, parentheses).
# This prevents code injection via eval().

ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

def eval_expr(node: ast.AST) -> float:
    if isinstance(node, ast.Num):  # <number>
        return node.n
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
        left = eval_expr(node.left)
        right = eval_expr(node.right)
        return ALLOWED_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_OPERATORS:
        operand = eval_expr(node.operand)
        return ALLOWED_OPERATORS[type(node.op)](operand)
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")

def compute_math(expression: str) -> float:
    """
    Parse and evaluate a simple arithmetic expression safely.
    """
    try:
        expr_ast = ast.parse(expression, mode="eval").body
        return eval_expr(expr_ast)
    except Exception as e:
        raise ValueError(f"Error parsing expression '{expression}': {e}")

# ─── Main Loop ───────────────────────────────────────────────────────────────────
def main():
    """
    Continuously read lines from stdin. Each line should be a complete MCPRequest JSON.
    For each request, compute the result and write an MCPResponse JSON to stdout.
    """
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            # Parse incoming MCPRequest envelope
            request: Dict[str, Any] = json.loads(line)
            # Expected structure: { "context": {...}, "payload": {...} }
            context = request.get("context", {})
            payload = request.get("payload", {})

            # Extract the arithmetic question: assume payload.inputs is a list of messages,
            # find the first user message, and treat its `content` as the expression.
            # Example payload.inputs: [ { "role": "user", "content": "what's (3 + 5) * 12?" } ]
            inputs = payload.get("inputs", [])
            expr = None
            for msg in inputs:
                if msg.get("role") == "user":
                    expr = msg.get("content")
                    break

            if expr is None:
                raise ValueError("No user message found with an expression to compute.")

            # Remove any non‐numeric characters except arithmetic symbols (optional).
            # Here, for simplicity, we assume the user’s content is exactly something like "(3 + 5) * 12".
            # If there’s text like "what's " or "?", strip them out:
            cleaned = "".join(ch for ch in expr if ch.isdigit() or ch in "+-*/()^ .*")
            cleaned = cleaned.replace("^", "**")  # allow caret as power

            result_value = compute_math(cleaned)

            # Build MCPResponse envelope
            response = {
                "context": {
                    **context,
                    "request_id": context.get("request_id", "math-req-1"),
                    "status": "ok"
                },
                "payload": {
                    "choices": [
                        {"text": str(result_value)}
                    ]
                }
            }
        except Exception as err:
            # On error, respond with an MCP envelope containing the error message
            response = {
                "context": {
                    "request_id": context.get("request_id", "math-req-err"),
                    "status": "error",
                    "error": str(err)
                },
                "payload": {
                    "choices": [
                        {"text": f"Error: {err}"}
                    ]
                }
            }

        # Write response as one JSON per line, so the client (MultiServerMCPClient) can parse it.
        sys.stdout.write(json.dumps(response))
        sys.stdout.write("\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
