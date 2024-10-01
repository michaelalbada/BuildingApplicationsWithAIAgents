from typing import Annotated, Literal

Operator = Literal["+", "-", "*", "/"]


def calculator(a: int, b: int, operator: Annotated[Operator, "operator"]) -> int:
   if operator == "+":
       return a + b
   elif operator == "-":
       return a - b
   elif operator == "*":
       return a * b
   elif operator == "/":
       return int(a / b)
   else:
       raise ValueError("Invalid operator")
   
import os
from autogen import ConversableAgent

# We define the assistant agent
assistant = ConversableAgent(
   name="ReliableCalculatorAssistant",
   system_message="You are a helpful AI assistant that can help with simple calculations. "
   "Return 'TERMINATE' when the task is done.",
   llm_config={"config_list": [{"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]}]},
)

# We define a user proxy agent to interacting with the assistant agent and execute skills.
user_proxy = ConversableAgent(
   name="UserProxy",
   llm_config=False,
   is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
   human_input_mode="NEVER",
)

# We then register the skill with the assistant agent.
assistant.register_for_llm(name="calculator", description="A basic calculator")(calculator)

# Register the tool function with the user proxy agent.
user_proxy.register_for_execution(name="calculator")(calculator)

chat_result = user_proxy.initiate_chat(assistant, message="What is (44232 + 13312 / (232 - 32)) * 5?")