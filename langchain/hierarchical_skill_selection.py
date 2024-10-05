# hierarchical_skill_selection.py

import os
import requests
import logging
import numpy as np

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.vectorstores import FAISS
import faiss

# -------------------------------
# 1. Setup and Configuration
# -------------------------------

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WOLFRAM_ALPHA_APP_ID = os.getenv("WOLFRAM_ALPHA_APP_ID")
ZAPIER_WEBHOOK_URL = os.getenv("ZAPIER_WEBHOOK_URL")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")

# Validate that all necessary API keys are present
if not all([OPENAI_API_KEY, WOLFRAM_ALPHA_APP_ID, ZAPIER_WEBHOOK_URL, SLACK_BOT_TOKEN]):
    logging.error("One or more API keys are missing. Please set them as environment variables.")
    exit(1)

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# -------------------------------
# 2. Define Tool Groups and Tools
# -------------------------------

# Define tool groups with descriptions
tool_groups = {
    "Computation": {
        "description": "Tools related to mathematical computations and data analysis.",
        "tools": []
    },
    "Automation": {
        "description": "Tools that automate workflows and integrate different services.",
        "tools": []
    },
    "Communication": {
        "description": "Tools that facilitate communication and messaging.",
        "tools": []
    }
}

# Define Tools

@tool
def query_wolfram_alpha(expression: str) -> str:
    """
    Query Wolfram Alpha to compute mathematical expressions or retrieve information.

    Args:
        expression (str): The mathematical expression or query to evaluate.

    Returns:
        str: The result of the computation or the retrieved information.

    Raises:
        ValueError: If the API request fails or returns an error.
    """
    api_url = f"https://api.wolframalpha.com/v1/result?i={requests.utils.quote(expression)}&appid={WOLFRAM_ALPHA_APP_ID}"
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.text
        else:
            raise ValueError(f"Wolfram Alpha API Error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to query Wolfram Alpha: {e}")

@tool
def trigger_zapier_webhook(zap_id: str, payload: dict) -> str:
    """
    Trigger a Zapier webhook to execute a predefined Zap.

    Args:
        zap_id (str): The unique identifier for the Zap to be triggered.
        payload (dict): The data to send to the Zapier webhook.

    Returns:
        str: Confirmation message upon successful triggering of the Zap.

    Raises:
        ValueError: If the API request fails or returns an error.
    """
    zapier_webhook_url = f"https://hooks.zapier.com/hooks/catch/{zap_id}/"
    try:
        response = requests.post(zapier_webhook_url, json=payload)
        if response.status_code == 200:
            return f"Zapier webhook '{zap_id}' successfully triggered."
        else:
            raise ValueError(f"Zapier API Error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to trigger Zapier webhook '{zap_id}': {e}")

@tool
def send_slack_message(channel: str, message: str) -> str:
    """
    Send a message to a specified Slack channel.

    Args:
        channel (str): The Slack channel ID or name where the message will be sent.
        message (str): The content of the message to send.

    Returns:
        str: Confirmation message upon successful sending of the Slack message.

    Raises:
        ValueError: If the API request fails or returns an error.
    """
    api_url = "https://slack.com/api/chat.postMessage"
    headers = {
        "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "channel": channel,
        "text": message
    }
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response_data = response.json()
        if response.status_code == 200 and response_data.get("ok"):
            return f"Message successfully sent to Slack channel '{channel}'."
        else:
            error_msg = response_data.get("error", "Unknown error")
            raise ValueError(f"Slack API Error: {error_msg}")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to send message to Slack channel '{channel}': {e}")

# Assign tools to their respective groups
tool_groups["Computation"]["tools"].append(query_wolfram_alpha)
tool_groups["Automation"]["tools"].append(trigger_zapier_webhook)
tool_groups["Communication"]["tools"].append(send_slack_message)

# -------------------------------
# 3. Embed Group and Tool Descriptions
# -------------------------------

# Embed group descriptions
group_names = []
group_embeddings = []
for group_name, group_info in tool_groups.items():
    group_names.append(group_name)
    group_embeddings.append(embeddings.embed_text(group_info["description"]))

# Create FAISS index for groups
group_embeddings_np = np.array(group_embeddings).astype('float32')
faiss.normalize_L2(group_embeddings_np)
group_index = faiss.IndexFlatL2(len(group_embeddings_np[0]))
group_index.add(group_embeddings_np)

# Embed tool descriptions within each group
tool_indices = {}  # Maps group name to its FAISS index and tool functions
for group_name, group_info in tool_groups.items():
    tools = group_info["tools"]
    tool_descriptions = []
    tool_functions = []
    for tool_func in tools:
        description = tool_func.__doc__.strip().split('\n')[0]  # First line of docstring
        tool_descriptions.append(description)
        tool_functions.append(tool_func)
    if tool_descriptions:
        tool_embeddings = embeddings.embed_texts(tool_descriptions)
        tool_embeddings_np = np.array(tool_embeddings).astype('float32')
        faiss.normalize_L2(tool_embeddings_np)
        tool_index = faiss.IndexFlatL2(len(tool_embeddings_np[0]))
        tool_index.add(tool_embeddings_np)
        tool_indices[group_name] = {
            "index": tool_index,
            "functions": tool_functions,
            "embeddings": tool_embeddings_np
        }

# -------------------------------
# 4. Hierarchical Skill Selection
# -------------------------------

def select_group(query: str, top_k: int = 1) -> list:
    """
    Select the most relevant group(s) based on the user's query.

    Args:
        query (str): The user's input query.
        top_k (int): Number of top groups to retrieve.

    Returns:
        list: List of selected group names.
    """
    query_embedding = embeddings.embed_text(query).astype('float32')
    faiss.normalize_L2(query_embedding.reshape(1, -1))
    D, I = group_index.search(query_embedding.reshape(1, -1), top_k)
    selected_groups = [group_names[idx] for idx in I[0]]
    return selected_groups

def select_tool(query: str, group_name: str, top_k: int = 1) -> list:
    """
    Select the most relevant tool(s) within a specific group based on the user's query.

    Args:
        query (str): The user's input query.
        group_name (str): The name of the group to search within.
        top_k (int): Number of top tools to retrieve.

    Returns:
        list: List of selected tool functions.
    """
    if group_name not in tool_indices:
        return []
    tool_info = tool_indices[group_name]
    query_embedding = embeddings.embed_text(query).astype('float32')
    faiss.normalize_L2(query_embedding.reshape(1, -1))
    D, I = tool_info["index"].search(query_embedding.reshape(1, -1), top_k)
    selected_tools = [tool_info["functions"][idx] for idx in I[0] if idx < len(tool_info["functions"])]
    return selected_tools

# -------------------------------
# 5. Initialize the Language Model
# -------------------------------

# Initialize the LLM with GPT-4 and set temperature to 0 for deterministic responses
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# -------------------------------
# 6. Handle User Interaction
# -------------------------------

def main():
    # Prompt user for input
    user_query = input("Enter your query: ").strip()
    
    if not user_query:
        print("No query entered. Exiting.")
        return
    
    # Step 1: Select the most relevant group
    selected_groups = select_group(user_query, top_k=1)
    
    if not selected_groups:
        print("No relevant skill group found for your query.")
        return
    
    selected_group = selected_groups[0]
    logging.info(f"Selected Group: {selected_group}")
    print(f"Selected Skill Group: {selected_group}")
    
    # Step 2: Select the most relevant tool within the group
    selected_tools = select_tool(user_query, selected_group, top_k=1)
    
    if not selected_tools:
        print("No relevant tool found within the selected group.")
        return
    
    selected_tool = selected_tools[0]
    logging.info(f"Selected Tool: {selected_tool.__name__}")
    print(f"Selected Tool: {selected_tool.__name__}")
    
    # Prepare arguments based on the tool
    args = {}
    if selected_tool == query_wolfram_alpha:
        # Assume the entire query is the expression
        args["expression"] = user_query
    elif selected_tool == trigger_zapier_webhook:
        # For demonstration, use placeholders
        args["zap_id"] = "123456"  # Replace with actual Zap ID
        args["payload"] = {"message": user_query}
    elif selected_tool == send_slack_message:
        # For demonstration, use placeholders
        args["channel"] = "#general"  # Replace with actual Slack channel
        args["message"] = user_query
    else:
        print("Selected tool is not recognized.")
        return
    
    # Invoke the selected tool
    try:
        tool_result = selected_tool.invoke(args)
        print(f"Tool '{selected_tool.__name__}' Result: {tool_result}")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
