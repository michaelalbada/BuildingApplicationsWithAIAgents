# skill_tool_integration.py

import os
import requests
import logging

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.tools import WikipediaQueryRun  # Not used in this script
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
# 2. Define Tools
# -------------------------------

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

# -------------------------------
# 3. Embed Tool Descriptions
# -------------------------------

# Define tool descriptions
tool_descriptions = {
    "query_wolfram_alpha": "Use Wolfram Alpha to compute mathematical expressions or retrieve information.",
    "trigger_zapier_webhook": "Trigger a Zapier webhook to execute predefined automated workflows.",
    "send_slack_message": "Send messages to specific Slack channels to communicate with team members."
}

# Create embeddings for each tool description
tool_embeddings = []
tool_names = []
for tool_name, description in tool_descriptions.items():
    embedding = embeddings.embed_text(description)
    tool_embeddings.append(embedding)
    tool_names.append(tool_name)

# -------------------------------
# 4. Initialize Vector Store
# -------------------------------

# Initialize FAISS vector store
dimension = len(tool_embeddings[0])  # Assuming all embeddings have the same dimension
index = faiss.IndexFlatL2(dimension)
faiss.normalize_L2(tool_embeddings)  # Normalize embeddings for cosine similarity

# Convert list to FAISS-compatible format
import numpy as np
tool_embeddings_np = np.array(tool_embeddings).astype('float32')
index.add(tool_embeddings_np)

# Map index to tool functions
index_to_tool = {
    0: query_wolfram_alpha,
    1: trigger_zapier_webhook,
    2: send_slack_message
}

# -------------------------------
# 5. Select Appropriate Tool
# -------------------------------

def select_tool(query: str, top_k: int = 1) -> list:
    """
    Select the most relevant tool(s) based on the user's query using vector-based retrieval.

    Args:
        query (str): The user's input query.
        top_k (int): Number of top tools to retrieve.

    Returns:
        list: List of selected tool functions.
    """
    query_embedding = embeddings.embed_text(query).astype('float32')
    faiss.normalize_L2(query_embedding.reshape(1, -1))
    D, I = index.search(query_embedding.reshape(1, -1), top_k)
    selected_tools = [index_to_tool[idx] for idx in I[0] if idx in index_to_tool]
    return selected_tools

# -------------------------------
# 6. Initialize the Language Model
# -------------------------------

# Initialize the LLM with GPT-4 and set temperature to 0 for deterministic responses
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# -------------------------------
# 7. Handle User Interaction
# -------------------------------

def main():
    # Define the user query
    user_query = input("Enter your query: ")
    
    # Select the most relevant tool based on the query
    selected_tools = select_tool(user_query, top_k=1)
    
    if not selected_tools:
        print("No suitable tools found for the query.")
        return
    
    # For simplicity, select the top tool
    tool = selected_tools[0]
    
    # Prepare arguments based on the tool
    if tool == query_wolfram_alpha:
        expression = user_query  # Assuming the entire query is a mathematical expression
        args = {"expression": expression}
    elif tool == trigger_zapier_webhook:
        # Example: Extract zap_id and payload from the query or define them
        # For simplicity, we'll use placeholders
        zap_id = "123456"  # Replace with actual Zap ID
        payload = {"data": user_query}
        args = {"zap_id": zap_id, "payload": payload}
    elif tool == send_slack_message:
        # Example: Extract channel and message from the query or define them
        # For simplicity, we'll use placeholders
        channel = "#general"  # Replace with actual Slack channel ID or name
        message = user_query
        args = {"channel": channel, "message": message}
    else:
        print("Selected tool is not recognized.")
        return
    
    # Invoke the selected tool
    try:
        tool_result = tool.invoke(args)
        print(f"Tool '{tool.__name__}' Result: {tool_result}")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
