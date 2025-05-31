import os
import requests
import logging
import numpy as np
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Initialize the LLM
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

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
# LLM-Based Hierarchical Skill Selection
# -------------------------------
def select_group_llm(query: str) -> str:
    """
    Use the LLM to determine the most appropriate skill group based on the user's query.
    
    Args:
        query (str): The user's input query.
        
    Returns:
        str: The name of the selected group.
    """
    prompt = f"Select the most appropriate skill group for the following query: '{query}'.\nOptions are: Computation, Automation, Communication."
    response = llm([HumanMessage(content=prompt)])
    return response.content.strip()

def select_tool_llm(query: str, group_name: str) -> str:
    """
    Use the LLM to determine the most appropriate tool within a group based on the user's query.
    
    Args:
        query (str): The user's input query.
        group_name (str): The name of the selected skill group.
        
    Returns:
        str: The name of the selected tool function.
    """
    prompt = f"Based on the query: '{query}', select the most appropriate tool from the group '{group_name}'."
    response = llm([HumanMessage(content=prompt)])
    return response.content.strip()

# Example user query
user_query = "Solve this equation: 2x + 3 = 7"

# Step 1: Select the most relevant skill group using LLM
selected_group_name = select_group_llm(user_query)
if not selected_group_name:
    print("No relevant skill group found for your query.")
else:
    logging.info(f"Selected Group: {selected_group_name}")
    print(f"Selected Skill Group: {selected_group_name}")

    # Step 2: Select the most relevant tool within the group using LLM
    selected_tool_name = select_tool_llm(user_query, selected_group_name)
    selected_tool = globals().get(selected_tool_name, None)
    
    if not selected_tool:
        print("No relevant tool found within the selected group.")
    else:
        logging.info(f"Selected Tool: {selected_tool.__name__}")
        print(f"Selected Tool: {selected_tool.__name__}")
        
        # Prepare arguments based on the tool
        args = {}
        if selected_tool == query_wolfram_alpha:
            # Assume the entire query is the expression
            args["expression"] = user_query
        elif selected_tool == trigger_zapier_webhook:
            # Use placeholders for demo
            args["zap_id"] = "123456"
            args["payload"] = {"message": user_query}
        elif selected_tool == send_slack_message:
            # Use placeholders for demo
            args["channel"] = "#general"
            args["message"] = user_query
        else:
            print("Selected tool is not recognized.")
        
        # Invoke the selected tool
        try:
            tool_result = selected_tool.invoke(args)
            print(f"Tool '{selected_tool.__name__}' Result: {tool_result}")
        except ValueError as e:
            print(f"Error: {e}")
