from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START
from langchain_core.messages import HumanMessage

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o")

# Function to call the LLM
def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": response}

class InsightAgent:
    def __init__(self):
        self.insights = []
        self.promoted_insights = []
        self.demoted_insights = []
        self.reflections = []

    def generate_insight(self, observation):
        # Use the LLM to generate an insight based on the observation
        messages = [HumanMessage(content=f"Generate an insightful analysis based on the following observation: '{observation}'")]

        # Build the state graph
        builder = StateGraph(MessagesState)
        builder.add_node("generate_insight", call_model)
        builder.add_edge(START, "generate_insight")
        graph = builder.compile()

        # Invoke the graph with the messages
        result = graph.invoke({"messages": messages})

        # Extract the generated insight
        generated_insight = result["messages"][-1].content
        self.insights.append(generated_insight)
        print(f"Generated: {generated_insight}")
        return generated_insight

    def promote_insight(self, insight):
        if insight in self.insights:
            self.insights.remove(insight)
            self.promoted_insights.append(insight)
            print(f"Promoted: {insight}")
        else:
            print(f"Insight '{insight}' not found in insights.")

    def demote_insight(self, insight):
        if insight in self.promoted_insights:
            self.promoted_insights.remove(insight)
            self.demoted_insights.append(insight)
            print(f"Demoted: {insight}")
        else:
            print(f"Insight '{insight}' not found in promoted insights.")

    def edit_insight(self, old_insight, new_insight):
        # Check in all lists
        if old_insight in self.insights:
            index = self.insights.index(old_insight)
            self.insights[index] = new_insight
        elif old_insight in self.promoted_insights:
            index = self.promoted_insights.index(old_insight)
            self.promoted_insights[index] = new_insight
        elif old_insight in self.demoted_insights:
            index = self.demoted_insights.index(old_insight)
            self.demoted_insights[index] = new_insight
        else:
            print(f"Insight '{old_insight}' not found.")
            return
        print(f"Edited: '{old_insight}' to '{new_insight}'")

    def show_insights(self):
        print("\nCurrent Insights:")
        print(f"Insights: {self.insights}")
        print(f"Promoted Insights: {self.promoted_insights}")
        print(f"Demoted Insights: {self.demoted_insights}")

    def reflect(self, reflexion_prompt):
        # Build the state graph for reflection
        builder = StateGraph(MessagesState)
        builder.add_node("reflection", call_model)
        builder.add_edge(START, "reflection")
        graph = builder.compile()

        # Invoke the graph with the reflection prompt
        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=reflexion_prompt
                    )
                ]
            }
        )
        reflection = result["messages"][-1].content
        self.reflections.append(reflection)
        print(f"Reflection: {reflection}")

# Example usage:
agent = InsightAgent()

# Generate insights based on observations
insight1 = agent.generate_insight("The sales have increased by 20% in the last quarter.")
insight2 = agent.generate_insight("Customer complaints have decreased significantly.")

# Promote an insight
agent.promote_insight(insight1)

# Demote an insight
agent.demote_insight(insight1)

# Edit an existing insight
agent.edit_insight(insight2, "Refined insight: Customer satisfaction is improving due to reduced complaints.")

# Display all insights
agent.show_insights()

# Perform reflection using the LLM
reflexion_prompt = "Reflect on the current insights and provide suggestions for improvement."
agent.reflect(reflexion_prompt)