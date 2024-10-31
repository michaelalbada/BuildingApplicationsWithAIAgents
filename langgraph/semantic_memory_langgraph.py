from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START

llm = ChatOpenAI(model="gpt-4o")

def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": response}


from vectordb import Memory

memory = Memory(chunking_strategy={'mode':'sliding_window', 'window_size': 128, 'overlap': 16})

text = """
Machine learning is a method of data analysis that automates analytical model building.

It is a branch of artificial intelligence based on the idea that systems can learn from data,
identify patterns and make decisions with minimal human intervention.

Machine learning algorithms are trained on data sets that contain examples of the desired output. For example, a machine learning algorithm that is used to classify images might be trained on a data set that contains images of cats and dogs.
Once an algorithm is trained, it can be used to make predictions on new data. For example, the machine learning algorithm that is used to classify images could be used to predict whether a new image contains a cat or a dog.
"""

metadata = {"title": "Introduction to Machine Learning", "url": "https://example.com/introduction-to-machine-learning"}

memory.save(text, metadata)

text2 = """
Artificial intelligence (AI) is the simulation of human intelligence in machines
that are programmed to think like humans and mimic their actions.

The term may also be applied to any machine that exhibits traits associated with
a human mind such as learning and problem-solving.

AI research has been highly successful in developing effective techniques for solving a wide range of problems, from game playing to medical diagnosis.
"""

metadata2 = {"title": "Introduction to Artificial Intelligence", "url": "https://example.com/introduction-to-artificial-intelligence"}

memory.save(text2, metadata2)

query = "What is the relationship between AI and machine learning?"

results = memory.search(query, top_n=3)

builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
graph = builder.compile()

input_message = {"type": "user", "content": "hi! I'm bob"}
for chunk in graph.stream({"messages": [input_message]}, {}, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

print(results)
