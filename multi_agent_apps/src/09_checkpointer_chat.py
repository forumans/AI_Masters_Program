"""

Checkpointer Chat Example

This example demonstrates how to use a checkpointer to maintain conversation history
in a chatbot using LangGraph.

While building a graph, the State object (TypedDict or Pydantic model) defines the structured schema your agent uses to flow data through graph nodes. 
When compiling the graph, the Checkpointer object provides persistence and memory, saving snapshots of that state

When you compile the graph with a checkpointer, i.e., the below line:
```python
checkpointer = InMemorySaver()
```
it allows you to maintain state across multiple invocations of the graph. This is useful for chatbots where you
want to maintain the conversation history.


"""

from ast import In
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict, Any


from dotenv import load_dotenv
load_dotenv()


llm = ChatOpenAI(model="gpt-4o")

# Chatbot node
def chatbot(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# Build the graph
builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# Create checkpointer for memory and compile graph using it
# For saving it to a database, use PostgresSaver()
#  Ex: checkpointer = PostgresSaver(connect_string="postgresql://user:password@localhost/dbname")
#      checkpointer.setup() - Call this only once to create the necessary tables in the database
# 

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Configuration
config = {"configurable": {"thread_id": "chat_session_1"}}

while True:
    user_input = input("User: ")

    if user_input.lower() == "exit":
        break

    
    result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)

    print(f"User: {user_input}")
    print(f"AI Response: {result['messages'][-1].content}")

