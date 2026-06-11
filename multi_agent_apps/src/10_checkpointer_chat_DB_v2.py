"""
Checkpointer Chat Example with state saved to database

This example demonstrates how to save checkpointer state to a database to maintain conversation history
in a chatbot using LangGraph.

While building a graph, the State object (TypedDict or Pydantic model) defines the structured schema your agent uses to flow data through graph nodes. 
When compiling the graph, the Checkpointer object provides persistence and memory, saving snapshots of that state

When you compile the graph with a checkpointer, i.e., the below line:
```python
checkpointer = InMemorySaver()  # or PostgresSaver()
```
It allows you to maintain state across multiple invocations of the graph. This is useful for chatbots where you
want to maintain the conversation history.

In this program, we will use PostgresSaver to save the state to a database.
When the user sends a message, the state is saved to the database.
At the end, when user exits, we will retrieve the summary of the conversation, 
    delete the detailed history from the database, and save only the summary.

THIS IS DESIGNED TO STORE THE STATE IN THE DATABASE FOR ONLY ONE USER.
FOR  MULTI-USER SESSION STORAGE, WE NEED TO USE A UNIQUE THREAD ID FOR EACH USER AND THAT'S GIVEN IN V3 PROGRAM.
"""

from langchain_openai import ChatOpenAI
#from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage
from psycopg_pool import ConnectionPool


from dotenv import load_dotenv
load_dotenv()


llm = ChatOpenAI(model="gpt-4o")

# Chatbot node
def chatbot(state: MessagesState):
    """
    Process user messages through the LLM and generate AI responses.
    
    Args:
        state: Current conversation state containing messages
        
    Returns:
        Dictionary with the AI response message
    """
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# Summary the conversation to replace the long history from the database
def summarize_conversation(graph, config):
    """
    Retrieve the current conversation thread/state/in-memory and generate a summary using the LLM.
    
    This function:
    1. Retrieves all messages from the current conversation thread
    2. If messages exist, invokes the LLM to generate a comprehensive summary
    3. Returns the summary or a default message if no conversation exists
    
    Args:
        graph: The compiled LangGraph with checkpointer
        config: Configuration dictionary containing the thread_id
        
    Returns:
        str: Generated summary of the conversation or default message
    """
    # Retrieve current state to get all conversation messages
    current_state = graph.get_state(config)
    messages = current_state.values.get("messages", [])
    
    # Generate summary using LLM if messages exist
    if messages:
        summary_prompt = f"Summarize the following conversation: {messages}"
        response = llm.invoke([HumanMessage(content=summary_prompt)])
        summary = response.content
    else:
        summary = "No conversation to summarize."
    
    return summary


# Build the graph
builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)


# Create checkpointer for memory and compile graph using it
# For saving it to a database, use PostgresSaver()
#  Ex: checkpointer = PostgresSaver(connect_string="postgresql://user:password@localhost/dbname")
#      checkpointer.setup() - Create the necessary tables in the database to store the state
# 


# Database connection string
connect_string="postgresql://langgraphcheckptr_user:password@localhost/langgraphcheckptrDB"

# Set up connection pool and Postgres checkpointer
# Using a connection pool ensures stable connection handling for long-running workflows
with ConnectionPool(conninfo=connect_string, max_size=10) as pool:
    with PostgresSaver.from_conn_string(connect_string) as checkpointer: # Create a db checkpointer for the graph

        checkpointer.setup() # Check whether the tables exist in the database, if not create them

        graph = builder.compile(checkpointer=checkpointer) # Compile the graph with the checkpointer

        # Configuration - thread_id  is used to store the conversation history in the database
        config = {"configurable": {"thread_id": "chat_session_1"}} 

        while True:
            user_input = input("User: ")

            if user_input.lower() == "exit":
                # 1. Generate summary of the entire conversation
                summary = summarize_conversation(graph, config)
                
                # 2. Delete all conversation from the database using checkpointer
                checkpointer.delete_thread(config)
                
                # 3. Explicitly truncate all checkpoint tables to ensure complete cleanup
                # This removes any remaining records from previous sessions
                with pool.getconn() as conn:
                    with conn.cursor() as cur:
                        cur.execute("TRUNCATE TABLE checkpoint_blobs CASCADE;")
                        cur.execute("TRUNCATE TABLE checkpoint_writes CASCADE;")
                        cur.execute("TRUNCATE TABLE checkpoint_migrations CASCADE;")
                        conn.commit()
                
                # 4. Write the summarized single message to the database
                # Save to the same thread_id so it's available in future sessions
                summary_message = AIMessage(content=f"Conversation Summary: {summary}")
                graph.invoke({"messages": [summary_message]}, config)
                
                print(f"Conversation summarized and saved to database. Summary: {summary}")
                break

            
            result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)

            print(f"User: {user_input}")
            print(f"AI Response: {result['messages'][-1].content}")

