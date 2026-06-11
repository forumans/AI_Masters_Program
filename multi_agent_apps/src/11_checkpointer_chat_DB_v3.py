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
It allows you to maintain state across multiple invocations of the graph. This is useful for chatbots 
where you want to maintain the conversation history.

In this program, we will use PostgresSaver to save the state to a database.
When the user sends a message, the state is saved to the database with user as the thread_id.
At the end, when user exits, we will retrieve the summary of the conversation, 
delete the detailed history from the database, and save only the summary.

THIS IS DESIGNED TO STORE THE STATE IN THE DATABASE FOR MULTIPLE USERS.
USER_NAME IS USED TO STORE THE STATE IN THE DATABASE FOR THREAD_ID.
"""

from langchain_openai import ChatOpenAI
#from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage
from psycopg_pool import ConnectionPool


from dotenv import load_dotenv
load_dotenv()


def login() -> str:
    return input("Enter your username (it will be used to save your conversation history in the database) : ").strip()


def setup_custom_summary_table(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_summaries (
            username   VARCHAR PRIMARY KEY,
            summary    TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)


def load_user_summary(conn, username: str):
    row = conn.execute(
        "SELECT summary FROM user_summaries WHERE username = %s;",
        (username,)
    ).fetchone()
    return row[0] if row else None


def save_user_summary(conn, username: str, summary: str) -> None:
    conn.execute("""
        INSERT INTO user_summaries (username, summary, updated_at)
        VALUES (%s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (username)
        DO UPDATE SET summary    = EXCLUDED.summary,
                      updated_at = CURRENT_TIMESTAMP;
    """, (username, summary))


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
    Retrieve the current conversation thread/state/in-memory cache and generate a summary using the LLM.
    
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
    # Retrieve current state to get all conversation messages from in-memory cache
    current_state = graph.get_state(config)
    messages = current_state.values.get("messages", [])
    
    # Generate summary of all the messages using LLM if messages exist
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
username = login()

with ConnectionPool(conninfo=connect_string, max_size=10) as pool:

    # Create Postgres checkpointer from connection string
    with PostgresSaver.from_conn_string(connect_string) as checkpointer:

        # checkpointer.setup() - Create the necessary tables in the database only if they don't exist.
        checkpointer.setup() 

        # Setup our own additional custom table in the database to store user summaries
        with pool.connection() as conn:
            setup_custom_summary_table(conn)

        # Load prior summary from the summary table for this user
        with pool.connection() as conn:
            prior_summary = load_user_summary(conn, username)

        graph = builder.compile(checkpointer=checkpointer) # Compile the graph with the checkpointer, and summary details

        config = {"configurable": {"thread_id": f"user_{username}"}} # Configure the graph with the user ID

        if prior_summary:
            graph.invoke({"messages": [AIMessage(content=f"Previous session summary: {prior_summary}")]}, config)
            print(f"Welcome back, {username}! Restoring your previous session context.")
        else:
            print(f"Welcome, {username}! Starting a fresh conversation.")

        while True:
            user_input = input("User: ")

            if user_input.lower() == "exit":
                summary = summarize_conversation(graph, config)

                if summary != "No conversation to summarize.":
                    with pool.connection() as conn:
                        save_user_summary(conn, username, summary) # Save the summary to the database

                # Explicitly delete this user's thread from all checkpoint tables 
                # as the summary is already saved to user_summary table
                thread_id = config["configurable"]["thread_id"]
                with pool.connection() as conn:
                    conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
                    conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
                    conn.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))

                print(f"\nConversation Summary:\n{summary}")
                break

            result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)

            print(f"User: {user_input}")
            print(f"AI Response: {result['messages'][-1].content}")

