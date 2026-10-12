"""
Write a simple and straight-forward python program with a class, and functions to implement HITL using interrupt(), and Command() from langgraph.types. 
User logs in with a user name on the command prompt, and then a prompt will be displayed on the command line asking the user to enter his question. 
Before sending the question to OpenAI LLM for response, HITL approval is needed by an admin user. 
When an admin user logs in, the system should display user name and his query asking for approval on the command line. 
Once the admin gives approval, next time when the user logs in, all his approved queries would be executed one by one using LLM API. 
If admin rejected the query request, when the user logs in next time, he would be displayed with a report of rejected queries.
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

# ==========================================
# 1. DEFINE STATE SHAPE
# ==========================================
class AgentState(TypedDict):
    username: str
    query: str
    admin_action: Optional[str]  # "approved" or "rejected"
    llm_response: Optional[str]
    rejection_report: Optional[str]

# ==========================================
# 2. WORKFLOW GRAPH CLASS
# ==========================================
class HITLWorkflowManager:
    def __init__(self):
        # MemorySaver acts as our persistent state backend (similar to Postgres)
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        """Builds and compiles the LangGraph workflow."""
        builder = StateGraph(AgentState)

        # Add processing nodes
        builder.add_node("request_gate", self.request_gate_node)
        builder.add_node("execute_llm", self.execute_llm_node)
        builder.add_node("handle_rejection", self.handle_rejection_node)

        # Establish direct sequential flow structure
        builder.add_edge(START, "request_gate")
        
        # Compile graph with persistence checkpointer enabled
        return builder.compile(checkpointer=self.checkpointer)

    @staticmethod
    def request_gate_node(state: AgentState):
        """
        Pauses execution using interrupt(). When resumed via a Command(), 
        it dynamically updates the state and routes to the correct node.
        """
        print(f"\n[SYSTEM] Pausing for Admin Review on query from user '{state['username']}'...")
        
        # interrupt() pauses execution right here and returns the payload sent during resume
        admin_response = interrupt({
            "message": "Approval required",
            "user": state["username"],
            "query": state["query"]
        })

        # Process the incoming command input data
        action = admin_response.get("action") # "approved" or "rejected"

        # Command allows updating state and overriding the next node destination on the fly
        if action == "approved":
            return Command(
                update={"admin_action": "approved"},
                goto="execute_llm"
            )
        else:
            return Command(
                update={"admin_action": "rejected"},
                goto="handle_rejection"
            )

    @staticmethod
    def execute_llm_node(state: AgentState):
        """Simulates executing the query against the OpenAI API upon approval."""
        print(f"\n[LLM] Processing approved query for {state['username']}...")
        # Mocking an OpenAI API completion response
        mock_response = f"Mocked OpenAI Response to: '{state['query']}'"
        return {"llm_response": mock_response}

    @staticmethod
    def handle_rejection_node(state: AgentState):
        """Handles administrative rejections by building a stateful report."""
        print(f"\n[SYSTEM] Logging rejection for {state['username']}...")
        report = f"REJECTED: Your query '{state['query']}' violated system prompts."
        return {"rejection_report": report}


# ==========================================
# 3. INTERACTIVE CLI EXECUTION FLOW
# ==========================================
if __name__ == "__main__":
    # Initialize our system workflow
    manager = HITLWorkflowManager()
    
    # Track simple thread metadata mappings
    config = {"configurable": {"thread_id": "session_123"}}
    
    print("=== STEP 1: USER CONSOLE ===")
    username = input("Enter your username: ")
    user_query = input("Enter your question: ")

    # Kick off the initial thread execution
    initial_input = {"username": username, "query": user_query}
    print("\n[EXECUTION] Submitting request into LangGraph...")
    
    # Stream runs until it hits the `interrupt()` inside 'request_gate'
    for event in manager.graph.stream(initial_input, config, stream_mode="values"):
        pass

    # Inspect current graph state to verify it is paused
    current_state = manager.graph.get_state(config)
    if current_state.next:
        print(f"\n[STATE] Graph paused successfully at: {current_state.next}")

    print("\n=== STEP 2: ADMIN CONSOLE ===")
    print(f"Logged in as: Admin")
    # Fetch pending payload directly out of the active interrupt state context
    pending_request = current_state.tasks[0].interrupts[0].value
    print(f"Pending User: {pending_request['user']}")
    print(f"Pending Query: {pending_request['query']}")
    
    admin_decision = input("Enter decision (approve/reject): ").strip().lower()
    action_value = "approved" if admin_decision == "approve" else "rejected"

    print("\n[EXECUTION] Resuming Graph via Command input routing...")
    
    # Resume execution by supplying the Command payload targeting the open interrupt
    resume_payload = {"action": action_value}
    for event in manager.graph.stream(Command(resume=resume_payload), config, stream_mode="values"):
        final_state_data = event

    print("\n=== STEP 3: USER RETURN RESULTS ===")
    if final_state_data.get("llm_response"):
        print(f"Status: SUCCESS")
        print(f"Result: {final_state_data['llm_response']}")
    elif final_state_data.get("rejection_report"):
        print(f"Status: BLOCKED")
        print(f"Report: {final_state_data['rejection_report']}")
