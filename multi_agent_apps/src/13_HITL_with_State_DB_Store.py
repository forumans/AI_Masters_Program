import os
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status, Depends
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START
from langgraph.types import interrupt, Command
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

# ==========================================
# 1. GRAPH STATE & PYDANTIC ROUTE SCHEMAS
# ==========================================
class AgentState(Dict[str, Any]):
    username: str
    query: str
    admin_action: Optional[str]
    llm_response: Optional[str]
    rejection_report: Optional[str]

class UserQueryRequest(BaseModel):
    username: str = Field(..., example="john_doe")
    query: str = Field(..., example="What is our policy on remote work?")

class AdminReviewRequest(BaseModel):
    action: str = Field(..., example="approved", description="Must be either 'approved' or 'rejected'")

# ==========================================
# 2. SYSTEM WORKFLOW DEFINITION (SERVICE LAYER)
# ==========================================
class CloudHITLGraphManager:
    """Manages compilation and execution of the Async Postgres LangGraph."""
    def __init__(self, pool: AsyncConnectionPool):
        # Async checkpointer configuration for production cloud volumes
        self.checkpointer = AsyncPostgresSaver(pool)
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("request_gate", self.request_gate_node)
        builder.add_node("execute_llm", self.execute_llm_node)
        builder.add_node("handle_rejection", self.handle_rejection_node)

        builder.add_edge(START, "request_gate")
        return builder.compile(checkpointer=self.checkpointer)

    @staticmethod
    async def request_gate_node(state: AgentState):
        """Asynchronously suspends thread execution awaiting administrative intervention."""
        # state is automatically dropped to HITL_Checkpointer_DB
        admin_response = interrupt({
            "username": state.get("username"),
            "query": state.get("query")
        })
        
        action = admin_response.get("action")
        if action == "approved":
            return Command(update={"admin_action": "approved"}, goto="execute_llm")
        return Command(update={"admin_action": "rejected"}, goto="handle_rejection")

    @staticmethod
    async def execute_llm_node(state: AgentState):
        # Simulate an async third-party LLM invocation (e.g., await openai_client.chat.completions.create)
        return {"llm_response": f"Processed query: {state.get('query')}"}

    @staticmethod
    async def handle_rejection_node(state: AgentState):
        return {"rejection_report": "Rejected: Violates corporate compliance guidelines."}

# ==========================================
# 3. GLOBAL APPLICATION LIFESPAN & DEPENDENCIES
# ==========================================
# Cloud environment variable parsing
DB_URL = os.getenv("DATABASE_URL", "postgresql://langgraphcheckptr_user:password@localhost:5432/HITL_Checkpointer_DB")

# Global variables for state access across async requests
db_pool: Optional[AsyncConnectionPool] = None
graph_manager: Optional[CloudHITLGraphManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles production-safe application setup and teardown tasks."""
    global db_pool, graph_manager
    # Initialize connection pool with sizing constraints optimized for production containers
    db_pool = AsyncConnectionPool(conninfo=DB_URL, min_size=2, max_size=20, open=False)
    await db_pool.open()
    
    # Initialize and run structural setups for LangGraph's engine tables
    graph_manager = CloudHITLGraphManager(db_pool)
    await graph_manager.checkpointer.setup()
    
    print("[LIFESPAN] Async Database Pool connected & Checkpointer schemas ready.")
    yield
    # Clean teardown upon container exit
    await db_pool.close()
    print("[LIFESPAN] Async Database Pool cleanly disconnected.")

app = FastAPI(title="LangGraph Persistent HITL Service", version="1.0.0", lifespan=lifespan)

def get_graph_manager() -> CloudHITLGraphManager:
    """Dependency injection wrapper to enforce runtime safety validation."""
    if not graph_manager:
        raise HTTPException(status_code=503, detail="Application manager state uninitialized.")
    return graph_manager

# ==========================================
# 4. ASYNCHRONOUS CONTROLLER ROUTE HANDLERS
# ==========================================

@app.post("/v1/session/{thread_id}/submit", status_code=status.HTTP_202_ACCEPTED)
async def submit_user_query(
    thread_id: str, 
    payload: UserQueryRequest, 
    manager: CloudHITLGraphManager = Depends(get_graph_manager)
):
    """Submits a user query. Streams execution until it hits the admin interrupt gateway."""
    config = {"configurable": {"thread_id": thread_id}}
    initial_input = {"username": payload.username, "query": payload.query}
    
    try:
        # Asynchronously step through graph nodes up to the interrupt step
        async for _ in manager.graph.astream(initial_input, config, stream_mode="values"):
            pass
        
        return {"status": "paused_for_review", "thread_id": thread_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph initialization failed: {str(e)}")


@app.get("/v1/session/{thread_id}/review", status_code=status.HTTP_200_OK)
async def get_pending_review(
    thread_id: str,
    manager: CloudHITLGraphManager = Depends(get_graph_manager)
):
    """Fetches details of the pending interrupt inside the database for admin UI visualization."""
    config = {"configurable": {"thread_id": thread_id}}
    state = await manager.graph.aget_state(config)
    
    if not state.next:
        return {"status": "active_or_completed", "pending_interrupt": None}
    
    # Safely extract values out of the async interrupt container metadata array
    interrupt_payload = state.tasks[0].interrupts[0].value if state.tasks else None
    return {
        "status": "awaiting_review",
        "next_node": state.next[0],
        "review_data": interrupt_payload
    }


@app.post("/v1/session/{thread_id}/approve", status_code=status.HTTP_200_OK)
async def process_admin_action(
    thread_id: str,
    payload: AdminReviewRequest,
    manager: CloudHITLGraphManager = Depends(get_graph_manager)
):
    """Resumes execution of an interrupted thread from the database using a steering Command."""
    config = {"configurable": {"thread_id": thread_id}}
    state = await manager.graph.aget_state(config)
    
    if not state.next:
        raise HTTPException(status_code=400, detail="Thread is not in an interrupted state.")
        
    if payload.action not in ["approved", "rejected"]:
        raise HTTPException(status_code=422, detail="Action must be 'approved' or 'rejected'.")

    try:
        final_state = {}
        # Hydrate and resume execution by injecting the Command abstraction directly
        resume_command = Command(resume={"action": payload.action})
        async for event in manager.graph.astream(resume_command, config, stream_mode="values"):
            final_state = event

        return {
            "status": "completed",
            "thread_id": thread_id,
            "result": {
                "llm_response": final_state.get("llm_response"),
                "rejection_report": final_state.get("rejection_report")
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resuming graph failed: {str(e)}")
