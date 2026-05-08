"""
State Management - The Memory System for Multi-Agent Workflows

PURPOSE:
--------
This file defines the data structures that store information as the
multi-agent system processes requests. Think of it as the system's
memory that tracks everything happening during a workflow.

WHAT IT TRACKS:
----------------
- User requests and what they want to accomplish
- Which agents are working and what they've done
- Current status of the workflow (pending, running, completed)
- Messages between agents and the system
- Results from each agent's analysis
- Errors that occurred and how many retries were attempted
- Timing information (when things started/finished)

WHY THIS IS IMPORTANT:
-----------------------
- Allows the system to remember what it's doing
- Enables recovery if something fails (can restart from saved state)
- Provides debugging information (what happened when)
- Supports complex workflows with multiple steps
- Tracks performance metrics

REAL-WORLD ANALOGY:
--------------------
Think of this like a doctor's chart for a patient:
- Patient info (user request)
- Treatment plan (workflow)
- Doctor notes (agent results)
- Progress updates (status changes)
- Lab results (timing and metrics)
"""

from typing import Annotated, List, Dict, Any, Optional, TypedDict, Literal
from datetime import datetime
import operator
import uuid

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages


def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dicts — used as reducer so agent results accumulate instead of replacing."""
    return {**a, **b}

from .request_models import (
    FinancialRequest, StockRequest, TaxRequest, RiskRequest,
    WorkflowType, AgentStatus, AnalysisType, RiskTolerance
)


class AgentState(TypedDict):
    """
    Main Memory Structure for Workflows
    
    This is like the brain of the workflow - it remembers everything
    that's happening and has happened so far.
    
    WHAT EACH FIELD MEANS:
    request: What the user asked for (stocks, taxes, etc.)
    workflow_id: Unique identifier for this specific workflow
    workflow_type: What kind of analysis we're doing
    current_step: Which step we're on (1 of 5, etc.)
    total_steps: How many total steps in this workflow
    status: Current state (pending, running, completed, failed)
    messages: Conversation history between agents
    agent_results: What each agent found out
    current_agent: Which agent is working right now
    started_at: When the workflow started
    last_updated: Last time something changed
    completed_at: When the workflow finished (if completed)
    errors: List of any errors that occurred
    retry_count: How many times we've tried to recover
    max_retries: Maximum retry attempts allowed
    """
    
    # Core request data
    request: FinancialRequest
    
    # Workflow metadata
    workflow_id: str
    workflow_type: WorkflowType
    current_step: int
    total_steps: int
    status: AgentStatus
    
    # Message history for LangGraph — add_messages reducer appends instead of replacing
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Agent execution tracking — merge_dicts reducer merges instead of replacing
    agent_results: Annotated[Dict[str, Any], merge_dicts]
    current_agent: Optional[str]
    
    # Timing and metadata
    started_at: datetime
    last_updated: datetime
    completed_at: Optional[datetime]
    
    # Error handling — operator.add reducer appends errors instead of replacing
    errors: Annotated[List[Dict[str, Any]], operator.add]
    retry_count: int
    
    # Configuration
    max_retries: int
    timeout_seconds: float


class EquityAnalysisState(TypedDict):
    """State specific to equity analysis workflow."""
    
    # Base state
    base: AgentState
    
    # Equity-specific data
    stocks: List[StockRequest]
    analyses: List[Dict[str, Any]]
    portfolio_recommendation: str
    confidence_score: float


class TaxConsultationState(TypedDict):
    """State specific to tax consultation workflow."""
    
    # Base state
    base: AgentState
    
    # Tax-specific data
    tax_request: TaxRequest
    tax_analysis: Dict[str, Any]
    opportunities: List[Dict[str, Any]]
    recommendations: List[str]


class RiskAssessmentState(TypedDict):
    """State specific to risk assessment workflow."""
    
    # Base state
    base: AgentState
    
    # Risk-specific data
    risk_request: RiskRequest
    risk_metrics: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    stress_tests: Dict[str, Any]


class ComprehensiveAnalysisState(TypedDict):
    """State for comprehensive analysis combining all agents."""
    
    # Base state
    base: AgentState
    
    # Component states
    equity_state: Optional[EquityAnalysisState]
    tax_state: Optional[TaxConsultationState]
    risk_state: Optional[RiskAssessmentState]
    
    # Aggregated results
    final_recommendations: List[str]
    overall_confidence: float


# State creation utility
def create_initial_state(request: FinancialRequest) -> AgentState:
    """Create initial workflow state."""
    workflow_id = str(uuid.uuid4())
    workflow_type = request.determine_workflow_type()
    
    # Calculate total steps for workflow type
    steps_map = {
        WorkflowType.EQUITY_ANALYSIS: 1,
        WorkflowType.TAX_CONSULTATION: 1,
        WorkflowType.RISK_ASSESSMENT: 1,
        WorkflowType.PORTFOLIO_REVIEW: 2,
        WorkflowType.COMPREHENSIVE_ANALYSIS: 3
    }
    total_steps = steps_map.get(workflow_type, 1)
    
    return {
        "request": request,
        "workflow_id": workflow_id,
        "workflow_type": workflow_type,
        "current_step": 0,
        "total_steps": total_steps,
        "status": AgentStatus.PENDING,
        "messages": [HumanMessage(content=f"Process request for {request.user_id}")],
        "agent_results": {},
        "current_agent": None,
        "started_at": datetime.now(),
        "last_updated": datetime.now(),
        "completed_at": None,
        "errors": [],
        "retry_count": 0,
        "max_retries": 5,
        "timeout_seconds": 60.0
    }


# State validation utility
def validate_state(state: AgentState) -> List[str]:
    """Validate state and return list of errors."""
    errors = []
    
    if not state.get("workflow_id"):
        errors.append("Missing workflow_id")
    
    if not state.get("workflow_type"):
        errors.append("Missing workflow_type")
    
    if not state.get("request"):
        errors.append("Missing request")
    
    return errors
