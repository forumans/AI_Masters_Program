"""
Data Models - The Blueprint for All Requests and Responses

PURPOSE:
--------
This file defines the structure for all data that flows through the system.
Think of these as blueprints or templates that ensure data is consistent
and valid throughout the multi-agent system.

WHAT IT DEFINES:
----------------
- Types of analysis (technical, fundamental, both)
- Risk levels (conservative, moderate, aggressive)
- Workflow types (what kind of work to do)
- Agent status (pending, running, completed, etc.)
- Request structures (what users can ask for)
- Response structures (what agents return)

WHY THIS IS IMPORTANT:
---------------------
- Ensures data is always in the right format
- Prevents errors from bad data
- Makes the system predictable and reliable
- Provides clear contracts between components

EXAMPLE:
--------
When a user wants to analyze a stock, they send:
{
    "symbol": "AAPL",
    "analysis_type": "technical"
}

The system validates this matches the StockRequest blueprint
before sending it to the equity agent.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Literal
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class AnalysisType(str, Enum):
    """
    Types of Stock Analysis Available
    
    TECHNICAL: Looks at charts, patterns, trading volumes
    FUNDAMENTAL: Looks at company finances, earnings, management  
    BOTH: Combines both types for complete analysis
    """
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    BOTH = "both"


class RiskTolerance(str, Enum):
    """
    How Much Risk the User is Comfortable With
    
    CONSERVATIVE: Low risk, prefers safe investments
    MODERATE: Balanced risk, accepts some volatility
    AGGRESSIVE: High risk, seeks high returns
    """
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class WorkflowType(str, Enum):
    """
    Available Types of Financial Analysis
    
    EQUITY_ANALYSIS: Analyze individual stocks
    TAX_CONSULTATION: Provide tax advice and planning
    RISK_ASSESSMENT: Evaluate portfolio risk
    PORTFOLIO_REVIEW: Review entire investment portfolio
    COMPREHENSIVE_ANALYSIS: All of the above combined
    """
    EQUITY_ANALYSIS = "equity_analysis"
    TAX_CONSULTATION = "tax_consultation"
    RISK_ASSESSMENT = "risk_assessment"
    PORTFOLIO_REVIEW = "portfolio_review"
    COMPREHENSIVE_ANALYSIS = "comprehensive_analysis"


class AgentStatus(str, Enum):
    """
    Current Status of Agent Execution
    
    PENDING: Waiting to start
    RUNNING: Currently working
    COMPLETED: Finished successfully
    FAILED: Encountered an error
    CANCELLED: Stopped before completion
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StockRequest(BaseModel):
    """
    Request for Stock Analysis
    
    WHAT IT CONTAINS:
    - symbol: Stock ticker (e.g., "AAPL", "GOOGL")
    - analysis_type: What kind of analysis to perform
    
    VALIDATION:
    - Symbol must be 1-10 capital letters
    - Automatically converts symbol to uppercase
    """
    symbol: str = Field(..., min_length=1, max_length=10, pattern="^[A-Z]+$")
    analysis_type: AnalysisType = AnalysisType.TECHNICAL
    
    @validator('symbol')
    def normalize_symbol(cls, v):
        """Convert stock symbol to uppercase for consistency."""
        return v.upper()


class TaxRequest(BaseModel):
    """Tax consultation request."""
    question: str = Field(..., min_length=10, max_length=1000)
    jurisdiction: str = "US"
    year: int = Field(default=2023, ge=2020, le=2030)


class RiskRequest(BaseModel):
    """Risk assessment request."""
    portfolio_value: float = Field(..., gt=0)
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE
    time_horizon: int = Field(..., gt=0, le=50)
    current_holdings: Optional[List[str]] = None


class FinancialRequest(BaseModel):
    """Main financial request model."""
    user_id: str = Field(..., min_length=1)
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    stocks: Optional[List[StockRequest]] = None
    tax: Optional[TaxRequest] = None
    risk: Optional[RiskRequest] = None
    created_at: datetime = Field(default_factory=datetime.now)
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if not v or v.strip() == "":
            raise ValueError("User ID cannot be empty")
        return v.strip()
    
    def has_stocks(self) -> bool:
        """Check if request has stock analysis."""
        return self.stocks is not None and len(self.stocks) > 0
    
    def has_tax_question(self) -> bool:
        """Check if request has tax question."""
        return self.tax is not None
    
    def has_risk_assessment(self) -> bool:
        """Check if request has risk assessment."""
        return self.risk is not None
    
    def determine_workflow_type(self) -> WorkflowType:
        """Determine workflow type based on request components."""
        has_stocks = self.has_stocks()
        has_tax = self.has_tax_question()
        has_risk = self.has_risk_assessment()
        
        if has_stocks and has_tax and has_risk:
            return WorkflowType.COMPREHENSIVE_ANALYSIS
        elif has_stocks and has_risk:
            return WorkflowType.PORTFOLIO_REVIEW
        elif has_stocks:
            return WorkflowType.EQUITY_ANALYSIS
        elif has_tax:
            return WorkflowType.TAX_CONSULTATION
        elif has_risk:
            return WorkflowType.RISK_ASSESSMENT
        else:
            raise ValueError("At least one component must be specified")


class AgentResult(BaseModel):
    """Result from an agent execution."""
    agent_name: str
    status: AgentStatus
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class WorkflowState(BaseModel):
    """Current state of workflow execution."""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request: FinancialRequest
    workflow_type: WorkflowType
    status: AgentStatus = AgentStatus.PENDING
    current_step: int = 0
    total_steps: int = 0
    agent_results: List[AgentResult] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def add_agent_result(self, result: AgentResult):
        """Add agent result to workflow state."""
        self.agent_results.append(result)
        
    def get_agent_result(self, agent_name: str) -> Optional[AgentResult]:
        """Get result for specific agent."""
        for result in self.agent_results:
            if result.agent_name == agent_name:
                return result
        return None
    
    def get_failed_agents(self) -> List[str]:
        """Get list of failed agents."""
        return [r.agent_name for r in self.agent_results if r.status == AgentStatus.FAILED]
    
    def is_complete(self) -> bool:
        """Check if workflow is complete."""
        return self.status in [AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.CANCELLED]
    
    def calculate_execution_time(self) -> Optional[float]:
        """Calculate total execution time."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class LangGraphState(BaseModel):
    """State for LangGraph workflow execution."""
    request: FinancialRequest
    workflow_state: WorkflowState
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    current_agent: Optional[str] = None
    step_count: int = 0
    max_steps: int = 10  # Prevent infinite loops
    
    def add_message(self, role: str, content: str, agent: Optional[str] = None):
        """Add a message to the state."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "agent": agent
        }
        self.messages.append(message)
    
    def increment_step(self):
        """Increment step counter."""
        self.step_count += 1
        if self.step_count > self.max_steps:
            raise ValueError(f"Maximum steps ({self.max_steps}) exceeded")


class SystemHealth(BaseModel):
    """System health status."""
    status: Literal["healthy", "degraded", "unhealthy"]
    timestamp: datetime = Field(default_factory=datetime.now)
    checks: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    active_workflows: int = 0
    error_rate: float = 0.0
    avg_response_time: float = 0.0


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    error_type: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class SuccessResponse(BaseModel):
    """Success response model."""
    status: str = "success"
    workflow_id: str
    workflow_type: WorkflowType
    results: Dict[str, Any]
    execution_time: float
    agent_results: List[AgentResult]
    timestamp: datetime = Field(default_factory=datetime.now)
