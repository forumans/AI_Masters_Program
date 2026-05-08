"""
Deterministic Router - Smart Traffic Director for Multi-Agent System

PURPOSE:
--------
This file replaces unreliable AI-based routing with predictable, rule-based routing.
Instead of asking an AI which agent to use, we use clear rules to decide.

WHY THIS IS BETTER:
-------------------
- No more infinite loops from AI confusion
- Always predictable behavior
- Much faster (no AI calls needed)
- Easier to debug and understand
- No random failures from AI hallucinations

HOW IT WORKS:
------------
1. User sends a request (e.g., stock analysis)
2. Router checks the request type
3. Uses predefined rules to select agents
4. Creates a workflow with those agents
5. Executes workflow in the right order

EXAMPLE:
--------
User asks: "Analyze AAPL stock"
Router rule: If request has stocks → Use equity_agent
User asks: "Help with taxes"  
Router rule: If request has tax question → Use tax_agent
User asks: "Review my portfolio"
Router rule: If request has multiple stocks → Use equity + risk agents
"""

from typing import Dict, Any, List, Literal, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor

from ..models.request_models import WorkflowType, FinancialRequest, LangGraphState
from ..agents.equity_agent import equity_agent
from ..agents.tax_agent import tax_agent
from ..agents.risk_agent import risk_agent
from ..utils.monitoring import get_logger, metrics

logger = get_logger(__name__)


@dataclass
class WorkflowStep:
    """
    Single Step in a Workflow
    
    WHAT IT DEFINES:
    - agent_name: Which agent to call (equity_agent, tax_agent, etc.)
    - required: Must this step succeed for workflow to continue?
    - depends_on: Which steps must finish before this one can start
    - condition: Special condition to decide if this step should run
    - timeout: How long to wait before giving up on this step
    - retry_count: How many times to retry if this step fails
    
    EXAMPLE:
    step = WorkflowStep(
        agent_name="equity_agent",
        required=True,
        timeout=30.0,
        retry_count=3
    )
    """
    agent_name: str
    required: bool = True
    depends_on: Optional[List[str]] = None
    condition: Optional[str] = None
    timeout: float = 30.0
    retry_count: int = 3
    
    def __post_init__(self):
        # Initialize dependencies list if not provided
        if self.depends_on is None:
            self.depends_on = []


@dataclass
class WorkflowDefinition:
    """
    Complete Workflow Blueprint
    
    WHAT IT DEFINES:
    - workflow_type: What kind of request this handles
    - name: Human-readable name for this workflow
    - description: What this workflow does
    - steps: List of all steps to execute in order
    - max_total_time: Maximum time for entire workflow
    
    EXAMPLE WORKFLOW:
    1. Analyze stocks (equity_agent)
    2. Assess risk (risk_agent) - depends on step 1
    3. Provide tax advice (tax_agent) - optional
    """
    workflow_type: WorkflowType
    name: str
    description: str
    steps: List[WorkflowStep]
    max_total_time: float = 300.0  # 5 minutes max
    
    def get_agent_names(self) -> List[str]:
        """Get all agent names used in this workflow."""
        return [step.agent_name for step in self.steps]
    
    def get_required_agents(self) -> List[str]:
        """Get required agent names."""
        # Filter for required agents only
        return [step.agent_name for step in self.steps if step.required]


# Pre-defined deterministic workflows
WORKFLOW_DEFINITIONS: Dict[WorkflowType, WorkflowDefinition] = {
    WorkflowType.EQUITY_ANALYSIS: WorkflowDefinition(
        workflow_type=WorkflowType.EQUITY_ANALYSIS,
        name="Equity Analysis",
        description="Analyze stocks and provide investment recommendations",
        steps=[
            WorkflowStep(
                agent_name="equity_expert",
                required=True,
                condition="has_stocks"
            )
        ]
    ),
    
    WorkflowType.TAX_CONSULTATION: WorkflowDefinition(
        workflow_type=WorkflowType.TAX_CONSULTATION,
        name="Tax Consultation",
        description="Provide tax advice and optimization strategies",
        steps=[
            WorkflowStep(
                agent_name="tax_expert",
                required=True,
                condition="has_tax_question"
            )
        ]
    ),
    
    WorkflowType.RISK_ASSESSMENT: WorkflowDefinition(
        workflow_type=WorkflowType.RISK_ASSESSMENT,
        name="Risk Assessment",
        description="Assess portfolio risk and provide risk management strategies",
        steps=[
            WorkflowStep(
                agent_name="risk_expert",
                required=True,
                condition="has_risk_assessment"
            )
        ]
    ),
    
    WorkflowType.PORTFOLIO_REVIEW: WorkflowDefinition(
        workflow_type=WorkflowType.PORTFOLIO_REVIEW,
        name="Portfolio Review",
        description="Comprehensive portfolio analysis including equity and risk",
        steps=[
            WorkflowStep(
                agent_name="equity_expert",
                required=True,
                condition="has_stocks"
            ),
            WorkflowStep(
                agent_name="risk_expert",
                required=True,
                condition="has_risk_assessment",
                depends_on=["equity_expert"]
            )
        ]
    ),
    
    WorkflowType.COMPREHENSIVE_ANALYSIS: WorkflowDefinition(
        workflow_type=WorkflowType.COMPREHENSIVE_ANALYSIS,
        name="Comprehensive Financial Analysis",
        description="Complete financial analysis including equity, tax, and risk",
        steps=[
            WorkflowStep(
                agent_name="equity_expert",
                required=True,
                condition="has_stocks"
            ),
            WorkflowStep(
                agent_name="tax_expert",
                required=True,
                condition="has_tax_question",
                depends_on=["equity_expert"]
            ),
            WorkflowStep(
                agent_name="risk_expert",
                required=True,
                condition="has_risk_assessment",
                depends_on=["equity_expert", "tax_expert"]
            )
        ]
    )
}


class DeterministicRouter:
    """Deterministic router that replaces LLM-based routing decisions."""
    
    def __init__(self):
        # Initialize router with predefined workflows and agent mappings
        self.definitions = WORKFLOW_DEFINITIONS
        self.agent_map = {
            "equity_expert": equity_agent,
            "tax_expert": tax_agent,
            "risk_expert": risk_agent
        }
        logger.info("Deterministic router initialized", workflows=list(self.definitions.keys()))
    
    def determine_workflow(self, request: FinancialRequest) -> WorkflowType:
        """Determine workflow type based on request components."""
        # Use request's built-in workflow type determination logic
        workflow_type = request.determine_workflow_type()
        
        logger.info("Workflow determined", 
                   workflow_type=workflow_type.value,
                   has_stocks=request.has_stocks(),
                   has_tax=request.has_tax_question(),
                   has_risk=request.has_risk_assessment())
        
        metrics.increment_counter('workflow_determined', 1, {'workflow_type': workflow_type.value})
        
        return workflow_type
    
    def get_workflow_definition(self, workflow_type: WorkflowType) -> WorkflowDefinition:
        """Get workflow definition by type."""
        # Retrieve predefined workflow or raise error if unknown
        if workflow_type not in self.definitions:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
        
        return self.definitions[workflow_type]
    
    def get_next_agent(self, state: LangGraphState) -> Optional[str]:
        """Get the next agent to execute based on current state."""
        # Determine workflow and get its definition
        workflow_type = state.workflow_state.workflow_type
        definition = self.get_workflow_definition(workflow_type)
        
        # Build set of completed agents from state
        completed_agents = set()
        if state.workflow_state.agent_results:
            for result in state.workflow_state.agent_results:
                if result.status.value == "completed":
                    completed_agents.add(result.agent_name)
        
        # Find next eligible agent to execute
        for step in definition.steps:
            # Skip if already completed
            if step.agent_name in completed_agents:
                continue
            
            # Check if condition is met
            if step.condition:
                if not self._check_condition(step.condition, state):
                    logger.debug("Skipping agent due to unmet condition", 
                                agent=step.agent_name, 
                                condition=step.condition)
                    continue
            
            # Check dependencies are satisfied
            if step.depends_on:
                if not all(dep in completed_agents for dep in step.depends_on):
                    logger.debug("Skipping agent due to unmet dependencies", 
                                agent=step.agent_name, 
                                dependencies=step.depends_on)
                    continue
            
            logger.info("Next agent determined", agent=step.agent_name)
            return step.agent_name
        
        # No more agents to execute
        logger.info("All agents completed or no more agents to execute")
        return None
    
    def _check_condition(self, condition: str, state: LangGraphState) -> bool:
        """Check if a condition is met."""
        # Extract request from state and evaluate condition
        request = state.request
        
        # Map condition names to request methods
        conditions = {
            "has_stocks": request.has_stocks(),
            "has_tax_question": request.has_tax_question(),
            "has_risk_assessment": request.has_risk_assessment()
        }
        
        return conditions.get(condition, False)
    
    def should_continue(self, state: LangGraphState) -> bool:
        """Determine if workflow should continue."""
        # Stop if max steps exceeded
        if state.step_count >= state.max_steps:
            logger.warning("Max steps exceeded", steps=state.step_count, max=state.max_steps)
            return False
        
        # Get workflow definition and check completion status
        workflow_type = state.workflow_state.workflow_type
        definition = self.get_workflow_definition(workflow_type)
        
        # Track completed required agents
        completed_required = set()
        if state.workflow_state.agent_results:
            for result in state.workflow_state.agent_results:
                if result.status.value == "completed":
                    # Check if this was a required agent
                    for step in definition.steps:
                        if step.agent_name == result.agent_name and step.required:
                            completed_required.add(result.agent_name)
        
        required_agents = set(definition.get_required_agents())
        
        # Stop if all required agents completed
        if required_agents.issubset(completed_required):
            logger.info("All required agents completed")
            return False
        
        # Check for failed required agents
        failed_required = set()
        if state.workflow_state.agent_results:
            for result in state.workflow_state.agent_results:
                if result.status.value == "failed":
                    for step in definition.steps:
                        if step.agent_name == result.agent_name and step.required:
                            failed_required.add(result.agent_name)
        
        if failed_required:
            logger.error("Required agents failed", agents=list(failed_required))
            return False
        
        return True
    
    def create_workflow_graph(self, workflow_type: WorkflowType) -> StateGraph:
        """Create a LangGraph workflow based on the definition."""
        # Get workflow definition and create graph
        definition = self.get_workflow_definition(workflow_type)
        
        # Create the graph with our state type
        workflow = StateGraph(LangGraphState)
        
        # Add nodes for each agent in the workflow
        for step in definition.steps:
            agent = self.agent_map.get(step.agent_name)
            if agent:
                workflow.add_node(step.agent_name, agent)
        
        # Add conditional routing function
        def route_decision(state: LangGraphState) -> str:
            """Decide which agent to route to next."""
            next_agent = self.get_next_agent(state)
            
            if next_agent is None:
                return "end"
            
            return next_agent
        
        # Add conditional edges from each agent
        for step in definition.steps:
            workflow.add_conditional_edges(
                step.agent_name,
                route_decision,
                {
                    "equity_expert": "equity_expert",
                    "tax_expert": "tax_expert",
                    "risk_expert": "risk_expert",
                    "end": END
                }
            )
        
        # Set entry point to first agent
        if definition.steps:
            workflow.set_entry_point(definition.steps[0].agent_name)
        
        return workflow
    
    def validate_workflow(self, workflow_type: WorkflowType) -> List[str]:
        """Validate a workflow definition."""
        errors = []
        definition = self.get_workflow_definition(workflow_type)
        
        # Check if all agents exist in our agent map
        for step in definition.steps:
            if step.agent_name not in self.agent_map:
                errors.append(f"Agent {step.agent_name} not found")
        
        # Check for dependencies on non-existent agents
        agent_names = [step.agent_name for step in definition.steps]
        for step in definition.steps:
            if step.depends_on:
                for dep in step.depends_on:
                    if dep not in agent_names:
                        errors.append(f"Agent {step.agent_name} depends on non-existent agent {dep}")
        
        return errors


# Global deterministic router
deterministic_router = DeterministicRouter()


def get_workflow_graph(request: FinancialRequest) -> StateGraph:
    """Get the appropriate workflow graph for a request."""
    # Determine workflow type and create corresponding graph
    workflow_type = deterministic_router.determine_workflow(request)
    return deterministic_router.create_workflow_graph(workflow_type)


def validate_all_workflows() -> Dict[str, List[str]]:
    """Validate all workflow definitions."""
    validation_results = {}
    
    # Validate each workflow type and collect results
    for workflow_type in WorkflowType:
        errors = deterministic_router.validate_workflow(workflow_type)
        validation_results[workflow_type.value] = errors
        
        if errors:
            logger.error("Workflow validation failed", 
                        workflow=workflow_type.value, 
                        errors=errors)
        else:
            logger.info("Workflow validation passed", workflow=workflow_type.value)
    
    return validation_results
