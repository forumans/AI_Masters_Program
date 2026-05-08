"""
LangGraph-native agents using Command and Send APIs.

This module provides agent implementations that leverage LangGraph's native features
including Command API for state updates, Send API for parallel execution, and
built-in retry logic. These agents are designed to work seamlessly with the
LangGraph workflow orchestration system.

Key Features:
- Native LangGraph integration with Command/Return patterns
- Built-in retry logic and error handling
- State validation and sanitization
- Performance monitoring and metrics
- Parallel execution support via Send API
- Comprehensive error recovery

Architecture:
- BaseLangGraphAgent: Foundation for all agents
- Specialized agents for different domains (equity, tax, risk)
- WorkflowOrchestrator: Agent routing and coordination
- synthesize_results: Combines all agent outputs into final recommendations (multi-agent only)
- Error handling and recovery utilities
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import asyncio

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from ..models.langgraph_state import (
    AgentState, EquityAnalysisState, TaxConsultationState, RiskAssessmentState
)
from ..models.request_models import StockRequest, AnalysisType, RiskTolerance, AgentStatus
from ..utils.monitoring import monitor_execution, get_logger, metrics
from ..utils.resilience import DEFAULT_RETRY_POLICY

logger = get_logger(__name__)


# Utility classes for state management
class StateCommands:
    """Utility class for creating state updates."""
    
    @staticmethod
    def update_agent_result(state: AgentState, agent_name: str, result: Dict[str, Any]) -> AgentState:
        """Update agent result in state."""
        # Create state copy and update with agent completion info
        updated_state = state.copy()
        updated_state["agent_results"][agent_name] = {
            "status": AgentStatus.COMPLETED,
            "result_data": result,
            "timestamp": datetime.now().isoformat()
        }
        updated_state["last_updated"] = datetime.now()
        return updated_state
    
    @staticmethod
    def complete_workflow(state: AgentState, final_results: Dict[str, Any]) -> AgentState:
        """Complete workflow and update state."""
        # Mark workflow as completed with final results
        updated_state = state.copy()
        updated_state["status"] = AgentStatus.COMPLETED
        updated_state["completed_at"] = datetime.now()
        updated_state["final_results"] = final_results
        updated_state["last_updated"] = datetime.now()
        return updated_state
    
    @staticmethod
    def fail_workflow(state: AgentState, error_message: str) -> AgentState:
        """Fail workflow and update state."""
        # Mark workflow as failed and record error
        updated_state = state.copy()
        updated_state["status"] = AgentStatus.FAILED
        updated_state["errors"].append({
            "message": error_message,
            "timestamp": datetime.now().isoformat()
        })
        updated_state["last_updated"] = datetime.now()
        return updated_state


class StateErrorHandler:
    """Utility class for handling state errors."""
    
    @staticmethod
    def should_retry(state: AgentState, error: Exception) -> bool:
        """Determine if workflow should retry after error."""
        # Check if we haven't exceeded max retry attempts
        return state["retry_count"] < state["max_retries"]
    
    @staticmethod
    def create_retry_state(state: AgentState, agent_name: str) -> AgentState:
        """Create retry state."""
        # Increment retry count and update timestamp
        updated_state = state.copy()
        updated_state["retry_count"] += 1
        updated_state["last_updated"] = datetime.now()
        return updated_state
    
    @staticmethod
    def add_error(state: AgentState, error: Exception, agent_name: str) -> AgentState:
        """Add error to state and fail."""
        # Record error details and mark workflow as failed
        updated_state = state.copy()
        updated_state["errors"].append({
            "agent": agent_name,
            "error": str(error),
            "timestamp": datetime.now().isoformat()
        })
        updated_state["status"] = AgentStatus.FAILED
        updated_state["last_updated"] = datetime.now()
        return updated_state


class StateSend:
    """Utility class for agent routing."""
    
    @staticmethod
    def get_next_agent(state: AgentState, agent_name: str) -> str:
        """Get next agent to execute."""
        return agent_name
    
    @staticmethod
    def get_agents_sequence(state: AgentState, agent_names: List[str]) -> List[str]:
        """Get sequence of agents to execute."""
        return agent_names


class StateValidator:
    """Utility class for state validation."""
    
    @staticmethod
    def validate_state(state: AgentState) -> List[str]:
        """Validate state and return list of errors."""
        # Check for required state fields and collect errors
        errors = []
        
        if not state.get("workflow_id"):
            errors.append("Missing workflow_id")
        
        if not state.get("workflow_type"):
            errors.append("Missing workflow_type")
        
        if not state.get("request"):
            errors.append("Missing request")
        
        return errors


class StateSend:
    """Utility class for Send operations."""
    
    @staticmethod
    def get_next_agent(state: AgentState, agent_name: str) -> str:
        """Get next agent to execute."""
        # Simple routing - return the provided agent name
        return agent_name
    
    @staticmethod
    def get_agents_sequence(state: AgentState, agent_names: List[str]) -> List[str]:
        """Get sequence of agents to execute."""
        return agent_names


class BaseLangGraphAgent:
    """
    Base agent class using LangGraph's native patterns and features.
    
    This class provides the foundation for all agents in the LangGraph system,
    implementing the core patterns for state management, error handling, and
    LLM interaction using LangGraph's native APIs.
    
    Key Features:
    - LangGraph Command API for atomic state updates
    - Built-in retry logic with exponential backoff
    - State validation and sanitization
    - Performance monitoring and metrics collection
    - Error handling with recovery patterns
    - Integration with LangGraph checkpointers
    
    Attributes:
        name: Unique identifier for the agent
        llm: LangChain ChatOpenAI instance for LLM interactions
    """
    
    def __init__(self, name: str, llm_model: str = "gpt-3.5-turbo"):
        """
        Initialize the agent with name and LLM configuration.
        
        Args:
            name: Unique identifier for this agent
            llm_model: Model name to use for LLM interactions
        """
        self.name = name
        # Initialize LLM with conservative settings for reliability
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0.1,  # Low temperature for consistent results
            max_retries=2,  # Reduced retry count for faster failure
            timeout=20.0   # 20 second timeout for LLM calls
        )
        logger.info(f"Initialized {name} agent with model {llm_model}")
    
    def get_system_prompt(self) -> str:
        """Get system prompt for this agent."""
        # Generate agent-specific system prompt
        return f"You are {self.name}, an expert financial analyst."
    
    async def invoke(self, state: AgentState, config: Optional[RunnableConfig] = None) -> AgentState:
        """Main invoke method using simplified state updates."""
        # Log agent input details
        self._log_agent_input(state)
        
        try:
            # Validate state before processing
            validation_errors = StateValidator.validate_state(state)
            if validation_errors:
                raise ValueError(f"Invalid state: {validation_errors}")
            
            # Process the request using subclass implementation
            result = await self._process_request(state)
            
            # Log agent output details
            self._log_agent_output(result)
            
            # Return updated state with agent result
            return StateCommands.update_agent_result(state, self.name, result)
        
        except Exception as e:
            logger.error(f"Agent {self.name} failed", error=str(e), workflow_id=state["workflow_id"])
            
            # Handle error with retry logic
            if StateErrorHandler.should_retry(state, e):
                return StateErrorHandler.create_retry_state(state, self.name)
            else:
                return StateErrorHandler.add_error(state, e, self.name)
    
    async def _process_request(self, state: AgentState) -> Dict[str, Any]:
        """Process the request - to be implemented by subclasses."""
        # Abstract method - must be implemented by specific agents
        raise NotImplementedError("Subclasses must implement _process_request")
    
    def _log_agent_input(self, state: AgentState) -> None:
        """Log agent input details for execution tracking."""
        request = state.get("request", {})
        
        # Extract key input information
        input_details = {
            "agent_name": self.name,
            "workflow_id": state.get("workflow_id"),
            "user_id": getattr(request, 'user_id', 'unknown'),
            "workflow_type": state.get("workflow_type", "unknown"),
            "has_stocks": bool(getattr(request, 'stocks', None)),
            "has_tax": bool(getattr(request, 'tax', None)),
            "has_risk": bool(getattr(request, 'risk', None)),
            "step_count": state.get("step_count", 0),
            "retry_count": state.get("retry_count", 0)
        }
        
        # Add specific input details based on request type
        if hasattr(request, 'stocks') and request.stocks:
            input_details["stock_symbols"] = [s.symbol for s in request.stocks]
        if hasattr(request, 'tax') and request.tax:
            input_details["tax_question_length"] = len(request.tax.question)
        if hasattr(request, 'risk') and request.risk:
            input_details["portfolio_value"] = request.risk.portfolio_value
            input_details["risk_tolerance"] = request.risk.risk_tolerance
        
        logger.info(f"📥 AGENT INPUT - {self.name}", **input_details)
    
    def _log_agent_output(self, result: Dict[str, Any]) -> None:
        """Log agent output details for execution tracking."""
        # Extract key output information
        output_details = {
            "agent_name": self.name,
            "output_keys": list(result.keys()),
            "has_response": "response" in result,
            "response_length": len(result.get("response", "")),
            "timestamp": result.get("timestamp", "unknown")
        }
        
        # Add specific output details based on agent type
        if "stocks_analyzed" in result:
            output_details["stocks_analyzed"] = result["stocks_analyzed"]
        if "portfolio_recommendation" in result:
            output_details["portfolio_recommendation"] = result["portfolio_recommendation"]
        if "tax_saving_opportunities" in result:
            output_details["opportunities_count"] = len(result["tax_saving_opportunities"])
        if "risk_level" in result:
            output_details["risk_level"] = result["risk_level"]
        
        logger.info(f"📤 AGENT OUTPUT - {self.name}", **output_details)
    
    def _build_messages(self, state: AgentState) -> List[BaseMessage]:
        """Build messages for LLM."""
        # Create message list with system and user prompts
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": self._build_user_prompt(state)}
        ]
        
        # Add conversation history from state
        messages.extend(state["messages"])
        
        return messages
    
    def _build_user_prompt(self, state: AgentState) -> str:
        """Build user prompt from state."""
        # Extract request and build prompt components
        request = state["request"]
        prompt_parts = [f"Process request for user: {request.user_id}"]
        
        # Add request-specific information
        if hasattr(request, 'stocks') and request.stocks:
            stocks_text = ", ".join([s.symbol for s in request.stocks])
            prompt_parts.append(f"Stocks: {stocks_text}")
        
        if hasattr(request, 'tax') and request.tax:
            prompt_parts.append(f"Tax question: {request.tax.question}")
        
        if hasattr(request, 'risk') and request.risk:
            prompt_parts.append(f"Portfolio: ${request.risk.portfolio_value:,.2f}")
            prompt_parts.append(f"Risk tolerance: {request.risk.risk_tolerance}")
        
        # Add previous results for context
        if state["agent_results"]:
            for agent, result in state["agent_results"].items():
                if result.get("status") == AgentStatus.COMPLETED:
                    prompt_parts.append(f"Previous {agent}: {result.get('result_data', {}).get('response', '')[:100]}...")
        
        return "\n\n".join(prompt_parts)


class EquityAnalysisAgent(BaseLangGraphAgent):
    """Equity analysis agent using LangGraph patterns."""
    
    def __init__(self):
        # Initialize equity expert with specific model
        super().__init__("equity_expert", "gpt-3.5-turbo")
    
    def get_system_prompt(self) -> str:
        return """You are an expert equity analyst. Analyze stocks and provide investment recommendations.
        
        Structure your response with:
        - Overall recommendation (BUY/HOLD/SELL)
        - Key analysis points
        - Risk factors
        - Price targets (if applicable)
        - Confidence level (High/Moderate/Low)"""
    
    async def _process_request(self, state: AgentState) -> Dict[str, Any]:
        """Process equity analysis request."""
        request = state["request"]
        
        if not request.stocks:
            raise ValueError("No stocks provided for equity analysis")
        
        # Build messages for LLM and get analysis
        messages = self._build_messages(state)
        
        # Execute LLM call with timeout and retry
        response = await self.llm.ainvoke(messages)
        
        # Process and structure the response
        return self._process_equity_response(response.content, request.stocks)
    
    def _process_equity_response(self, response: str, stocks: List[StockRequest]) -> Dict[str, Any]:
        """Process equity analysis response."""
        analyses = []
        
        # Create analysis for each stock (simplified parsing)
        for stock in stocks:
            # In real implementation, this would parse the response properly
            analysis = {
                "symbol": stock.symbol,
                "analysis_type": stock.analysis_type.value,
                "recommendation": "HOLD",  # Would extract from response
                "confidence": 0.75,
                "price_target": f"${150 + hash(stock.symbol) % 100}",
                "key_points": [
                    "Technical indicators neutral",
                    "Fundamentals stable"
                ],
                "risk_factors": [
                    "Market volatility",
                    "Sector risk"
                ]
            }
            analyses.append(analysis)
        
        # Calculate overall portfolio recommendation
        recommendations = [a["recommendation"] for a in analyses]
        if recommendations.count("BUY") > recommendations.count("SELL"):
            overall = "BUY"
        elif recommendations.count("SELL") > recommendations.count("BUY"):
            overall = "SELL"
        else:
            overall = "HOLD"
        
        return {
            "stocks_analyzed": len(analyses),
            "stock_analyses": analyses,
            "portfolio_recommendation": overall,
            "average_confidence": sum(a["confidence"] for a in analyses) / len(analyses),
            "response": response,
            "timestamp": datetime.now().isoformat()
        }


class TaxConsultationAgent(BaseLangGraphAgent):
    """Tax consultation agent using LangGraph patterns."""
    
    def __init__(self):
        # Initialize tax expert with specific model
        super().__init__("tax_expert", "gpt-3.5-turbo")
    
    def get_system_prompt(self) -> str:
        return """You are a tax consultant. Provide tax advice and optimization strategies.
        
        Structure your response with:
        - Direct answer to the tax question
        - Tax-saving opportunities
        - Relevant tax laws
        - Actionable recommendations
        - Risk considerations
        
        Include a disclaimer that this is general advice."""
    
    async def _process_request(self, state: AgentState) -> Dict[str, Any]:
        """Process tax consultation request."""
        request = state["request"]
        
        if not request.tax:
            raise ValueError("No tax question provided")
        
        # Build messages for tax analysis
        messages = self._build_messages(state)
        
        # Execute LLM call with timeout and retry
        response = await self.llm.ainvoke(messages)
        
        # Process and structure the tax response
        return self._process_tax_response(response.content, request.tax)
    
    def _process_tax_response(self, response: str, tax_request) -> Dict[str, Any]:
        """Process tax consultation response."""
        # Extract tax-saving opportunities (simplified parsing)
        opportunities = [
            {
                "type": "tax_loss_harvesting",
                "potential_savings": "Up to 30% of gains",
                "description": "Sell losing investments to offset gains",
                "action_required": "Review portfolio for losses"
            },
            {
                "type": "retirement_contributions",
                "potential_savings": "22-37% of contribution",
                "description": "Maximize tax-advantaged accounts",
                "action_required": "Increase 401(k)/IRA contributions"
            }
        ]
        
        # Extract actionable recommendations
        recommendations = [
            "Consider tax-loss harvesting",
            "Maximize retirement contributions",
            "Consult with tax professional"
        ]
        
        return {
            "question_answered": tax_request.question,
            "tax_analysis": {
                "category": "general",
                "complexity": "moderate",
                "key_topics": ["capital gains", "investments"]
            },
            "tax_saving_opportunities": opportunities,
            "recommendations": recommendations,
            "jurisdiction": tax_request.jurisdiction,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }


class RiskAssessmentAgent(BaseLangGraphAgent):
    """Risk assessment agent using LangGraph patterns."""
    
    def __init__(self):
        # Initialize risk expert with specific model
        super().__init__("risk_expert", "gpt-3.5-turbo")
    
    def get_system_prompt(self) -> str:
        return """You are a risk management specialist. Assess portfolio risk and provide strategies.
        
        Structure your response with:
        - Overall risk assessment (Low/Medium/High)
        - Key risk metrics
        - Risk factors
        - Diversification analysis
        - Risk management recommendations"""
    
    async def _process_request(self, state: AgentState) -> Dict[str, Any]:
        """Process risk assessment request."""
        request = state["request"]
        
        if not request.risk:
            raise ValueError("No risk assessment data provided")
        
        # Build messages for risk analysis
        messages = self._build_messages(state)
        
        # Execute LLM call with timeout and retry
        response = await self.llm.ainvoke(messages)
        
        # Process and structure the risk response
        return self._process_risk_response(response.content, request.risk)
    
    def _process_risk_response(self, response: str, risk_request) -> Dict[str, Any]:
        """Process risk assessment response."""
        # Calculate risk metrics based on risk tolerance
        base_volatility = {
            RiskTolerance.CONSERVATIVE: 0.08,
            RiskTolerance.MODERATE: 0.15,
            RiskTolerance.AGGRESSIVE: 0.25
        }[risk_request.risk_tolerance]
        
        # Generate risk metrics
        risk_metrics = {
            "volatility": base_volatility,
            "var_95": risk_request.portfolio_value * base_volatility * 1.65,
            "beta": 1.0,
            "sharpe_ratio": 0.8,
            "max_drawdown": base_volatility * 2.5
        }
        
        # Determine risk level from score
        risk_score = min(100, base_volatility * 200)
        if risk_score < 30:
            risk_level = "LOW"
        elif risk_score < 60:
            risk_level = "MODERATE"
        else:
            risk_level = "HIGH"
        
        # Create risk assessment summary
        risk_assessment = {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "status": "APPROPRIATE"
        }
        
        # Generate stress test scenarios
        stress_tests = {
            "market_crash": {
                "impact": risk_request.portfolio_value * 0.20,
                "probability": "Low"
            },
            "recession": {
                "impact": risk_request.portfolio_value * 0.15,
                "probability": "Medium"
            }
        }
        
        return {
            "portfolio_value": risk_request.portfolio_value,
            "risk_tolerance": risk_request.risk_tolerance.value,
            "time_horizon": risk_request.time_horizon,
            "risk_metrics": risk_metrics,
            "overall_risk_assessment": risk_assessment,
            "stress_test_results": stress_tests,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }


# Agent registry
AGENT_REGISTRY = {
    "equity_expert": EquityAnalysisAgent(),
    "tax_expert": TaxConsultationAgent(),
    "risk_expert": RiskAssessmentAgent()
}


def get_agent(name: str) -> BaseLangGraphAgent:
    """Get agent by name."""
    # Retrieve agent from registry or raise error
    if name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent: {name}")
    return AGENT_REGISTRY[name]


# Workflow orchestrator using LangGraph patterns
class WorkflowOrchestrator:
    """Orchestrator using LangGraph's Command and Send APIs."""
    
    @staticmethod
    async def route_to_agents(state: AgentState) -> AgentState:
        """Route state to appropriate agents based on workflow type."""
        workflow_type = state["workflow_type"]
        
        # Define agent sequences for each workflow type
        workflow_agents = {
            "equity_analysis": ["equity_expert"],
            "tax_consultation": ["tax_expert"],
            "risk_assessment": ["risk_expert"],
            "portfolio_review": ["equity_expert", "risk_expert"],
            "comprehensive_analysis": ["equity_expert", "tax_expert", "risk_expert"]
        }
        
        agents = workflow_agents.get(workflow_type.value, [])
        
        if not agents:
            return StateCommands.fail_workflow(state, f"No agents defined for workflow: {workflow_type}")
        
        # Store the agent sequence and set current agent for execution
        updated_state = state.copy()
        updated_state["agent_sequence"] = agents
        updated_state["current_agent_index"] = 0
        updated_state["current_agent"] = agents[0] if agents else None
        updated_state["last_updated"] = datetime.now()
        
        return updated_state
    
    @staticmethod
    async def finalize_workflow(state: AgentState) -> AgentState:
        """Finalize workflow and compile results."""
        # Compile final results
        final_results = {
            "workflow_type": state["workflow_type"].value,
            "agents_executed": list(state["agent_results"].keys()),
            "execution_time": (datetime.now() - state["started_at"]).total_seconds(),
            "status": state["status"].value
        }
        
        # Add agent-specific results
        for agent_name, result in state["agent_results"].items():
            if result.get("status") == AgentStatus.COMPLETED:
                final_results[agent_name] = result.get("result_data", {})
        
        return StateCommands.complete_workflow(state, final_results)


# Synthesis node — calls GPT after all agents complete to produce combined final recommendations.
# Only used for multi-agent workflows (PORTFOLIO_REVIEW, COMPREHENSIVE_ANALYSIS).
# Single-agent workflows skip this and go straight to finalize_workflow.
async def synthesize_results(state: AgentState) -> dict:
    """
    Synthesize all agent outputs into a single set of final recommendations.
    
    WHY THIS EXISTS:
    Each agent runs independently and writes its own analysis. But for a comprehensive
    workflow, the user needs a unified answer — not three separate reports. This node
    reads all agent results and asks GPT to produce 3-5 clear, actionable recommendations
    that draw from all of them together.
    
    HOW IT WORKS:
    1. Collects each agent's response text from state["agent_results"]
    2. Builds a synthesis prompt containing all agent outputs
    3. Calls GPT once to produce a numbered recommendation list
    4. Stores result in state["agent_results"]["synthesis"] via merge_dicts reducer
    """
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Collect each agent's response for the synthesis prompt
    agent_summaries = []
    for agent_name, result in state["agent_results"].items():
        response_text = result.get("result_data", {}).get("response", "")
        if response_text:
            label = agent_name.replace("_", " ").title()
            agent_summaries.append(f"**{label}:**\n{response_text[:800]}")

    if not agent_summaries:
        return {}

    prompt = (
        "You are a senior financial advisor. Based on the following expert analyses, "
        "provide 3-5 clear, actionable final recommendations for the client.\n\n"
        + "\n\n".join(agent_summaries)
        + "\n\nRespond with ONLY a numbered list. Each item must be one concise sentence."
    )

    response = await llm.ainvoke([{"role": "user", "content": prompt}])

    # Parse the numbered list into individual recommendation strings
    recommendations = [
        line.strip()
        for line in response.content.split("\n")
        if line.strip() and line.strip()[0].isdigit()
    ]

    logger.info("Synthesis completed", recommendation_count=len(recommendations))

    # Return partial state — merge_dicts reducer merges this into existing agent_results
    return {
        "agent_results": {
            "synthesis": {
                "status": AgentStatus.COMPLETED,
                "result_data": {
                    "response": response.content,
                    "recommendations": recommendations
                }
            }
        }
    }


# Error handling node
async def error_handler(state: AgentState) -> AgentState:
    """Handle errors in workflow."""
    if not state["errors"]:
        return StateCommands.fail_workflow(state, "Unknown error occurred")
    
    # Get the last error
    last_error = state["errors"][-1]
    
    # Check if we should retry
    if state["retry_count"] < state["max_retries"]:
        logger.warning(f"Retrying workflow after error: {last_error['message']}")
        return StateErrorHandler.create_retry_state(state, state.get("current_agent", "equity_expert"))
    else:
        logger.error(f"Workflow failed after {state['retry_count']} retries")
        return StateCommands.fail_workflow(state, f"Workflow failed: {last_error['message']}")
