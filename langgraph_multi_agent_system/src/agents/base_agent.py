"""
Base Agent - Foundation for All AI Agents

PURPOSE:
--------
This file defines the base class that all AI agents inherit from. It provides
common functionality so each agent doesn't have to repeat the same code.

WHAT IT PROVIDES:
-----------------
- LLM interaction (talking to OpenAI)
- Protection from failures (circuit breakers, retries, timeouts)
- Performance tracking (how long things take)
- Error handling (what to do when things go wrong)
- Logging (keeping records of what happens)

HOW TO CREATE A NEW AGENT:
-------------------------
1. Inherit from BaseLangGraphAgent
2. Implement the required methods:
   - get_system_prompt(): What the AI should act like
   - validate_input(): Check if request is valid
   - process_request(): Do the actual work
3. Your agent automatically gets all the protection features!

EXAMPLE:
--------
class MyAgent(BaseLangGraphAgent):
    def get_system_prompt(self):
        return "You are a helpful assistant."
    
    def validate_input(self, state):
        return True  # Always accept input
    
    def process_request(self, state):
        return {"response": "Hello!"}
"""

import time
import asyncio
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from ..models.request_models import AgentResult, AgentStatus, StockRequest
from ..utils.resilience import resilient, DEFAULT_RETRY_POLICY
from ..utils.monitoring import monitor_execution, get_logger, metrics

logger = get_logger(__name__)


@dataclass
class AgentConfig:
    """
    Settings for Each AI Agent
    
    PURPOSE:
    --------
    Stores configuration options for each agent. Like a settings file
    for individual AI agents.
    
    WHAT EACH SETTING DOES:
    -----------------------
    name: What to call this agent (for logs and monitoring)
    model: Which OpenAI model to use (gpt-3.5-turbo, gpt-4, etc.)
    temperature: How creative the AI should be (0.0 = serious, 1.0 = creative)
    max_tokens: Maximum length of AI responses
    timeout: How long to wait before giving up (seconds)
    max_retries: How many times to retry if it fails
    circuit_breaker_threshold: How many failures before stopping calls
    circuit_breaker_timeout: How long to wait before trying again
    enable_monitoring: Whether to track performance metrics
    log_level: How much detail to log (DEBUG, INFO, WARNING, ERROR)
    
    EXAMPLE USAGE:
    ---------------
    config = AgentConfig(
        name="equity_expert",
        model="gpt-3.5-turbo",
        temperature=0.1,  # Very analytical
        timeout=30
    )
    """
    name: str
    description: str
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 1000
    temperature: float = 0.1
    timeout: float = 30.0  # seconds
    max_retries: int = 3  # retry attempts
    circuit_breaker_threshold: int = 5  # failures before stopping calls
    circuit_breaker_timeout: float = 60.0  # seconds
    enable_monitoring: bool = True
    log_level: str = "INFO"


class BaseLangGraphAgent(ABC, Runnable):
    """
    Base Class for All AI Agents
    
    PURPOSE:
    --------
    This is the parent class that every AI agent inherits from. It provides
    all the basic functionality so agents can focus on their specific tasks.
    
    WHAT IT PROVIDES:
    -----------------
    - Connection to OpenAI (the AI brain)
    - Protection from failures (circuit breakers, retries, timeouts)
    - Performance tracking (how fast and reliable the agent is)
    - Error handling (what to do when things go wrong)
    - Integration with LangGraph workflows
    
    TRACKED METRICS:
    ----------------
    - execution_count: How many times this agent ran
    - success_count: How many times it succeeded
    - failure_count: How many times it failed
    - total_execution_time: Total time spent running
    
    HOW TO USE:
    -----------
    Inherit from this class and implement these methods:
    - get_system_prompt(): Define the AI's personality
    - validate_input(): Check if the request makes sense
    - process_request(): Do the actual work
    """
    
    def __init__(self, config: AgentConfig):
        """
        Set up the agent with its configuration and AI connection.
        
        WHAT IT DOES:
        - Stores the agent's configuration
        - Creates connection to OpenAI
        - Sets up performance counters
        - Logs that the agent is ready
        
        Args:
            config: Settings for this agent (model, timeout, etc.)
        """
        self.config = config
        # Create OpenAI connection with agent's settings
        self.llm = ChatOpenAI(
            model=config.llm_model,      # Which AI model to use
            max_tokens=config.max_tokens,  # Max response length
            temperature=config.temperature,  # How creative to be
            timeout=config.timeout        # How long to wait
        )
        # Set up performance tracking
        self.execution_count = 0           # How many times ran
        self.success_count = 0          # How many times succeeded
        self.failure_count = 0          # How many times failed
        self.total_execution_time = 0.0 # Total time spent
        
        logger.info("Agent initialized", 
                   agent=config.name, 
                   model=config.llm_model,
                   timeout=config.timeout)
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Define the AI's personality and role.
        
        WHAT IT DOES:
        - Tells the AI what kind of expert it should be
        - Sets the rules for how it should behave
        - Defines what knowledge it should use
        
        EXAMPLE:
        "You are a stock market expert. Analyze stocks professionally
        and give clear buy/sell recommendations."
        
        Returns:
            str: The personality and role instructions for the AI
        """
        pass
    
    @abstractmethod
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """
        Check if the request has everything this agent needs.
        
        WHAT IT DOES:
        - Looks at the incoming request
        - Makes sure required data is present
        - Returns True if agent can work with this request
        
        EXAMPLE:
        Stock analyst would check for 'symbol' in request
        Tax expert would check for 'tax_question' in request
        
        Args:
            state: The incoming request data
            
        Returns:
            bool: True if request is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def process_request(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Do the actual work - this is where the magic happens!
        
        WHAT IT DOES:
        - Takes the validated request
        - Uses the AI to analyze or process it
        - Returns the results in a structured format
        
        THIS IS THE MAIN METHOD:
        Each agent implements its specific logic here:
        - Stock analyst: Analyzes stocks and gives recommendations
        - Tax expert: Provides tax advice
        - Risk expert: Assesses portfolio risk
        
        Args:
            state: The validated request data
            
        Returns:
            Dict with the agent's analysis and recommendations
        """
        pass
    
    # This decorator adds protection: retry on failure, timeout after 30s, limit concurrent calls
    @resilient(
        agent_name=None,  # Will be set dynamically based on agent name
        retry_policy=DEFAULT_RETRY_POLICY,  # Retry 3 times with exponential backoff
        timeout_seconds=30.0,  # Give up after 30 seconds
        bulkhead_name="agent_calls"  # Limit to 5 concurrent agent calls
    )
    async def invoke(self, input_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main entry point - this is what LangGraph calls to run the agent.
        
        WHAT IT DOES:
        - Validates the input request
        - Tracks performance metrics
        - Calls the AI to do the actual work
        - Handles errors gracefully
        - Returns structured results
        
        PROTECTION:
        This method is protected by the @resilient decorator above which:
        - Retries on failures
        - Times out after 30 seconds
        - Limits concurrent calls to prevent overload
        
        Args:
            input_data: The request data from the workflow
            config: Optional runtime configuration
            
        Returns:
            Dict with agent results or error information
        """
        # Start the agent execution
        start_time = time.time()
        self.execution_count += 1
        
        try:
            # Execute within monitoring context for observability
            with monitor_execution(f"agent_{agent_name}", {"agent": agent_name}):
                # Validate input data before processing
                if not self.validate_input(input_data):
                    raise ValueError(f"Invalid input for agent {agent_name}")
                
                # Execute the core agent logic with monitoring
                result = await self._execute_with_monitoring(input_data)
                
                # Update performance metrics for successful execution
                execution_time = time.time() - start_time
                self._update_metrics(True, execution_time)
                
                # Format response for LangGraph compatibility
                return {
                    "messages": [AIMessage(content=result.get("response", ""))],
                    "agent_results": {
                        agent_name: AgentResult(
                            agent_name=agent_name,
                            status=AgentStatus.COMPLETED,
                            result_data=result,
                            execution_time=execution_time
                        ).dict()
                    },
                    "current_agent": None  # Clear current agent to indicate completion
                }
        
        except Exception as e:
            # Handle execution failure with proper error reporting
            execution_time = time.time() - start_time
            self._update_metrics(False, execution_time)
            
            logger.error("Agent execution failed", 
                        agent=agent_name, 
                        error=str(e), 
                        execution_time=execution_time)
            
            # Return error result in LangGraph format
            return {
                "messages": [AIMessage(content=f"Error in {agent_name}: {str(e)}")],
                "agent_results": {
                    agent_name: AgentResult(
                        agent_name=agent_name,
                        status=AgentStatus.FAILED,
                        error_message=str(e),
                        execution_time=execution_time
                    ).dict()
                },
                "current_agent": None,
                "error": str(e)
            }
    
    async def _execute_with_monitoring(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the core agent logic with enhanced context and monitoring.
        
        This private method handles the actual LLM interaction and response processing.
        It enriches the input state with agent context, builds appropriate prompts,
        and processes the LLM response into a structured format.
        
        Args:
            state: Validated input state dictionary
            
        Returns:
            Dict[str, Any]: Processed response with structured data
        """
        # Enhance state with agent context for better LLM understanding
        enhanced_state = state.copy()
        enhanced_state["agent_name"] = self.config.name
        enhanced_state["agent_description"] = self.config.description
        
        # Build conversation messages for LLM
        messages = self._build_messages(enhanced_state)
        
        # Execute LLM call asynchronously
        response = await self.llm.ainvoke(messages)
        
        # Process and structure the LLM response
        return self.process_response(response.content, enhanced_state)
    
    def _build_messages(self, state: Dict[str, Any]) -> List[BaseMessage]:
        """
        Build the complete message list for LLM interaction.
        
        This method constructs the conversation context for the LLM by combining:
        - System prompt defining agent behavior
        - User prompt with specific request details
        - Conversation history (if available)
        
        The message structure follows the OpenAI chat format for optimal compatibility.
        
        Args:
            state: Enhanced state with agent context and request data
            
        Returns:
            List[BaseMessage]: List of LangChain messages for LLM conversation
        """
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": self._build_user_prompt(state)}
        ]
        
        # Include conversation history for context continuity
        if "messages" in state:
            for msg in state["messages"]:
                if isinstance(msg, dict):
                    messages.append(msg)
        
        return messages
    
    def _build_user_prompt(self, state: Dict[str, Any]) -> str:
        """Build user prompt from state."""
        prompt_parts = [f"You are {self.config.name}: {self.config.description}"]
        
        # Add request information
        if "request" in state:
            request = state["request"]
            if hasattr(request, 'stocks') and request.stocks:
                stocks_text = ", ".join([s.symbol for s in request.stocks])
                prompt_parts.append(f"Stocks to analyze: {stocks_text}")
            
            if hasattr(request, 'tax') and request.tax:
                prompt_parts.append(f"Tax question: {request.tax.question}")
            
            if hasattr(request, 'risk') and request.risk:
                prompt_parts.append(f"Portfolio value: ${request.risk.portfolio_value:,.2f}")
                prompt_parts.append(f"Risk tolerance: {request.risk.risk_tolerance}")
        
        return "\n\n".join(prompt_parts)
    
    def process_response(self, response: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process LLM response and return structured result."""
        return {
            "response": response,
            "agent": self.config.name,
            "timestamp": datetime.now().isoformat(),
            "confidence": self._extract_confidence(response),
            "recommendations": self._extract_recommendations(response)
        }
    
    def _extract_confidence(self, response: str) -> float:
        """Extract confidence score from response."""
        # Simple confidence extraction - can be made more sophisticated
        if "high confidence" in response.lower():
            return 0.9
        elif "moderate confidence" in response.lower():
            return 0.7
        elif "low confidence" in response.lower():
            return 0.5
        else:
            return 0.75  # Default
    
    def _extract_recommendations(self, response: str) -> List[str]:
        """Extract recommendations from response."""
        recommendations = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith(('Recommendation:', '- Recommendation', '* Recommendation')):
                recommendations.append(line.replace('Recommendation:', '').strip('- ').strip('* ').strip())
        
        return recommendations
    
    def _update_metrics(self, success: bool, execution_time: float):
        """Update agent metrics."""
        self.execution_count += 1
        self.total_execution_time += execution_time
        
        if success:
            self.success_count += 1
            metrics.increment_counter('agent_success', 1, {'agent': self.config.name})
        else:
            self.failure_count += 1
            metrics.increment_counter('agent_failure', 1, {'agent': self.config.name})
        
        metrics.histogram('agent_execution_time', execution_time, {'agent': self.config.name})
        metrics.set_gauge('agent_success_rate', 
                         self.success_count / self.execution_count if self.execution_count > 0 else 0,
                         {'agent': self.config.name})
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        return {
            "name": self.config.name,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_count / self.execution_count if self.execution_count > 0 else 0,
            "average_execution_time": self.total_execution_time / self.execution_count if self.execution_count > 0 else 0,
            "total_execution_time": self.total_execution_time
        }
    
    def reset_metrics(self):
        """Reset agent metrics."""
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_execution_time = 0.0
        logger.info("Agent metrics reset", agent=self.config.name)
    
    # LangGraph compatibility methods
    def transform(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform input for LangGraph."""
        return input_data
    
    async def atransform(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Async transform for LangGraph."""
        return await self.invoke(input_data)
    
    def stream(self, input_data: Dict[str, Any]):
        """Stream for LangGraph (not implemented)."""
        raise NotImplementedError("Streaming not implemented for this agent")
    
    async def astream(self, input_data: Dict[str, Any]):
        """Async stream for LangGraph (not implemented)."""
        raise NotImplementedError("Streaming not implemented for this agent")


class AgentRegistry:
    """
    Central registry for managing all agent instances in the system.
    
    This class provides a singleton pattern for agent management, allowing
    centralized access to agents, their metrics, and lifecycle management.
    It serves as the primary interface for the orchestrator to discover
    and interact with available agents.
    
    Key Features:
    - Agent registration and discovery
    - Centralized metrics collection
    - Agent lifecycle management
    - Thread-safe operations (implicitly through Python GIL)
    
    Attributes:
        agents: Dictionary mapping agent names to agent instances
    """
    
    def __init__(self):
        """Initialize the agent registry."""
        self.agents: Dict[str, BaseLangGraphAgent] = {}
        logger.info("Agent registry initialized")
    
    def register(self, agent: BaseLangGraphAgent):
        """
        Register an agent instance in the registry.
        
        This method adds an agent to the registry, making it available
        for discovery and execution by the orchestrator. The agent name
        must be unique within the registry.
        
        Args:
            agent: BaseLangGraphAgent instance to register
            
        Raises:
            ValueError: If an agent with the same name is already registered
        """
        if agent.config.name in self.agents:
            logger.warning("Agent already registered, overwriting", agent=agent.config.name)
        
        self.agents[agent.config.name] = agent
        logger.info("Agent registered", agent=agent.config.name)
    
    def get(self, name: str) -> Optional[BaseLangGraphAgent]:
        """
        Retrieve an agent instance by name.
        
        Args:
            name: The unique identifier of the agent to retrieve
            
        Returns:
            BaseLangGraphAgent: The agent instance if found, None otherwise
        """
        return self.agents.get(name)
    
    def list_agents(self) -> List[str]:
        """
        Get a list of all registered agent names.
        
        Returns:
            List[str]: List of agent names currently registered
        """
        return list(self.agents.keys())
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Collect performance metrics from all registered agents.
        
        This method aggregates metrics from all agents, providing a
        comprehensive view of system performance across all agents.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping agent names to their metrics
        """
        return {name: agent.get_metrics() for name, agent in self.agents.items()}
    
    def reset_all_metrics(self):
        """
        Reset performance metrics for all registered agents.
        
        This method clears all performance counters and timing data
        across all agents, typically used for testing or manual
        metric resets.
        """
        for agent in self.agents.values():
            agent.reset_metrics()
        logger.info("All agent metrics reset")


# Global agent registry
agent_registry = AgentRegistry()
