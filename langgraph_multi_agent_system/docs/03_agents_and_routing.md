# Agents and Routing

## The Three Agents

### Equity Agent (`equity_expert`)
**File:** `src/agents/equity_agent.py`

Analyzes stocks using technical and fundamental analysis. Given one or more stock symbols, it returns BUY/HOLD/SELL recommendations with confidence levels and risk factors.

**Triggered when:** The request contains `stocks`

### Tax Agent (`tax_expert`)
**File:** `src/agents/tax_agent.py`

Answers tax questions and identifies tax-saving opportunities. It applies current tax law knowledge to provide actionable advice.

**Triggered when:** The request contains a `tax_question`

### Risk Agent (`risk_expert`)
**File:** `src/agents/risk_agent.py`

Assesses portfolio risk based on value, risk tolerance, and time horizon. Returns a risk score, diversification analysis, and recommendations.

**Triggered when:** The request contains `portfolio_value`

---

## How Agents Are Built

All agents inherit from `BaseLangGraphAgent` (`src/agents/base_agent.py`). Every agent must implement 3 methods:

```python
class MyAgent(BaseLangGraphAgent):

    def get_system_prompt(self) -> str:
        # Returns the AI persona — "You are an expert in X..."
        # This controls how the LLM behaves
        return "You are an expert..."

    def validate_input(self, state: dict) -> bool:
        # Returns True if this agent has what it needs to run
        # If False, the agent is skipped with an error
        return "request" in state

    def process_request(self, state: dict) -> dict:
        # The actual work — builds a prompt, calls the LLM, returns results
        # This is called by the base class after validation passes
        return {"response": "..."}
```

The base class handles everything else: calling the LLM, retrying on failure, timing out, recording metrics.

---

## Deterministic Routing

### The Problem With LLM Routing

In standard LangGraph you might write:
```python
# BAD — asks an LLM "what should we do next?"
def router(state):
    response = llm.invoke("Based on the state, what agent should run next?")
    return response.content  # could be anything
```

This is fragile. The LLM might hallucinate an agent name, pick the wrong one, or loop forever.

### The Solution — Rule-Based Routing

Instead, routing is just Python:
```python
# GOOD — simple rules, always correct
def determine_workflow_type(request):
    has_stocks = bool(request.stocks)
    has_tax = bool(request.tax)
    has_risk = bool(request.risk)

    if has_stocks and has_tax and has_risk:
        return WorkflowType.COMPREHENSIVE_ANALYSIS
    elif has_stocks and has_risk:
        return WorkflowType.PORTFOLIO_REVIEW
    elif has_stocks:
        return WorkflowType.EQUITY_ANALYSIS
    elif has_tax:
        return WorkflowType.TAX_CONSULTATION
    else:
        return WorkflowType.RISK_ASSESSMENT
```

The workflow type is determined once at the start. From then on, the router just checks which required agents have completed and picks the next one.

---

## Workflow Definitions

Each workflow type has a `WorkflowDefinition` listing exactly which agents to call:

```python
WORKFLOW_DEFINITIONS = {
    WorkflowType.EQUITY_ANALYSIS: WorkflowDefinition(
        steps=[WorkflowStep(agent_name="equity_expert", required=True)]
    ),
    WorkflowType.COMPREHENSIVE_ANALYSIS: WorkflowDefinition(
        steps=[
            WorkflowStep(agent_name="equity_expert", required=True),
            WorkflowStep(agent_name="tax_expert",    required=True),
            WorkflowStep(agent_name="risk_expert",   required=True),
        ]
    ),
    # ...
}
```

The orchestrator reads this definition, sees which agents have already run (`state["agent_results"]`), and routes to the next unfinished one.

---

## Adding a New Agent

1. **Create the agent class** in `src/agents/`:
```python
from .base_agent import BaseLangGraphAgent, AgentConfig, agent_registry

class CustomAgent(BaseLangGraphAgent):
    def __init__(self):
        config = AgentConfig(name="custom_expert", ...)
        super().__init__(config)
        agent_registry.register(self)

    def get_system_prompt(self): ...
    def validate_input(self, state): ...
    def process_request(self, state): ...
```

2. **Add a workflow step** in `src/workflows/deterministic_router.py`:
```python
WorkflowType.CUSTOM_WORKFLOW: WorkflowDefinition(
    steps=[WorkflowStep(agent_name="custom_expert", required=True)]
)
```

3. **Add a node** in `src/orchestrator_langgraph.py`:
```python
workflow.add_node("custom_expert", self._create_agent_node("custom_expert"))
```

4. **Add an edge** in the conditional routing maps in `_build_workflow`.
