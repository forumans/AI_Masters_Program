# Architecture

## How the System Is Structured

```
User Request
     │
     ▼
orchestrator_langgraph.py      ← Entry point. Builds and runs the workflow.
     │
     ▼
deterministic_router.py        ← Looks at the request, decides which agents to call.
     │
     ├──► equity_agent          ← Analyzes stocks
     ├──► tax_agent             ← Answers tax questions
     └──► risk_agent            ← Assesses portfolio risk
     │
     ▼  (multi-agent workflows only)
synthesize_results             ← One GPT call to combine all outputs into final recommendations
     │
     ▼
Combined Result                ← All agent outputs + synthesis returned
```

> **Single-agent workflows** (equity only, tax only, risk only) skip synthesis and go straight to the result.

---

## The Components

### `src/orchestrator_langgraph.py` — The Controller
Builds a LangGraph `StateGraph` and runs the workflow. It:
- Creates the graph with nodes for each agent plus a synthesis node
- Wires up conditional edges (routing logic)
- Executes the graph with a checkpointer (for state persistence)
- Collects and formats the final result including synthesis recommendations

### `src/workflows/deterministic_router.py` — The Router
Contains pre-defined `WorkflowDefinition` objects — one per workflow type. Each definition lists exactly which agents to call. The router looks at the request type and returns the matching definition. No LLM involved.

### `src/agents/` — The Agents
Each agent is a Python class that:
1. Validates its input (does the request have what it needs?)
2. Calls the OpenAI API with a domain-specific system prompt
3. Returns a structured result

All three agents inherit from `BaseLangGraphAgent` which provides the shared retry/timeout/monitoring infrastructure.

The module also contains `synthesize_results` — a standalone async function (not a full agent class) that reads all completed agent results from state and makes one GPT call to produce combined final recommendations.

### `src/models/` — The Data Contracts
Pydantic models that define exactly what data looks like at every point:
- `request_models.py` — what users send in
- `langgraph_state.py` — the shared state object that flows through the graph

### `src/utils/resilience.py` — The Safety Net
Circuit breakers, retry policies, timeouts, and bulkheads. Applied to every agent call automatically via the `@resilient` decorator.

### `src/utils/monitoring.py` — The Observer
Structured JSON logging and Prometheus-compatible metrics. Every agent logs what it received, what it returned, and how long it took.

---

## How Data Flows

```
1. User calls: orchestrator.process_financial_request(user_id, stocks, tax_question, ...)

2. Orchestrator creates an AgentState:
   {
     request: FinancialRequest,
     workflow_type: COMPREHENSIVE_ANALYSIS,
     agent_results: {},        ← starts empty
     status: PENDING,
     messages: []
   }

3. State enters the LangGraph workflow at START

4. Router checks workflow_type → decides: call equity, tax, risk agents

5. Each agent node:
   - Reads state
   - Calls OpenAI
   - Writes result into state["agent_results"]["agent_name"]

6. After each agent, router checks: are all required agents done?
   - No  → route to next agent
   - Yes → for multi-agent workflows: route to synthesize_results
         → for single-agent workflows: route to finalize_workflow

7. synthesize_results (multi-agent only):
   - Reads all agent response texts from state
   - Calls OpenAI once with all outputs as context
   - Writes 3-5 final recommendations into state["agent_results"]["synthesis"]

8. finalize_workflow marks workflow complete → END

9. Orchestrator returns the final state as a formatted dict including final_recommendations
```

---

## Why LangGraph?

LangGraph provides:
- **StateGraph** — a directed graph where nodes are functions and edges define flow
- **Checkpointing** — saves state at every step so you can resume or inspect
- **Conditional edges** — edges that call a function to decide where to go next

We use LangGraph's infrastructure but replace its LLM-based routing with our own deterministic routing functions.
