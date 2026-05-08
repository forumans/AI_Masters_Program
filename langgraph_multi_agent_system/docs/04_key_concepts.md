# Key Engineering Concepts

This doc explains the engineering patterns used in this system. Understanding these is the main learning goal.

---

## 1. State Management (LangGraph `AgentState`)

**File:** `src/models/langgraph_state.py`

The entire workflow shares a single state object — a Python `TypedDict` called `AgentState`. Every agent reads from it and writes results back to it.

```python
class AgentState(TypedDict):
    request: FinancialRequest      # What the user asked for
    workflow_type: WorkflowType    # Which workflow to run
    agent_results: dict            # What each agent returned
    status: AgentStatus            # PENDING → RUNNING → COMPLETED
    messages: list                 # Chat history for LangGraph
    errors: list                   # Any errors that occurred
    started_at: datetime           # For timing
```

**Why this matters:** Shared state is how agents "talk to each other" without direct coupling. Agent A writes its result, Agent B reads it — they never call each other directly.

LangGraph also **checkpoints** this state after every node. This means:
- You can inspect the state at any point in time
- If a workflow fails, you can see exactly where and why
- Advanced: you can "time travel" back to a previous state and re-run from there

### Critical: State Reducers

By default, when a LangGraph node returns a value, it **replaces** the existing field. This causes a serious bug in multi-agent workflows: Agent 2's result wipes Agent 1's result, so the router never sees Agent 1 as "done" and loops forever.

The fix is **reducers** — functions that tell LangGraph how to *combine* a new value with the existing one instead of replacing it:

```python
from typing import Annotated
import operator
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # add_messages appends new messages instead of replacing the list
    messages: Annotated[List[BaseMessage], add_messages]

    # merge_dicts merges {agent2: result} into {agent1: result} instead of replacing
    agent_results: Annotated[Dict[str, Any], merge_dicts]

    # operator.add appends new errors instead of replacing the list
    errors: Annotated[List[Dict], operator.add]
```

**Rule of thumb:** Any field that multiple agents write to needs a reducer, or later agents will silently overwrite earlier agents' data.

---

## 2. Circuit Breaker Pattern

**File:** `src/utils/resilience.py`

A circuit breaker wraps a function and tracks how often it fails. If it fails too many times in a row, the circuit "opens" and the function is blocked from being called — it immediately raises an error instead of trying again.

```
CLOSED (normal) → fails 3 times → OPEN (blocked)
                                        │
                              30 seconds pass
                                        │
                              HALF-OPEN (one test call)
                                   │         │
                                 pass       fail
                                   │         │
                                CLOSED     OPEN
```

**Why this matters:** Without a circuit breaker, a failing external API (like OpenAI) will cause every request to wait for the full timeout before failing. With a circuit breaker, once you know it's failing, you stop trying immediately and fail fast.

---

## 3. Retry with Exponential Backoff

**File:** `src/utils/resilience.py`

When a call fails (network blip, rate limit, temporary error), retry it — but wait longer between each attempt:

```
Attempt 1 fails → wait 1s → Attempt 2 fails → wait 2s → Attempt 3 fails → give up
```

The `@resilient` decorator applied to agent `invoke()` methods does this automatically. The retry policy is configurable per agent.

**Why this matters:** Most transient errors resolve themselves within a few seconds. Retrying intelligently recovers from these without the user noticing.

---

## 4. Timeout Protection

**File:** `src/utils/resilience.py`

Every agent call has a maximum time limit. If the agent doesn't respond within that time, the call is cancelled and an error is returned.

```python
AgentConfig(
    timeout=30.0,   # Each agent call gets 30 seconds max
    max_retries=3
)
```

**Why this matters:** Without timeouts, a slow or hung API call blocks the entire workflow forever. Timeouts ensure the system always makes progress.

---

## 5. Bulkhead Pattern

**File:** `src/utils/resilience.py`

A bulkhead limits how many concurrent calls can happen at once to a given resource. If the limit is reached, new calls are rejected immediately rather than queuing up.

```python
# Max 10 concurrent calls to the "agent_calls" bulkhead
bulkhead_name="agent_calls"
```

**Why this matters:** Named after ship compartments — if one section floods, the bulkhead prevents it from sinking the whole ship. In software, one overloaded component can't overwhelm everything else.

---

## 6. Pydantic Data Validation

**File:** `src/models/request_models.py`

All inputs and outputs use Pydantic models. Pydantic validates that data matches the expected shape before anything runs.

```python
class StockRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10, pattern="^[A-Z]+$")
    analysis_type: AnalysisType = AnalysisType.TECHNICAL
```

If someone passes `symbol=123` or `symbol=""`, Pydantic raises a clear error immediately — before any agent is called.

**Why this matters:** Catching bad data at the boundary is far simpler than debugging a cryptic error deep inside an agent.

---

## 7. Structured Logging

**File:** `src/utils/monitoring.py`

Every log entry is a JSON object with consistent fields:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "info",
  "event": "agent_execution_complete",
  "agent": "equity_expert",
  "execution_time": 2.34,
  "stocks_analyzed": 2
}
```

**Why this matters:** Plain text logs like `"Agent done"` are useless when debugging production issues. Structured logs can be searched, filtered, and aggregated. You can ask: *"Show me all equity_expert calls that took longer than 5 seconds in the last hour."*

---

## Summary: How These Patterns Combine

```
User Request
    │
    ▼ Pydantic validates input immediately
    │
    ▼ Deterministic router picks agents (no AI guesswork)
    │
    ▼ Agent runs with:
       @resilient decorator provides:
         - Timeout (30s max per call)
         - Retry (up to 3 times with backoff)
         - Circuit breaker (stops calling if agent keeps failing)
         - Bulkhead (limits concurrent calls)
    │
    ▼ State checkpointed after each step (LangGraph)
    │
    ▼ Structured log written with timing + result
    │
    ▼ Result returned
```

This stack of patterns is what makes the difference between a demo that works in a notebook and a system that runs reliably in production.
