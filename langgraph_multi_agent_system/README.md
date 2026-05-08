# LangGraph Multi-Agent System

A learning project that shows how to build a **robust, enterprise-grade multi-agent system** using LangGraph with deterministic routing. The goal is to understand the patterns that make AI agent systems reliable in production.

## What This Teaches You

Standard LangGraph lets an LLM decide which agent to call next — this is unpredictable and can loop forever. This project replaces that with **deterministic routing**: simple rules that always pick the right agent. You also learn:

- **Circuit breakers** — stop calling a failing agent automatically
- **Retry logic** — retry failed calls with backoff
- **State management** — track workflow progress with checkpoints
- **Structured logging** — see exactly what every agent did and why

## Quick Start

### 1. Set up your environment
```bash
pip install -r requirements.txt
# Add your OpenAI API key to src/.env
# OPENAI_API_KEY=your-key-here
```

### 2. Run the test
```bash
python test_workflow.py
```

### 3. Try it yourself
```python
import asyncio
from src.orchestrator_langgraph import LangGraphNativeOrchestrator

async def main():
    orchestrator = LangGraphNativeOrchestrator()
    result = await orchestrator.process_financial_request(
        user_id="user123",
        stocks=[{"symbol": "AAPL", "analysis_type": "technical"}],
        tax_question="How can I reduce capital gains tax?",
        portfolio_value=100000,
        risk_tolerance="moderate",
        time_horizon=5
    )
    print(result)

asyncio.run(main())
```

## Project Structure

```
langgraph_multi_agent_system/
├── src/
│   ├── agents/
│   │   ├── base_agent.py           # Abstract base class all agents inherit from
│   │   ├── equity_agent.py         # Stock market analysis agent
│   │   ├── tax_agent.py            # Tax advice agent
│   │   ├── risk_agent.py           # Portfolio risk assessment agent
│   │   └── langgraph_agents.py     # LangGraph node wrappers + WorkflowOrchestrator
│   ├── models/
│   │   ├── request_models.py       # Pydantic input/output data models
│   │   └── langgraph_state.py      # Workflow state (the shared memory)
│   ├── workflows/
│   │   └── deterministic_router.py # Rules that decide which agents to call
│   ├── utils/
│   │   ├── resilience.py           # Circuit breakers, retry logic, timeouts
│   │   └── monitoring.py           # Structured logging and metrics
│   └── orchestrator_langgraph.py   # Main entry point - runs the workflow
├── docs/                           # Simple learning docs
├── test_workflow.py                # Run this to see everything work
└── requirements.txt
```

## The 5 Workflow Types

| Request Contains | Workflow | Agents Used |
|---|---|---|
| Stocks only | Equity Analysis | equity_expert |
| Tax question only | Tax Consultation | tax_expert |
| Portfolio value only | Risk Assessment | risk_expert |
| Stocks + portfolio | Portfolio Review | equity_expert + risk_expert |
| All three | Comprehensive | All 3 agents |

The router automatically detects which type applies — no AI involved in that decision.

## Documentation

- [Overview & Concepts](docs/01_overview.md) — What this is and why it's built this way
- [Architecture](docs/02_architecture.md) — How the components fit together
- [Agents & Routing](docs/03_agents_and_routing.md) — The agents and deterministic routing explained
- [Key Engineering Concepts](docs/04_key_concepts.md) — Circuit breakers, state, resilience patterns
