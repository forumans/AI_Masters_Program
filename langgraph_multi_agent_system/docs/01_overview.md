# Overview

## What Is This Project?

This is a **financial analysis multi-agent system** that teaches you how to build reliable, production-ready AI agent systems. It uses three AI agents — equity, tax, and risk — that work together to answer complex financial questions.

The bigger goal is to learn the **engineering patterns** that make agent systems safe to run in production.

---

## The Core Problem This Solves

Standard LangGraph asks an LLM to decide what to do next. That means:
- The LLM might pick the wrong agent
- It might loop forever between agents
- You can't predict what will happen

This project fixes that with **deterministic routing** — a set of simple rules written in Python that always pick the correct agent based on what's in the request. No AI guesswork involved in routing.

---

## The 3 Design Goals

### 1. Deterministic Agent Calling
Every request follows a predictable path. Given the same input, the system always calls the same agents in the same order. This makes it testable, debuggable, and trustworthy.

### 2. Resilient Execution
Agents can fail. Networks time out. APIs return errors. The system handles all of this gracefully using circuit breakers, retries, and timeouts — so one bad agent can't bring down the whole workflow.

### 3. Observable Behavior
Every step is logged with structured data. You can always answer: *What happened? Which agent ran? How long did it take? Did it fail?*

---

## What the System Does (The Use Case)

The domain is financial analysis. A user submits a request that may include:
- **Stock symbols** to analyze (triggers the Equity Agent)
- **A tax question** (triggers the Tax Agent)
- **Portfolio details** like value, risk tolerance, time horizon (triggers the Risk Agent)

The system automatically figures out which agents to call, runs them in sequence, and returns a combined result.

---

## Key Files to Read First

| File | What to Learn From It |
|---|---|
| `src/workflows/deterministic_router.py` | How routing rules work |
| `src/orchestrator_langgraph.py` | How the workflow is built and executed |
| `src/agents/base_agent.py` | The pattern all agents follow |
| `src/utils/resilience.py` | How circuit breakers and retries work |
| `test_workflow.py` | A working end-to-end example |
