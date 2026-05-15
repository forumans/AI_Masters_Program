import os
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ToolUseMetric,
    TaskCompletionMetric,
    GEval
)

# Set up your environment
load_dotenv()

def test_agent_evaluation():
    # --- 1. Retrieval Quality ---
    # Measures if the most relevant info is at the top of the context.
    retrieval_metric = ContextualPrecisionMetric(threshold=0.7)

    # --- 2. Tool Selection ---
    # Checks if the correct tool (and arguments) were chosen.
    tool_metric = ToolUseMetric(threshold=0.8)

    # --- 3. Reasoning Correctness ---
    # Uses G-Eval to score logic and step-by-step thinking.
    reasoning_metric = GEval(
        name="Reasoning Correctness",
        criteria="Assess if the agent's logic is sound and correctly connects the input to the output.",
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        threshold=0.7
    )

    # --- 4. Formatting Compliance ---
    # Ensures output strictly matches required formats like JSON or Markdown.
    formatting_metric = GEval(
        name="Formatting Compliance",
        criteria="Verify if the actual output is a valid JSON object with 'status' and 'summary' keys.",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        threshold=1.0 # Requires 100% compliance
    )

    # --- 5. Policy Adherence ---
    # Checks for safety, brand tone, or specific business rules.
    policy_metric = GEval(
        name="Policy Adherence",
        criteria="Ensure the agent does not use prohibited slang and maintains a professional tone.",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        threshold=0.8
    )

    # --- 6. Workflow Completion ---
    # Measures if the end goal of the request was fully satisfied.
    completion_metric = TaskCompletionMetric(threshold=0.8)

    # Define a complex Agentic Test Case
    test_case = LLMTestCase(
        input="Search for the 'Bluebird' project status and format it as a JSON summary.",
        actual_output='{"status": "On-hold", "summary": "Waiting for Q3 budget approval."}',
        retrieval_context=["Project Bluebird: Current status is On-hold pending Q3 budget."],
        # List of tools the agent actually called during its run
        tools_called=[
            ToolCall(name="project_db_search", arguments={"query": "Bluebird"})
        ],
        expected_output='{"status": "On-hold", "summary": "Waiting for Q3 budget approval."}'
    )

    # Execute the local evaluation
    evaluate(
        test_cases=[test_case],
        metrics=[
            retrieval_metric, tool_metric, reasoning_metric, 
            formatting_metric, policy_metric, completion_metric
        ]
    )

if __name__ == "__main__":
    test_agent_evaluation()
