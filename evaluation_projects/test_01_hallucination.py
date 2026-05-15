import os
from deepeval import evaluate
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

# Set the API Key for the judge model
# Note: For a fully local setup, configure DeepEval to use local providers like Ollama
load_dotenv()

def test_hallucination():
    # The ground truth information
    context = ["The project 'Alpha' was completed on Jan 15, 2024."]

    # The output being tested (contains a hallucinated date for demonstration)
    actual_output = "The project Alpha was finished on March 10th, 2024."

    test_case = LLMTestCase(
        input="When was project Alpha completed?",
        actual_output=actual_output,
        retrieval_context=context
    )

    # Initialize the metric. The threshold determines the passing score.
    metric = HallucinationMetric(threshold=0.5)

    # Execute the evaluation
    evaluate(test_cases=[test_case], metrics=[metric])

if __name__ == "__main__":
    test_hallucination()
