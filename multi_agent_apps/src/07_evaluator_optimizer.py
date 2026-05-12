"""
Evaluator and Optimizer Pattern (LLM-as-a-Judge)

v1: The initial data object is a structured product description with features and audience.

"""
from json import load
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Literal
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()


# Define the state schema for the optimization process
class OptimizationState(TypedDict):
    product_name: str # name of the product
    product_features: List[str] # features of the product
    target_audience: str # target audience for the product
    current_description: str # current generated product details
    evaluation_result: dict # evaluation result from the evaluator
    feedback: str # feedback from the evaluator
    iteration_count: int # current iteration count
    max_iterations: int # maximum number of iterations
    is_approved: bool # whether the description is approved
    iteration_history: List[dict] # Store each evaluation_result's details


# Define the output schema for the product description (Generator)
class ProductDescription(BaseModel):
    headline: str = Field(description="Catchy headline for the product (max 10 words)")
    description: str = Field(description="Main product description (max 100 words)")
    key_benefits: List[str] = Field(description="3-5 key benefits as bullet points")
    call_to_action: str = Field(description="Compelling call to action")


# Define the output schema for the evaluation (Evaluator)
class EvaluationResult(BaseModel):
    overall_score: int = Field(description="Overall Quality score from 1-10", ge=1, le=10)
    clarity_score: int = Field(description="Clarity score 1-10", ge=1, le=10)
    persuasiveness_score: int = Field(description="Persuasiveness score 1-10", ge=1, le=10)
    audience_fit_score: int = Field(description="Target audience fit score 1-10", ge=1, le=10)
    is_approved: bool = Field(description="Whether description meets standards (score >= 8)")
    strengths: List[str] = Field(description="What works well")
    weaknesses: List[str] = Field(description="What needs improvement")
    specific_feedback: str = Field(description="Detailed, actionable feedback for revision")


openai_llm = ChatOpenAI(model="gpt-4o-mini")
gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


# Generator/Optimizer/Producer/Creator Node
def generate_description(state: OptimizationState) -> OptimizationState:
    iteration = state['iteration_count']
    print(f"\n{'='*60}")
    print(f"ITERATION {iteration}")
    print(f"{'='*60}")

    optimizer_llm = openai_llm.with_structured_output(ProductDescription)

    if iteration == 1:
        print("Creating initial product description...")
        
        prompt = f"""Create a complelling product description for:
        Product: {state['product_name']}
        Features: {state['product_features']}
        Target Audience: {state['target_audience']}
        
        Requirements:
        - Headline: Catchy and concise (max 10 words)
        - Description: Engaging and informative (100-150 words)
        - Key Benefits: 3-5 clear, compelling benefits
        - Call to Action: Strong, action-oriented CTA
        
        Make it persuasive and tailored to the target audience.
        """
    else:
        print("Revising product description based on evaluation feedback...")

        prompt = f"""Improve this product description based on feedback:

        Product: {state['product_name']}
        Target Audience: {state['target_audience']}
        Current Description: {state['current_description']}
        Feedback: {state['feedback']}

        Evaluation Scores:
        - Overall: {state['evaluation_result'].get('overall_score', 0)}/10
        - Clarity: {state['evaluation_result'].get('clarity_score', 0)}/10
        - Persuasiveness: {state['evaluation_result'].get('persuasiveness_score', 0)}/10
        - Audience Fit: {state['evaluation_result'].get('audience_fit_score', 0)}/10

        Feedback to address: {state['feedback']}

        Critical: Focus on the specific weakness mentioned. Make targeted improvements to:
        1. Address each point in the feedback
        2. Maintain the strengths that were working
        3. Increase scores in weak areas

        Generate an improved version that addresses all feedback.
        """


    llm_output_desc = optimizer_llm.invoke(prompt)

    formatted_description = f"""
        HEADLINE: {llm_output_desc.headline}

        DESCRIPTION: {llm_output_desc.description}

        KEY BENEFITS: {chr(10).join(f"- {benefit}" for benefit in llm_output_desc.key_benefits)}

        CALL TO ACTION: {llm_output_desc.call_to_action}
    """
    
    print("Generated Description:")
    print("-" * 50)
    print(formatted_description + "\n")
    print("-" * 50)

    return {
        "current_description": formatted_description,
        "feedback": state.get("feedback", ""),
        "evaluation_result": state.get("evaluation_result", {}),
        "iteration_count": iteration + 1
    }


# Evaluator Node
def evaluate_description(state: OptimizationState) -> OptimizationState:
    print(f"\n{'='*60}")
    print(f"EVALUATOR: Reviewing Description....")
    print(f"{'='*60}")

    
    evaluator_llm = gemini_llm.with_structured_output(EvaluationResult)
    
    prompt = f"""Evaluate this product description for:
    Product: {state['product_name']}
    Target Audience: {state['target_audience']}
    
    Description:
    {state['current_description']}
    
    Evaluation Criteria:
    1. Overall Quality (1-10): How well does it meet marketing standards?
    2. Clarity (1-10): Is it easy to understand?
    3. Persuasiveness (1-10): Does it convince the reader to buy?
    4. Audience Fit (1-10): Does it resonate with the target audience?
    
    Approval Criteria: Overall score must be 8 or higher to approve.

    Provide:
    - Overall score (1-10)
    - Individual scores for each criterion
    - Whether it meets approval standards (score >= 8)
    - Strengths (what works well)
    - Weaknesses (what needs improvement)
    - Specific, actionable feedback for improvement
    """
    
    gemini_evaluation_result = evaluator_llm.invoke(prompt)
    
    print(f"Evaluation Overall Score: {gemini_evaluation_result.overall_score}/10")
    print(f"Clarity: {gemini_evaluation_result.clarity_score}/10")
    print(f"Persuasiveness: {gemini_evaluation_result.persuasiveness_score}/10")
    print(f"Audience Fit: {gemini_evaluation_result.audience_fit_score}/10")
    print(f"Status: {'✅ APPROVED' if gemini_evaluation_result.is_approved else '❌ NEEDS REVISION'}")
    print(f"Strengths:") 
    for strength in gemini_evaluation_result.strengths:
        print(f"  - {strength}")

    print(f"Weaknesses:") 
    for weakness in gemini_evaluation_result.weaknesses:
        print(f"  - {weakness}")

    print(f"Feedback: {gemini_evaluation_result.specific_feedback}")

    # Create iteration record for history log
    iteration_record = {
        "iteration": state["iteration_count"],
        "description": state["current_description"],
        "scores": {
            "overall": gemini_evaluation_result.overall_score,
            "clarity": gemini_evaluation_result.clarity_score,
            "persuasiveness": gemini_evaluation_result.persuasiveness_score,
            "audience_fit": gemini_evaluation_result.audience_fit_score
        },
        "approved": gemini_evaluation_result.is_approved,
        "feedback": gemini_evaluation_result.specific_feedback
    }

    # Update iteration history by appending the current iteration record
    history = state.get("iteration_history", [])
    history.append(iteration_record)

    return {
        "evaluation_result": gemini_evaluation_result.model_dump(),
        "feedback": gemini_evaluation_result.specific_feedback,
        "is_approved": gemini_evaluation_result.is_approved,
        "iteration_history": history
    }



# Decision edge function
def should_continue(state: OptimizationState) -> Literal["send_to_optimizer", "end"]:
    if state["is_approved"]: # if the description is approved, end the loop
        print("\n✅ SUCCESS: Description is Approved!!!")
        return "end"
    
    elif state["iteration_count"] >= 3: # if the iteration count is 3, end the loop
        print(f"\n⚠️ MAXIMUM Iterations reached. Stopping at iteration {state['iteration_count']}")
        return "end"
    
    else: # if the description is not approved, continue the loop
        print("\n❌ Description needs revision...")
        return "send_to_optimizer"
    

# Build the graph
workflow = StateGraph(OptimizationState)

# Add nodes
workflow.add_node("generator", generate_description)
workflow.add_node("evaluator", evaluate_description)

# Add edges
workflow.add_edge(START, "generator")
workflow.add_edge("generator", "evaluator")
workflow.add_conditional_edges(
    "evaluator",
    should_continue,
    {
        "send_to_optimizer": "generator",
        "end": END
    }
)


# Compile the graph
graph = workflow.compile()

# Test the graph
if __name__ == "__main__":
    print("🚀 Starting Evaluator-Optimizer Workflow")
    
    initial_data = {
        "product_name": "Fitpulse Pro Smart watch",
        "product_features": [
            "Heart rate monitoring", 
            "GPS tracking", 
            "Waterproof to 50m", 
            "7-day battery life",
            "Smartphone notifications"
        ],
        "target_audience": "Health-conscious professionals aged 25-45",
        "current_description": "",
        "evaluation_result": {},
        "feedback": "",
        "iteration_count": 1,
        "max_iterations": 5,
        "is_approved": False,
        "iteration_history": []
    }

    # Run the workflow
    result = graph.invoke(initial_data)
    
    # Print results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    print(f"\n✅ Final Description: {result['current_description']}")
    print(f"\n📊 Evaluation Score: {result['evaluation_result']['overall_score']}/10")
    print(f"✅ Approved: {result['is_approved']}")
    
    print("\n" + "="*60)
    print("ITERATION HISTORY")
    print("="*60)
    
    for i, iteration in enumerate(result['iteration_history'], 1):
        print(f"\nIteration {iteration['iteration']}:")
        print(f"  Score: {iteration['scores']['overall']}/10")
        print(f"  Status: {'✅ Approved' if iteration['approved'] else '❌ Needs Revision'}")
        print(f"  Feedback: {iteration['feedback'][:100]}...")
    
    print("\n" + "="*60)
    print("🎉 Workflow Complete!")
    print("="*60)
