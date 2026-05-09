"""
POC for prompt chaining with LangGraph 
Prompt chaining - Output of each step is fed as input to the next step

How it works: 
1. LLM generates initial draft for a given topic and quality requirements
2. That initial draft is sent as an input to the fact checker for to identify any improvements needed
3. The draft and improvements needed details are fed to the content improver to improve the draft
4. This improved version is fed as input to the final formatter for structured formatting, readability and publication ready content


Task: Generate blog post with defined quality controls

Input:
- Topic 
- Quality requirements

Steps:
- Generate an initial draft
- Fact check the draft
- Improve the draft based on recommendations from the previous step
- Format for publication
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)

load_dotenv()

class ContentState(TypedDict):
    topic: str
    quality_requirements: str
    draft: str
    fact_check_results: str
    improved_draft: str
    final_content: str

# Create LLM instance
llm = ChatOpenAI(model="gpt-4o-mini")


# Create nodes for each step
def generate_draft(state: ContentState) -> ContentState:
    """Generate initial draft"""

    logging.info(f"============= INSIDE GENERATE INITIAL DRAFT ================================")

    prompt = f"""
        You are a content writer. Generate a blog post that is not more than 100 words about : {state['topic']}

        Quality requirements: {state['quality_requirements']}

        Generate an engaging, and informative draft.
    """

    draft = llm.invoke(prompt).content

    print(f"Initial Draft: {draft}\n\n\n")

    return {"draft": draft}


def fact_check(state: ContentState) -> ContentState:
    """Check the draft for accuracy and completeness"""

    logging.info(f"======================== INSIDE DRAFT REVIEW FOR IMPROVEMENTS ================================")

    prompt = f"""
    Review the following blog post draft for factual accuracy and completeness:
    {state['draft']}
    
    Identify:
    1. Any factual claims that seem questionable
    2. Internal inconsistencies
    3. Statements that need citations

    Provide a brief report.    
    """
    
    fact_check_results = llm.invoke(prompt).content
    
    print(f"Fact Check Results: {fact_check_results}\n\n\n")
    
    return {"fact_check_results": fact_check_results}


def improve_draft(state: ContentState) -> ContentState:
    """Improve the draft based on fact check results"""
    
    logging.info(f"======================== INSIDE IMPROVE DRAFT ================================")
    
    prompt = f"""
    Improve the following blog post draft based on the fact check report:
    
    Draft:
    {state['draft']}
    
    Fact Check Report:
    {state['fact_check_results']}
    
    Provide an improved version of the draft that addresses all issues mentioned in the fact check report.
    """
    
    improved_draft = llm.invoke(prompt).content
    
    print(f"Improved Draft: {improved_draft}\n\n\n")
    
    return {"improved_draft": improved_draft}


def format_for_publication(state: ContentState) -> ContentState:
    """Format the improved draft for publication"""
    
    logging.info(f"======================== INSIDE FORMAT FOR PUBLICATION ================================")
    
    prompt = f"""
    Format the following improved blog post draft for publication:
    
    Improved Draft:
    {state['improved_draft']}
    
    Ensure the content is:
    1. Well-structured with proper headings
    2. Engaging and readable
    3. Optimized for the target audience
    4. Ready for publication
    """
    
    final_content = llm.invoke(prompt).content
    
    #print(f"Final Content: {final_content}\n\n\n")
    
    return {"final_content": final_content}


# Create the graph
graph = StateGraph(ContentState)

# Add nodes
graph.add_node("generate_draft", generate_draft)
graph.add_node("fact_check", fact_check)
graph.add_node("improve_draft", improve_draft)
graph.add_node("format_for_publication", format_for_publication)

# Define edges
graph.add_edge(START, "generate_draft")
graph.add_edge("generate_draft", "fact_check")
graph.add_edge("fact_check", "improve_draft")
graph.add_edge("improve_draft", "format_for_publication")
graph.add_edge("format_for_publication", END)

# Compile the graph
workflow = graph.compile()

# Run the workflow
result = workflow.invoke({
    "topic": "The Benefits of AI in Content Creation",
    "quality_requirements": "Professional, well-researched, and engaging for a business audience"
})

# Display results
print("\n=== FINAL CONTENT ===")
print(result['final_content'])
