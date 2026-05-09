"""
POC for parallelization with LangGraph

Topic: Post the topic to 4 different social media platforms (Twitter, LinkedIn, Facebook, Instagram) at the same time

Note while working on Parallelization:
- Each agent returns the full state dict including topic. 
- When 4 agents run in parallel, they all try to write topic simultaneously—LangGraph rejects this. 
- The fix: return only the field each agent modifies.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from dotenv import load_dotenv


load_dotenv()

class OverallState(TypedDict):
    topic: str
    instagram_post: str
    twitter_post: str
    linkedin_post: str
    facebook_post: str
    final_output: str

llm = ChatOpenAI(model="gpt-4o")

def instagram_agent(state: OverallState) -> OverallState:
    prompt = ChatPromptTemplate.from_template(
        "Write a short 1 line Instagram post for the topic: {topic}"
    )
    response = llm.invoke(prompt.format_prompt(topic=state["topic"]))
    return {"instagram_post": response.content} # Return only the field this agent modifies

def twitter_agent(state: OverallState) -> OverallState:
    prompt = ChatPromptTemplate.from_template(
        "Write a short 1 line Twitter post for the topic: {topic}"
    )
    response = llm.invoke(prompt.format_prompt(topic=state["topic"]))
    return {"twitter_post": response.content} # Return only the field this agent modifies

def linkedin_agent(state: OverallState) -> OverallState:
    prompt = ChatPromptTemplate.from_template(
        "Write a short 1 line LinkedIn post for the topic: {topic}"
    )
    response = llm.invoke(prompt.format_prompt(topic=state["topic"]))
    return {"linkedin_post": response.content} # Return only the field this agent modifies

def facebook_agent(state: OverallState) -> OverallState:
    prompt = ChatPromptTemplate.from_template(
        "Write a short 1 line Facebook post for the topic: {topic}"
    )
    response = llm.invoke(prompt.format_prompt(topic=state["topic"]))
    return {"facebook_post": response.content} # Return only the field this agent modifies

def final_agent(state: OverallState) -> OverallState:
    prompt = ChatPromptTemplate.from_template(
        "Combine all posts into a final output one below the other with headings: Instagram: {instagram_post}\n\nTwitter: {twitter_post}\n\nLinkedIn: {linkedin_post}\n\nFacebook: {facebook_post}"
    )
    response = llm.invoke(prompt.format_prompt(**state))
    return {"final_output": response.content} # Return only the field this agent modifies

graph = StateGraph(OverallState)
graph.add_node("instagram_agent", instagram_agent)
graph.add_node("twitter_agent", twitter_agent)
graph.add_node("linkedin_agent", linkedin_agent)
graph.add_node("facebook_agent", facebook_agent)
graph.add_node("final_agent", final_agent)

# Parallel execution of all social media agents
graph.add_edge(START, "instagram_agent")
graph.add_edge(START, "twitter_agent")
graph.add_edge(START, "linkedin_agent")
graph.add_edge(START, "facebook_agent")
graph.add_edge("instagram_agent", "final_agent")
graph.add_edge("twitter_agent", "final_agent")
graph.add_edge("linkedin_agent", "final_agent")
graph.add_edge("facebook_agent", "final_agent")
graph.add_edge("final_agent", END)

app = graph.compile()

result = app.invoke({
    "topic": "AI in Healthcare"
    }) # Invoke the graph with the details and store them in to the state.

print(result["final_output"]) # Print the final output
