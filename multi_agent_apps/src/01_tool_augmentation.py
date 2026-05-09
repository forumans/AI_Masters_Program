"""
Tool Augmentation Example
"""

import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

@tool
def get_weather(location: str) -> str:
    """Get the weather for a given location."""
    return f"The weather in {location} is sunny."


@tool
def calculate_tip(bill_amount: float, tip_percentage: float) -> float:
    """Calculate the tip amount based on bill amount and tip percentage."""
    return bill_amount * (tip_percentage / 100)


llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools([
    get_weather,
    calculate_tip    
    ])

def main():
    """
    Sample prompts:
    - "What is the weather in Ashburn, VA?"
    - "What is the 15% tip for a bill of $150?"
    """
    user_prompt = input("Enter your prompt: ")
    response = llm_with_tools.invoke(user_prompt)
    tool_calls = response.tool_calls

    # Execute the tool
    for tool_call in tool_calls:
        tool_call_name = tool_call['name']
        tool_call_args = tool_call['args']

        if tool_call_name == "calculate_tip":
            result = calculate_tip.invoke(tool_call_args)
            print(result)
        elif tool_call_name == "get_weather":
            result = get_weather.invoke(tool_call_args)
            print(result)

if __name__ == "__main__":
    main()
