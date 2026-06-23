'''
    Source code generator using custom tools
    This program creates custom tools for the agent to use
    The agent invokes these tools based on the user's input to create a program in either Python or JavaScript
    It then creates a directory structure and files for the program
'''
import os
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


# 1. Configure your API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 2. Define Programmatic AI Skills (Tools)
@tool
def calculate_server_capacity(users: int) -> str:
    """Calculates the required server instances based on active concurrent users."""
    # Programmatic logic representing a specialized infrastructure skill
    instances_needed = max(1, users // 5000)
    return f"For {users} concurrent users, you need at least {instances_needed} cloud server instances."

@tool
def generate_project_boilerplate(language: str) -> str:
    """Generates standard project folder structure and files for a specific language."""
    if language.lower() == "python":
        return "Created directory structure: \n- src/main.py\n- tests/\n- requirements.txt\n- README.md"
    elif language.lower() == "javascript":
        return "Created directory structure: \n- src/index.js\n- package.json\n- README.md"
    else:
        return f"Boilerplate for {language} is not yet supported programmatically."

# 3. Bundle skills into a tool list
my_ai_skills = [calculate_server_capacity, generate_project_boilerplate]

# 4. Initialize the LLM (The reasoning engine)
# Using a temperature of 0 ensures stable, predictable tool selection
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 5. Define a ReAct System Prompt to let the LLM handle tool execution loops
prompt_template = """
You are a helpful assistant equipped with specific execution skills.
You have access to the following tools:

{tools}

To use a tool, please use the following format:
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

When you have a response for the user, or if you do not need to use a tool, use the format:
Thought: Do I need to use a tool? No
Final Answer: [your response here]

Begin!
Question: {input}
Thought:{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(prompt_template)

# 6. Construct the programmatic Agent and Executor
agent = create_react_agent(llm=llm, tools=my_ai_skills, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=my_ai_skills, verbose=True)

# 7. Execute the program programmatically
if __name__ == "__main__":
    user_query = "We expect 23,000 active users tomorrow. How many servers do we need, and can you generate a python project setup for it?"
    
    print(f"Triggering Agent with query: '{user_query}'\n")
    response = agent_executor.invoke({"input": user_query})
    
    print("\n--- Final Agent Execution Output ---")
    print(response["output"])
