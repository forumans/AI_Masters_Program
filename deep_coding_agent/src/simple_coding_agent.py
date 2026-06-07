import os
import re
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# This is where all project workspaces will be stored
# Create projects directory if it doesn't exist
PROJECTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "projects")
os.makedirs(PROJECTS_DIR, exist_ok=True)


# Tool to name a project
# This tool will be used to create a project folder with the given name
# It should be called FIRST before writing any files
def create_project_folder(project_name: str) -> str:
    """Create a project folder with the given name
    Call this FIRST before writing any files. The name should be a short, 
    descriptive, lowercase name using hyphens (e.g., 'email-validator',
    'todo-api', 'csv-parser'). The folder will be created under ./projects/.

    Args:
        project_name (str): A short, lowercase, hyphen-separated project name
    Returns:
        str: The project path
    """

    validatedProjectName = project_name.lower().strip() # convert to lower case and strip whitespace
    validatedProjectName = re.sub(r"[\s_]+", "-", validatedProjectName) # replace spaces and underscores with hyphens
    validatedProjectName = re.sub(r"[^a-z0-9\-]", "", validatedProjectName) # remove non-alphanumeric characters except hyphens
    validatedProjectName = validatedProjectName.strip("-") # strip leading and trailing hyphens

    if not validatedProjectName:
        raise ValueError("Project name must contain at least one alphanumeric character")

    project_path = os.path.join(PROJECTS_DIR, validatedProjectName)
    os.makedirs(project_path, exist_ok=True)

    return f"Project folder created at: {project_path}.\nWrite all files into this folder using the folder name as prefix - e.g write_file('{validatedProjectName}/main.py', '...')"


# Tool to review code
# This tool will be used to review the code for best practices and potential issues
# It will be used when code has been written and needs a quality check before delivery
code_reviewer = {
    "name": "code_reviewer",
    "description": ("Review code for best practices and potential issues"
                    "Use this tool when code has been written and needs a quality check before delivery"),
    "system_prompt": """
    You are an expert code reviewer. 
    You will be given a task describing which files to review. 
    Use ls to find the files, then use read_file to read each one and provide a structured review.
    
    ## Review Checklist
    1. **Correctness** - Are there logic errors or bugs?
    2. **Best Practices** - Does the code follow Python best practices?
    3. **Simplicity** - Can anything be simplified without losing clarity?
    4. **Edge cases** - Does the code handle empty inputs, None values, boundary conditions?
    5. **Code Quality** - Is the code readable and well-structured?
    6. **Security** - Are there any security vulnerabilities?
    7. **Performance** - Are there any performance issues?

    Also check that a README.md exists and is accurate.

    ## Output Format

    For each file, respond with: 
    - File name
    - Status: PASS or NEEDS CHANGES
    - Issues: List each issue with line reference and suggested fix
    - Strengths: what the code does well
    - Recommendations: Specific recommendations for improvement (if any)

    If all files pass, say "All files pass review - code is ready for delivery."

    Keep your review concise and actionable. Do NOT rewrite the code - just describe the issues.
    """,

    "tools": []
}


# System prompt for the agent that describes the agent's role and workflow. 
# This is the main instruction that tells the agent what to do and how to behave
# It defines the agent's capabilities, tools, and expected behavior
SYSTEM_PROMPT = """
You are a senior code developer. Your job is to take coding tasks from the user and produce clean, well-structured projects.

## Your Workflow

1. **Name the project.** Use the name_project tool with a short, descritive slug - e.g. "task-manager", "web-scraper", "data-analyzer".
   This creates the project folder. After naming, write all files using the folder name as a path prefix (e.g., write_file("my-project/main.py", ...)).
2. **Plan.** use write_todos to break the task into clear implementation steps.
3. **Write code.** Save all code files using write_file. Use descriptive filenames. Always include docstrings and type hints.
4. **Write README.md.** Create a README.md file that includes:
   - Project description of what it does
   - Setup instructions:
   '''
        python -m venv venv
        source venv/bin/activate # On Windows: venv\\Scripts\\activate
        pip install -r requirements.txt # Only if there are dependencies
   '''
   - How to run the program (e.g., `python main.py`)
   - Example output (if applicable)
5. **Write requirements.txt** If the project uses any third-party libraries, create a requirements.txt file with the dependencies.
        Otherwise, don't create a requirements.txt file.
6. **Request a review.** Delegate a code review to the "code-reviewer" subagent using the task tool. In your task description, tell it to use ls and review all files.
7. **Apply fixes.** Read the review feedback carefully. If the reviewer flagged any issues, use edit_file to fix each one. If the review is clean, skip to step 8.
8. **Deliver.** After all fixes are applied:
    a. Use ls to confirm the final list of files.
    b. Tell the user the project folder name and how to get started (point them to the README).
    c. Update your to-do list to mark everything as completed.

## Guidelines

- Write product-quality code - not pseudocode or sketches.
- Each function should do one thing well.
- Include a brief module-level docstring for each file explaining the purpose.
- If the task is complex, break it down into smaller, manageable steps with a clear entry point.
- Always update your to-do list as you progress.
"""


# Initialize the checkpointer
checkpointer = MemorySaver()

# Initialize the model
model = ChatOpenAI(
    model="gpt-4o-mini",
    max_retries=3,
    request_timeout=60, # 60 seconds
)


# Create the deep coding agent
agent = create_deep_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT, # System prompt for the agent that: Names the project, creates the project folder, writes the main.py file, writes the README.md file, writes the requirements.txt file, requests a review, applies fixes, delivers the project
    tools=[create_project_folder], # Tools: that create the project folder
    subagents = [code_reviewer], # Subagents: that review the code and provide feedback
    backend = FilesystemBackend(root_dir=PROJECTS_DIR, virtual_mode=True), # Filesystem backend for the agent: FilesystemBackend
    checkpointer=checkpointer # Checkpointer for the agent: MemorySaver
)


def main():
    banner = r"""
    # ****************************************************************
    # *                                                              *
    # *  ________  _______  _________  _______  _________  ___       *
    # * |\   __  \|\  ___ \|\___   ___\\  ___ \|\___   ___\\  \      *
    # * \ \  \|\  \ \   __/\|___ \  \_\ \   __/\|___ \  \_\ \  \     *
    # *  \ \   ____\ \  \_|/__  \ \  \ \ \  \_|/__  \ \  \ \ \  \    *
    # *   \ \  \___|\ \  \_|\ \  \ \  \ \ \  \_|\ \  \ \  \ \ \  \   *
    # *    \ \__\    \ \_______\  \ \__\ \ \_______\  \ \__\ \ \__\  *
    # *     \|__|     \|_______|   \|__|  \|_______|   \|__|  \|__|  *
    # *                                                              *
    # *                                                              *
    # *                                                              *
    # *  ________  ________  ________  ___  ________   ________      *
    # * |\   ____\|\   __  \|\   ___ \|\  \|\   ___  \|\   ____\     *
    # * \ \  \___|\ \  \|\  \ \  \_|\ \ \  \ \  \\ \  \ \  \___|     *
    # *  \ \  \    \ \  \\\  \ \  \ \\ \ \  \ \  \\ \  \ \  \  ___   *
    # *   \ \  \____\ \  \\\  \ \  \_\\ \ \  \ \  \\ \  \ \  \|\  \  *
    # *    \ \_______\ \_______\ \_______\ \__\ \__\\ \__\ \_______\ *
    # *     \|_______|\|_______|\|_______|\|__|\|__| \|__|\|_______| *
    # *                                                              *
    # *                                                              *
    # *                                                              *
    # *  ________  ________  _______   ________   _________          *
    # * |\   __  \|\   ____\|\  ___ \ |\   ___  \|\___   ___\        *
    # * \ \  \|\  \ \  \___|\ \   __/|\ \  \\ \  \|___ \  \_|        *
    # *  \ \   __  \ \  \  __\ \  \_|/_\ \  \\ \  \   \ \  \         *
    # *   \ \  \ \  \ \  \|\  \ \  \_|\ \ \  \\ \  \   \ \  \        *
    # *    \ \__\ \__\ \_______\ \_______\ \__\\ \__\   \ \__\       *
    # *     \|__|\|__|\|_______|\|_______|\|__| \|__|    \|__|       *
    # *                                                              *
    # ****************************************************************
    """
    
    print(banner)
    print("Plan -> Write -> Review -> Fix -> Deliver")
    print("=" * 60)
    print()
    print(f"Projects will be saved to : {PROJECTS_DIR}")
    print()
    print("Describe a coding task. Type 'quit' to exit")
    print()

    task_count = 0

    while True:
        user_input = input(">").strip()
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        task_count += 1
        config = {"configurable": {"thread_id": f"task_{task_count:03d}"}}

        print()
        print("-" * 60)
        print("Agent is Working...  ")
        print("-" * 60 )
        print()

        for step in agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            stream_mode="updates",
        ):
            for node_name, update in step.items():
                if update and (messages := update.get("messages")):
                    for message in (
                        messages if isinstance(messages, list) else [messages]
                    ):
                        if isinstance(message, BaseMessage):
                            message.pretty_print()

        print()


if __name__ == "__main__":
    main()
