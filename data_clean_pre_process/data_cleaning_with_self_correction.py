import os
import sys
import traceback
from datetime import datetime  # Used to add exact timestamps to our audit logs
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI API client. This automatically reads your
# OPENAI_API_KEY environment variable set in VSCode or Windows.
client = OpenAI()

# Create a custom logger that prints to terminal and logs to file
class AgentLogger:
    """
    Custom logging utility that handles duplicate routing.
    Every message sent to this logger is printed directly in the VSCode terminal
    and instantly appended to a permanent text report file.
    """
    def __init__(self, filename="cleaning_report.txt"):
        self.filename = filename
        # Open the file in write mode ('w') to clear out previous runs and establish a fresh header
        with open(self.filename, "w", encoding="utf-8") as f:
            f.write(f"==================================================\n")
            f.write(f"AI DATA PREPROCESSING AGENT LOG AUDIT\n")
            f.write(f"Executed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"==================================================\n\n")

    def log(self, message: str):
        """Prints a clean message to the terminal and records it in the report file."""
        print(message)
        # Open in append mode ('a') so we don't erase previous steps recorded during this run
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(f"{message}\n")

    def log_section(self, title: str):
        """Creates an easily scannable visual header block in the log outputs."""
        separator = "-" * 60
        self.log(f"\n{separator}\n{title.upper()}\n{separator}")

# Instantiate our terminal/file logging class globally so all functions can use it
logger = AgentLogger()


# This function profiles the dataset and returns a string with metadata
# It is used to provide the LLM with information about the dataset
def get_data_metadata(df: pd.DataFrame) -> str:
    """
    Profiles the shape, types, and issues of the dataset locally.
    Compiles this structural summary into a compact string prompt for the LLM.
    """
    buffer = []
    buffer.append("--- DATASET SUMMARY ---")
    buffer.append(f"Total Rows: {len(df)}")
    buffer.append(f"Total Columns: {len(df.columns)}")
    buffer.append("\n--- COLUMN DETAILS & MISSING VALUES ---")
    
    # Pre-calculate missing value metrics to save computation inside the loop
    missing_info = df.isnull().sum()
    
    # Build an itemized layout describing the footprint of every column
    for col in df.columns:
        buffer.append(f"- Column: '{col}' | Type: {df[col].dtype} | Missing Values: {missing_info[col]}")
    
    buffer.append("\n--- SAMPLE DATA (FIRST 3 ROWS) ---")
    # Convert the top rows to a string block so the AI evaluates formatting patterns
    buffer.append(df.head(3).to_string())
    
    return "\n".join(buffer)



# This function strips structural markdown decorators (like ```python) out of the LLM's raw text string,
# yielding a clean string containing only pure, executable Python expressions.
def clean_llm_code(llm_output: str) -> str:
    """
    Strips structural markdown decorators (like ```python) out of the LLM's raw text string,
    yielding a clean string containing only pure, executable Python expressions.
    """
    # Check if the standard python markdown block exists
    if "```python" in llm_output:
        # Split by the opening tag and take everything to the right of it ([1])
        after_opening = llm_output.split("```python")[1]
        # From that remainder, split by the closing tag and take everything to the left ([0])
        pure_code = after_opening.split("```")[0]
        return pure_code.strip()
    
    # Check for a generic markdown code block if 'python' wasn't explicitly specified
    elif "```" in llm_output:
        after_opening = llm_output.split("```")[1]
        pure_code = after_opening.split("```")[0]
        return pure_code.strip()
        
    return llm_output.strip()


# This function is the orchestrator. It dispatches the data footprint to the AI model, 
# safely executes the returned logic, and loops error logs back into context for self-correction if it breaks.
def run_correction_loop(metadata: str, initial_df: pd.DataFrame, max_retries: int = 3) -> pd.DataFrame:
    """
    The orchestrator. Dispatches the data footprint to the AI model, safely executes the
    returned logic, and loops error logs back into context for self-correction if it breaks.
    """
    # System prompt establishing role-play guardrails, output structural constraints, 
    # and code execution patterns to stop conversation chatter from crashing compilation.
    system_prompt = (
        "You are an expert Senior Data Engineer AI. Write a complete, self-contained Python function "
        "named `clean_data(df)` that performs comprehensive preprocessing on the provided dataset schema.\n"
        "RULES:\n"
        "- Return ONLY valid Python code block inside a ```python ``` markdown wrapper.\n"
        "- Do NOT include any chat text, explanations, or introductory sentences.\n"
        "- The function MUST accept a DataFrame and return the modified DataFrame: `return df`.\n"
        "- Do not import pandas inside the function; assume `import pandas as pd` is available."
    )
    
    # Maintain conversation state history in a list to support contextual multi-turn code repair
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user", 
            "content": f"Here is the dataset metadata:\n\n{metadata}\n\nPlease generate the `clean_data(df)` function."
        }
    ]

    # Enter the execution loop up to your designated retry ceiling
    for attempt in range(1, max_retries + 1):
        logger.log_section(f"AI Code Generation Attempt {attempt}/{max_retries}")
        
        # Initiate API invocation using low temperature (0.1) to suppress unpredictable logic paths
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.1
        )
        
        raw_output = response.choices[0].message.content

        # Push the output into history so the agent tracking system remembers its previous code strategy
        messages.append({"role": "assistant", "content": raw_output})
        
        # Strip code formatting blocks to isolate logic strings
        code = clean_llm_code(raw_output)
        logger.log(f"--- GENERATED CODE PIPELINE ---\n{code}\n--------------------------------")
        
        # STATE ISOLATION SAFEGUARD: Pass a brand-new deep copy of the original dataset.
        # This keeps a broken snippet from dirtying or fragmenting your base values between attempts.
        df_copy = initial_df.copy()
        
        # Define local environmental scope maps for the runtime compiler engine
        local_vars = {"df": df_copy, "pd": pd}
        
        try:
            # Step A: Dynamically compile the raw text string into a callable runtime declaration
            exec(code, globals(), local_vars)
            
            # Step B: Call the newly established `clean_data` mapping against our local data segment
            cleaned_df = local_vars['clean_data'](df_copy)
            
            # Record successful completion details into the log and report files
            logger.log_section("Execution Successful")
            logger.log(f"[SUCCESS] Code executed flawlessly on attempt {attempt}!")
            return cleaned_df
            
        except Exception as e:
            # Catch any unexpected code exceptions, formatting conflicts, or calculation errors
            error_msg = traceback.format_exc()
            logger.log_section(f"Execution Failed on Attempt {attempt}")
            logger.log(f"[BUG ENCOUNTERED] Python traceback captured:\n{error_msg}")
            
            # If the retry threshold is exhausted, terminate processing completely to protect assets
            if attempt == max_retries:
                logger.log("\n[CRITICAL ERROR] Max self-correction retries reached. Exiting pipeline.")
                sys.exit(1)
                
            # Draft a deterministic correction directive pairing the previous script with its error dump
            feedback = (
                f"Your code failed with the following traceback error:\n\n{error_msg}\n\n"
                "Please fix the bug, ensure data types match, and rewrite the entire corrected "
                "`clean_data(df)` function code block."
            )
            # Append feedback into context so the engine uses this failure data to fix its logic next turn
            messages.append({"role": "user", "content": feedback})



# This function is the main entry point for the data cleaning pipeline.
def main():
    # Configure path metrics for local data storage assets
    input_file = "./data/sales_data_sample.csv" 
    output_file = "./data/cleaned_output.csv"
    
    # Safety boundary intercept to avoid a standard file missing system dump
    if not os.path.exists(input_file):
        logger.log(f"[ERROR] Input file '{input_file}' not found. Pipeline aborted.")
        return

    logger.log(f"[1/3] Loading source data tracking: {input_file}...")
    # Load dataset utilizing 'latin1' encoding map to prevent byte decoding interruptions
    df = pd.read_csv(input_file, encoding='latin1')

    logger.log("[2/3] Analyzing structure and building data metrics...")
    # Extract metadata properties from the target file frame
    metadata = get_data_metadata(df)
    
    # Document core dataset baseline sizing properties right into the logger reports
    logger.log(f"-> Detected Rows: {len(df)} | Detected Columns: {len(df.columns)}")
    
    logger.log("[3/3] Launching self-correcting preprocessing script engine...")
    # Trigger the operational code execution loop mechanism
    cleaned_df = run_correction_loop(metadata, df, max_retries=3)
    
    # Safely write your completed and transformed values to a clean output file
    cleaned_df.to_csv(output_file, index=False)
    
    # Write a post-processing variance report block at the end of the text file audit trail
    logger.log_section("Final Pipeline Summary")
    logger.log(f"Original dataset dimensions : {df.shape}")
    logger.log(f"Preprocessed dataset dimensions: {cleaned_df.shape}")
    logger.log(f"Rows dropped/filtered       : {len(df) - len(cleaned_df)}")
    logger.log(f"[PIPELINE COMPLETE] Exported cleaned file to: {output_file}")
    logger.log(f"[REPORT PRODUCED] Step-by-step audit logs saved to: {logger.filename}")

# Protect execution context blocks when invoking script modules directly in the shell
if __name__ == "__main__":
    main()
