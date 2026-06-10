import os
import sys
import pandas as pd
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

# Initialize the LLM client (Requires OPENAI_API_KEY environment variable)
client = OpenAI()


# This function extracts structural metadata from the DataFrame to send to the LLM.
def get_data_metadata(df: pd.DataFrame) -> str:
    """Extract structural metadata from the DataFrame to send to the LLM."""
    buffer = []
    buffer.append("--- DATASET SUMMARY ---")
    buffer.append(f"Total Rows: {len(df)}")
    buffer.append(f"Total Columns: {len(df.columns)}")
    buffer.append("\n--- COLUMN DETAILS & MISSING VALUES ---")
    
    # Capture missing values and data types
    missing_info = df.isnull().sum()
    for col in df.columns:
        buffer.append(f"- Column: '{col}' | Type: {df[col].dtype} | Missing Values: {missing_info[col]}")
    
    buffer.append("\n--- SAMPLE DATA (FIRST 3 ROWS) ---")
    buffer.append(df.head(3).to_string())
    
    return "\n".join(buffer)



# This function prompts the LLM to write a comprehensive data cleaning python source code.
# Writes a complete, self-contained Python function named `clean_data(df)`
# Takes the metadata as input and returns the generated code as a string.
def generate_cleaning_code(metadata: str) -> str:
    """Prompt the LLM to write a comprehensive data cleaning script."""
    prompt = f"""
You are an expert Senior Data Engineer AI. Analyze the following metadata of an Excel dataset:

{metadata}

Write a complete, self-contained Python function named `clean_data(df)` that performs comprehensive data preprocessing.
The function MUST handle all common data cleaning steps relevant to this dataset structure:
1. Handle missing values globally or column-specifically (e.g., fill numerical with median, categorical with mode or 'Unknown').
2. Drop or handle explicit duplicate rows.
3. Fix date formats into standardized YYYY-MM-DD strings or datetime objects if date columns exist.
4. Clean text columns (strip whitespace, normalize casing if applicable).
5. Detect and manage extreme numerical outliers (e.g., using IQR or capping).

CRITICAL RULES:
- Return ONLY valid Python code block inside a ```python ``` markdown wrapper.
- Do NOT include any markdown chat text, explanations, or introductory sentences.
- Ensure the function returns the modified DataFrame `return df`.
- Do not import pandas inside the function; assume `import pandas as pd` is already done globally.
"""
    
    response = client.chat.completions.create(
        model="gpt-4o",  # Or "claude-3-5-sonnet" via Anthropic API
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    
    return response.choices[0].message.content


# This function extracts the raw python source code from the markdown block and executes it locally.
def extract_and_execute_code(llm_output: str, df: pd.DataFrame) -> pd.DataFrame:
    """Extracts raw code from the markdown block and executes it locally."""
    # Clean the code block wrappers out of the string
    if "```python" in llm_output:
        code = llm_output.split("```python")[1].split("```")[0].strip()
    elif "```" in llm_output:
        code = llm_output.split("```")[1].split("```")[0].strip()
    else:
        code = llm_output.strip()

    print("\n[AGENT] Generated Cleaning Code:\n", code)
    
    # Create a local execution context
    local_vars = {"df": df, "pd": pd}
    try:
        # Execute the string as Python definitions
        exec(code, globals(), local_vars)
        # Call the dynamically generated function
        cleaned_df = local_vars['clean_data'](df)
        return cleaned_df
    except Exception as e:
        print(f"\n[ERROR] Code execution failed: {e}")
        sys.exit(1)


# This is the main function that orchestrates the entire data cleaning process.
def main():
    # 1. Change file extensions to .csv
    input_file = "./data/sales_data_sample.csv"  # Updated
    output_file = "./data/cleaned_output.csv"   # Updated
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Drop your CSV file into the same VSCode directory.")
        return

    # 2. Change pd.read_excel to pd.read_csv
    print(f"[1/4] Loading dataset: {input_file}...")
    df = pd.read_csv(input_file, encoding='latin-1')  # Updated

    print("[2/4] Profiling data structure and generating metadata...")
    metadata = get_data_metadata(df)
    
    print("[3/4] Consulting AI Model for comprehensive cleaning strategy...")
    raw_code = generate_cleaning_code(metadata)
    
    print("[4/4] Executing AI-generated pipeline locally...")
    cleaned_df = extract_and_execute_code(raw_code, df)
    
    # 3. Change to_excel to to_csv
    cleaned_df.to_csv(output_file, index=False)  # Updated
    print(f"\n[SUCCESS] Pipeline complete! Cleaned file saved to: {output_file}")


if __name__ == "__main__":
    main()
