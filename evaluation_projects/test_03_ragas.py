"""
RAGAS evaluation for the HR Assistant RAG pipeline.

Measures four RAG-specific metrics across 12 HR policy questions:
  - faithfulness       : answer is grounded in retrieved context (no hallucination)
  - answer_relevancy   : answer addresses the question asked
  - context_precision  : retrieved chunks are the relevant ones (low noise)
  - context_recall     : retrieved chunks cover all facts in the ground truth

Run:
    python evaluation_projects/test_03_ragas.py
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


# ===========================================================================
# SECTION 1: PATH SETUP
#
# WHY: This script can be run from any folder on your machine. Python's import
#      system and file-path lookups are relative to wherever you run the script
#      from, not where the script lives. We fix all paths to absolute locations
#      upfront so nothing breaks regardless of the working directory.
#
# HOW: Path(__file__) gives the absolute path of THIS script file. We walk
#      up the folder tree from there to locate the hr_assistant_api folder.
# ===========================================================================

SCRIPT_DIR = Path(__file__).resolve().parent           # .../evaluation_projects/
REPO_ROOT  = SCRIPT_DIR.parent                         # .../GenAI_Masters_Program/
API_DIR    = REPO_ROOT / "hr_assistant" / "hr_assistant_api"  # where the app lives


# ===========================================================================
# SECTION 2: LOAD ENVIRONMENT VARIABLES BEFORE IMPORTING THE APP
#
# WHY: The HR Assistant app reads configuration (API keys, DB URL, model names,
#      file paths) from a .env file. Specifically, app/config.py does:
#
#          settings = Settings()   <-- runs the moment the file is imported
#
#      Pydantic checks that every required field is present. If any variable is
#      missing, it raises a validation error and the import crashes. So we MUST
#      load the .env into os.environ BEFORE we do "from app.services import ...".
#
# HOW: dotenv_values() reads the .env file and returns a plain dictionary
#      (it does NOT automatically put values into os.environ). We then push
#      every key-value pair into os.environ manually so Pydantic can find them.
#      We give dotenv_values() an explicit absolute path so it finds the file
#      regardless of which folder you ran the script from.
# ===========================================================================

from dotenv import dotenv_values  # noqa: E402 (intentionally imported after path setup)

env = dotenv_values(API_DIR / ".env")

# Push every variable from .env into the process environment.
# Skip entries where the value is None (blank lines in .env can produce those).
os.environ.update({k: v for k, v in env.items() if v is not None})

# Tell Python where to find the "app" package so that
# "from app.services.vector_store import ..." resolves correctly.
sys.path.insert(0, str(API_DIR))


# ===========================================================================
# SECTION 3: RESOLVE ABSOLUTE FILE PATHS FOR CHROMADB AND THE PDF
#
# WHY: The .env file stores these paths as relative strings (e.g. "./chroma_data").
#      Relative paths are interpreted from wherever you run the script, which
#      varies. We convert them to absolute paths anchored to API_DIR.
#
# HOW:
#   - lstrip("./\\") strips leading dot, slash, or backslash characters so that
#     "./chroma_data" becomes "chroma_data", which we can safely join onto API_DIR.
#   - .resolve() on a Path object collapses any ".." segments and returns a
#     clean absolute path.
# ===========================================================================

_chroma_rel = env.get("CHROMA_PERSIST_DIR", "./chroma_data").lstrip("./\\")
CHROMA_DIR  = str(API_DIR / _chroma_rel)   # absolute path to the ChromaDB folder on disk

_pdf_rel  = env.get("HR_POLICIES_PATH", "../hr_policies/hr_policies.pdf")
PDF_PATH  = str((API_DIR / _pdf_rel).resolve())   # absolute path to the HR policies PDF

OPENAI_API_KEY = env["OPENAI_API_KEY"]


# ===========================================================================
# SECTION 4: IMPORT THE HR ASSISTANT SERVICES
#
# WHY: We import after setting os.environ so Pydantic's settings validation
#      (which runs at import time) can find all required variables. We only
#      need two services: VectorStoreService (retrieval) and AIService (generation).
#      We do NOT import FastAPI, the database session, or any web-layer code —
#      this evaluation script talks directly to the core logic.
# ===========================================================================

from app.services.vector_store import VectorStoreService  # noqa: E402
from app.services.ai_service import AIService             # noqa: E402


# ===========================================================================
# SECTION 5: TEST DATASET
#
# WHY: RAGAS is a supervised evaluation framework. It needs a set of questions
#      we already know the correct answers to (called "ground truth") so it can
#      judge whether the RAG pipeline retrieved the right content and generated
#      a faithful, relevant answer.
#
# HOW: Each entry is a dict with three fields:
#   - "topic"        : category label (used only for display in the report)
#   - "question"     : the question we will ask the HR Assistant
#   - "ground_truth" : the correct answer, written by a human from the policy PDF
#
# IMPORTANT CONSTRAINT: None of the questions contain keywords like "on leave"
#      or "on vacation". Those keywords make AIService route the request to
#      _handle_leave_query(), which queries the PostgreSQL database. In this
#      evaluation we intentionally run without a database connection (db=None),
#      so hitting that code path would crash the script.
# ===========================================================================

TEST_DATASET = [
    # --- Leave policy ---
    {
        "topic": "Leave",
        "question": "When can employees start using their annual leave after being hired?",
        "ground_truth": (
            "Employees begin to accrue annual leave immediately upon hire, "
            "but may not use annual leave until after 90 days of employment."
        ),
    },
    {
        "topic": "Leave",
        "question": "How much vacation time do employees earn based on years of service?",
        "ground_truth": (
            "Employees earn 1.5 pro-rated days per month for less than 2 years of service, "
            "1.75 pro-rated days per month for years 2 through 6, and 2 pro-rated days per "
            "month for 7 or more years of service."
        ),
    },
    {
        "topic": "Leave",
        "question": "What happens to unused vacation time when an employee is terminated?",
        "ground_truth": (
            "Any earned but unused vacation will be paid at the time of termination."
        ),
    },
    {
        "topic": "Leave",
        "question": "How far in advance must employees request vacation approval?",
        "ground_truth": (
            "Employees should request approval in writing at least two weeks in advance "
            "before taking vacation."
        ),
    },
    # --- Expense policy ---
    {
        "topic": "Expense",
        "question": "What expenses does the organization not reimburse employees for while traveling?",
        "ground_truth": (
            "The organization does not reimburse for personal activities while traveling "
            "or other expenses such as entertainment, liquor, dry cleaning, etc."
        ),
    },
    {
        "topic": "Expense",
        "question": "How many business days does an employee have to submit a travel expense report after returning from a trip?",
        "ground_truth": (
            "Employees should submit a travel expense report containing receipts within "
            "7 business days of completion of travel."
        ),
    },
    # --- Working hours ---
    {
        "topic": "Working Hours",
        "question": "What are the standard working hours at the organization?",
        "ground_truth": (
            "Standard working hours are 8:00 a.m. to 4:30 p.m., Monday through Friday, "
            "with an unpaid meal period of thirty minutes."
        ),
    },
    {
        "topic": "Working Hours",
        "question": "When does the organization's workweek begin and end?",
        "ground_truth": (
            "The workweek begins at 12:00 a.m. Saturday and ends at 11:59 p.m. Friday."
        ),
    },
    # --- Benefits / Holidays ---
    {
        "topic": "Benefits",
        "question": "Which employees are eligible for paid holidays?",
        "ground_truth": (
            "Regular full-time and regular part-time employees are eligible for paid holidays."
        ),
    },
    {
        "topic": "Benefits",
        "question": "If a paid holiday falls during an employee's scheduled vacation, what happens to their vacation day?",
        "ground_truth": (
            "If a paid holiday falls during an employee's scheduled vacation period, "
            "holiday pay will be provided and the employee will still have a vacation day to use."
        ),
    },
    # --- Code of conduct ---
    {
        "topic": "Code of Conduct",
        "question": "What forms of harassment does the organization prohibit in the workplace?",
        "ground_truth": (
            "The organization prohibits any form of harassment based on race, color, religion, "
            "sex, national origin, age, disability, veteran status, pregnancy, marital status, "
            "medical condition, sexual orientation, or any other status protected by Federal "
            "and state law."
        ),
    },
    {
        "topic": "Code of Conduct",
        "question": "What is the organization's policy on outside employment?",
        "ground_truth": (
            "Employees may hold outside employment as long as it does not interfere with "
            "performance or ability to meet the job requirements at the organization."
        ),
    },
]


# ===========================================================================
# SECTION 6: PASS/FAIL THRESHOLDS
#
# WHY: A raw score (e.g. 0.72) is meaningless without a benchmark to compare
#      against. These thresholds represent the minimum acceptable quality level
#      for each metric. Scores at or above the threshold are labelled PASS;
#      below it, FAIL.
#
# HOW: The values are set conservatively based on typical RAG quality targets:
#   - faithfulness and answer_relevancy at 0.80 (high bar — answers must be
#     accurate and on-topic)
#   - context_precision at 0.70 (some noise in retrieval is acceptable)
#   - context_recall at 0.75 (most relevant facts must be retrieved)
# ===========================================================================

THRESHOLDS = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.80,
    "context_precision": 0.70,
    "context_recall": 0.75,
}


# ===========================================================================
# FUNCTION: extract_answer_text
#
# WHY: The HR Assistant's AIService.generate_response() always returns a JSON
#      string, not plain text. For example:
#          '{"type": "text", "data": "The leave policy states...", "meta": {...}}'
#      RAGAS expects a plain English string as the answer. This function peels
#      off the JSON wrapper and returns only the human-readable content.
#
# HOW: It parses the JSON, reads the "type" field to know the response shape,
#      then extracts and formats the "data" field accordingly:
#   - "text"   → the data value is already a plain string, return as-is
#   - "table"  → data contains columns + rows; flatten into a readable text table
#   - "metric" → data contains a label + number; format as "Label: 42"
#   - anything else (e.g. "tree") → convert to string as a fallback
#      If JSON parsing fails entirely, the raw string is returned unchanged.
# ===========================================================================

def extract_answer_text(raw_response: str) -> str:
    try:
        parsed = json.loads(raw_response)
        response_type = parsed.get("type", "text")
        data = parsed.get("data", "")

        if response_type == "text":
            # Most policy questions return a plain text answer — just unwrap it.
            return str(data)

        if response_type == "table" and isinstance(data, dict):
            # Tables come as {"columns": [...], "rows": [[...], ...]}
            # Join each row's values with " | " to make a readable plain-text table.
            columns = data.get("columns", [])
            rows    = data.get("rows", [])
            lines   = [" | ".join(str(c) for c in columns)]
            for row in rows:
                lines.append(" | ".join(str(v) for v in row))
            return "\n".join(lines)

        if response_type == "metric" and isinstance(data, dict):
            # Metrics are {"label": "Headcount", "value": 42} — format as a sentence.
            return f"{data.get('label', '')}: {data.get('value', '')}"

        # Fallback for "tree" type or any unexpected format.
        return str(data)

    except (json.JSONDecodeError, AttributeError):
        # If the response isn't valid JSON at all, use it as-is.
        return raw_response


# ===========================================================================
# FUNCTION: build_ragas_dataset
#
# WHY: RAGAS needs four parallel lists — one entry per question:
#       question, answer, contexts, ground_truth
#      This function builds those lists by actually running the HR Assistant's
#      RAG pipeline on every question in TEST_DATASET.
#
# HOW: For each question it makes two calls:
#   1. vs.query(question, n_results=5)
#      → asks ChromaDB to find the 5 most relevant text chunks from the
#        HR policies PDF. These are the "contexts" RAGAS will evaluate.
#   2. ai.generate_response(question, ...)
#      → sends the question (plus the retrieved contexts internally) to the
#        OpenAI model and gets back a JSON response. We extract the plain text.
#
#      time.sleep(0.5) adds a small pause between questions so we don't hit
#      OpenAI's rate limits when processing all 12 questions back-to-back.
#
# RETURNS: A plain dict with four keys, each holding a list of 12 items.
#          This is the exact shape the HuggingFace Dataset class expects.
# ===========================================================================

def build_ragas_dataset(vs: VectorStoreService, ai: AIService) -> dict:
    questions, answers, contexts_list, ground_truths = [], [], [], []
    n = len(TEST_DATASET)

    for i, item in enumerate(TEST_DATASET):
        q = item["question"]
        print(f"  [{i + 1:02d}/{n}] [{item['topic']}] {q[:70]}...")

        # Step A: Retrieve the top-5 relevant chunks from ChromaDB.
        contexts = vs.query(q, n_results=5)

        # Step B: Generate an answer using the LLM (RAG chain inside AIService).
        #         We pass db=None because we have no database session here —
        #         that is safe as long as no question triggers the leave-query path.
        raw = ai.generate_response(
            question=q,
            employee_context=None,       # no specific employee context needed
            conversation_history=None,   # no chat history; each question is standalone
            db=None,                     # no database session — policy questions only
        )

        # Step C: Strip the JSON wrapper so we have plain text for RAGAS.
        answer = extract_answer_text(raw)

        # Accumulate results for this question.
        questions.append(q)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(item["ground_truth"])

        time.sleep(0.5)  # brief pause between API calls to respect rate limits

    return {
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts_list,   # list of lists: each entry is 5 chunk strings
        "ground_truth": ground_truths,
    }


# ===========================================================================
# FUNCTION: print_report
#
# WHY: The raw RAGAS result object is hard to read. This function formats it
#      into a human-friendly console table showing both overall (aggregate)
#      scores and per-question scores, with a clear PASS/FAIL verdict.
#
# HOW:
#   - result.to_pandas() converts the RAGAS output into a DataFrame where each
#     row is one question and each column is one metric score.
#   - df[key].mean() computes the average score across all 12 questions for
#     each metric — this is the "aggregate" score we report at the top.
#   - We then iterate row-by-row to print the per-question breakdown.
#   - pd.isna() guards against NaN values that RAGAS produces when a metric
#     couldn't be computed for a particular question (e.g. due to an API error).
#
# RETURNS: agg — a dict of {metric_name: average_score} used later when saving
#          the JSON results file.
# ===========================================================================

def print_report(result, run_date: str, model: str, num_questions: int) -> dict:
    import pandas as pd

    width = 65
    print("\n" + "=" * width)
    print("RAGAS Evaluation Report — HR Assistant RAG Pipeline")
    print(f"Run Date : {run_date}")
    print(f"Model    : {model}")
    print(f"Questions: {num_questions}")
    print("=" * width)

    # Convert RAGAS result into a DataFrame (one row per question, one column per metric).
    df = result.to_pandas()

    metric_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    # Compute the average score for each metric across all questions.
    agg = {}
    for key in metric_keys:
        if key in df.columns:
            agg[key] = df[key].mean()

    # Print the aggregate summary with pass/fail verdict.
    print("\nAGGREGATE SCORES:")
    for metric, score in agg.items():
        threshold = THRESHOLDS.get(metric, 0.0)
        status = "PASS" if score >= threshold else "FAIL"
        print(f"  {metric:<22}: {score:.4f}  [target >= {threshold:.2f}]  {status}")

    # Print the per-question breakdown table.
    print("\nPER-QUESTION BREAKDOWN:")
    header = (
        f"  {'#':>2}  {'Topic':<16}  {'Q (truncated)':<45}  "
        + "  ".join(f"{k[:8]:>8}" for k in metric_keys)
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for idx, row in df.iterrows():
        topic   = TEST_DATASET[idx]["topic"]
        q_short = TEST_DATASET[idx]["question"][:45]
        # Format each metric score; show "N/A" if RAGAS couldn't compute it.
        scores_str = "  ".join(
            f"{row[k]:>8.4f}" if k in df.columns and not pd.isna(row[k]) else f"{'N/A':>8}"
            for k in metric_keys
        )
        print(f"  {idx + 1:>2}  {topic:<16}  {q_short:<45}  {scores_str}")

    print("=" * width)
    return agg


# ===========================================================================
# FUNCTION: main
#
# WHY: Orchestrates all the steps in the correct order:
#        init services → build dataset → evaluate → report → save
#      Keeping this in a function (rather than at module level) means the
#      evaluation only runs when you explicitly execute the script, not when
#      another file imports it.
#
# HOW: Five numbered steps that mirror the plan document exactly.
# ===========================================================================

def main():
    print("\n=== HR Assistant — RAGAS Evaluation ===\n")

    # ------------------------------------------------------------------
    # Step 1: Initialise the HR Assistant's core services.
    #
    # VectorStoreService connects to the existing ChromaDB on disk.
    # load_pdf() checks if documents are already stored; if the ChromaDB
    # folder already has 203 chunks (from a previous run), it skips
    # re-ingestion and returns immediately — no wasted time or API calls.
    #
    # AIService MUST be created after load_pdf() because its __init__
    # calls vector_store.vector_store.as_retriever(), which requires the
    # Chroma object to already be initialised inside VectorStoreService.
    # ------------------------------------------------------------------
    print("Initialising services...")
    vs = VectorStoreService(persist_directory=CHROMA_DIR, openai_api_key=OPENAI_API_KEY)
    vs.load_pdf(PDF_PATH)

    ai = AIService(openai_api_key=OPENAI_API_KEY, vector_store=vs)
    model_name = ai.model
    print(f"Services ready. Model: {model_name}\n")

    # ------------------------------------------------------------------
    # Step 2: Run the RAG pipeline on every test question.
    #
    # build_ragas_dataset() returns a plain dict with four lists:
    #   question, answer, contexts, ground_truth
    # Each list has one entry per question (12 total).
    # ------------------------------------------------------------------
    print("Building evaluation dataset (calling RAG pipeline)...")
    data = build_ragas_dataset(vs, ai)
    print(f"\nDataset built: {len(data['question'])} samples.\n")

    # ------------------------------------------------------------------
    # Step 3: Feed the dataset into RAGAS for evaluation.
    #
    # Dataset.from_dict() wraps the plain dict into a HuggingFace Dataset
    # object that RAGAS expects. Think of it as a typed, validated table.
    #
    # In RAGAS 0.4.x the metrics from ragas.metrics are plain dataclasses —
    # they are instantiated without arguments (Faithfulness(), etc.) and the
    # LLM + embeddings are supplied once to evaluate() rather than to each
    # metric individually. This is different from ragas.metrics.collections,
    # which uses a separate class hierarchy that evaluate() does not accept.
    #
    # llm_factory() creates a RAGAS-native LLM wrapper around the OpenAI
    # client. LangchainEmbeddingsWrapper adapts the LangChain embeddings
    # class so RAGAS can call embed_query() internally (required by
    # AnswerRelevancy when it computes semantic similarity).
    #
    # raise_exceptions=False tells RAGAS to record NaN for a question if
    # scoring fails (e.g. due to an API timeout) rather than crashing the
    # whole evaluation run.
    # ------------------------------------------------------------------
    print("Running RAGAS evaluation (LLM-as-judge)...")
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    from ragas.llms import llm_factory
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import OpenAIEmbeddings as LCOpenAIEmbeddings
    from openai import OpenAI

    # Create the judge LLM — this is the model RAGAS uses to score answers,
    # NOT the model that generated the answers (that was AIService above).
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    judge_llm = llm_factory(model_name, client=openai_client)

    # AnswerRelevancy computes a semantic similarity score between the
    # question and the answer, so it needs an embeddings model.
    judge_emb = LangchainEmbeddingsWrapper(
        LCOpenAIEmbeddings(
            model=env.get("EMBEDDING_MODEL_NAME", "text-embedding-3-small"),
            openai_api_key=OPENAI_API_KEY,
        )
    )

    ds = Dataset.from_dict(data)
    result = evaluate(
        ds,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
        llm=judge_llm,
        embeddings=judge_emb,
        raise_exceptions=False,
    )
    print("Evaluation complete.\n")

    # ------------------------------------------------------------------
    # Step 4: Print the human-readable report to the console.
    # ------------------------------------------------------------------
    run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    agg = print_report(result, run_date, model_name, len(data["question"]))

    # ------------------------------------------------------------------
    # Step 5: Save the full results to a timestamped JSON file.
    #
    # We build a per_question list so the JSON file contains every detail:
    # the question, the answer the model gave, the expected answer, and
    # every metric score — useful for later analysis or sharing.
    #
    # pd.isna() checks for NaN before writing so the JSON stays valid
    # (JSON has no NaN; we write null/None instead).
    # ------------------------------------------------------------------
    import pandas as pd
    df = result.to_pandas()
    per_question = []
    metric_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    for idx, row in df.iterrows():
        per_question.append({
            "index":        idx + 1,
            "topic":        TEST_DATASET[idx]["topic"],
            "question":     TEST_DATASET[idx]["question"],
            "answer":       data["answer"][idx],
            "ground_truth": TEST_DATASET[idx]["ground_truth"],
            # Write the score as a rounded float, or None if RAGAS couldn't compute it.
            **{
                k: (None if pd.isna(row[k]) else round(float(row[k]), 4))
                for k in metric_keys
                if k in df.columns
            },
        })

    output = {
        "run_date":        run_date,
        "model":           model_name,
        "embedding_model": env.get("EMBEDDING_MODEL_NAME", "unknown"),
        "num_questions":   len(data["question"]),
        "aggregate_scores": {k: round(v, 4) for k, v in agg.items()},
        "thresholds":      THRESHOLDS,
        "per_question":    per_question,
    }

    # Timestamp the filename so each run produces a unique file and old
    # results are never overwritten.
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = SCRIPT_DIR / f"ragas_results_{timestamp}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {results_path}")


# ===========================================================================
# ENTRY POINT
#
# WHY: This guard ensures main() only runs when you execute the script
#      directly (e.g. "python test_03_ragas.py"). If another file imports
#      this module, main() will NOT run automatically.
# ===========================================================================

if __name__ == "__main__":
    main()
