"""
RAGAS evaluation for the HR Assistant — live API mode.

Answers are collected by calling the running FastAPI server at http://localhost:5000/api/chat.
Contexts (retrieved chunks) are pulled directly from ChromaDB for RAG-path questions.
DB-path questions (leave queries, org charts) return empty contexts and are scored only on
faithfulness and answer_relevancy.

Prerequisites before running:
  1. PostgreSQL must be running.
  2. FastAPI backend must be running: cd hr_assistant/hr_assistant_api && python run.py

Run:
    python evaluation_projects/test_03_ragas.py

Metrics measured:
  - faithfulness       : answer is grounded in retrieved context (no hallucination)
  - answer_relevancy   : answer addresses the question asked
  - context_precision  : retrieved chunks are the relevant ones (low noise)   [RAG-path only]
  - context_recall     : retrieved chunks cover all facts in the ground truth  [RAG-path only]
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
# SECTION 3b: LANGSMITH TRACING CONFIGURATION
#
# WHY: LangSmith is an observability platform built by LangChain. When enabled,
#      it automatically captures every LLM call, embedding call, and retrieval
#      step made by LangChain components and logs them to a web dashboard.
#      This lets you see exactly what happened inside each RAGAS judge call
#      and each ChromaDB retrieval — far more detail than the final score alone.
#
# HOW: LangSmith tracing is activated by setting four environment variables
#      before any LangChain import. LangChain reads these at import time:
#
#   LANGCHAIN_TRACING_V2  = "true"  → turns on tracing globally
#   LANGCHAIN_API_KEY     = "ls__…" → your LangSmith account key
#   LANGCHAIN_PROJECT     = "…"     → groups all runs under one project name
#   LANGCHAIN_ENDPOINT    = "…"     → LangSmith server (default is cloud)
#
#      To get a LangSmith API key:
#        1. Sign up at https://smith.langchain.com (free tier available)
#        2. Go to Settings → API Keys → Create API Key
#        3. Add LANGCHAIN_API_KEY=ls__<your_key> to hr_assistant_api/.env
#        4. Optionally add LANGCHAIN_PROJECT=hr-assistant-ragas-eval
#
#      If LANGCHAIN_API_KEY is absent from .env, tracing is silently skipped —
#      the evaluation still runs normally, just without LangSmith logging.
#
# WHAT GETS TRACED (from this script):
#   - RAGAS judge LLM calls (faithfulness, answer_relevancy, etc.)
#   - RAGAS embedding calls (semantic similarity for answer_relevancy)
#   - ChromaDB retrieval calls (via langchain-chroma)
#   - The build_ragas_dataset() loop (each question as a named trace)
#
# WHAT IS NOT TRACED (runs inside the FastAPI server process):
#   - The actual answer generation (GPT-4 call inside the live API)
#   - To trace that too, add the same env vars to hr_assistant_api/.env
#     and restart the backend server.
# ===========================================================================

_langsmith_key = env.get("LANGCHAIN_API_KEY", "")
LANGSMITH_ENABLED = bool(_langsmith_key)

if LANGSMITH_ENABLED:
    # Set LangChain tracing env vars before any LangChain import so that
    # the SDK picks them up at module-load time.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]     = _langsmith_key
    os.environ["LANGCHAIN_PROJECT"]     = env.get(
        "LANGCHAIN_PROJECT", "hr-assistant-ragas-eval"
    )
    os.environ.setdefault(
        "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
    )

# ===========================================================================
# SECTION 3c: LIVE API CONFIGURATION
#
# WHY: We now call the running FastAPI server instead of the Python classes
#      directly. These two constants control where the API is and which employee
#      ID to use for personalized test cases.
#
# HOW: API_BASE_URL points to the running backend. TEST_EMPLOYEE_ID must be
#      set to a valid employee ID that exists in your PostgreSQL database —
#      this is used for questions in Category 3 (personalized responses).
#      If you don't have an employee to test with, set it to None; those
#      questions will then run without employee context.
# ===========================================================================

API_BASE_URL     = "http://localhost:5000"  # FastAPI backend URL
TEST_EMPLOYEE_ID = 1                        # replace with a real employee ID from your DB


# ===========================================================================
# SECTION 4: IMPORT THE HR ASSISTANT VECTOR STORE SERVICE
#
# WHY: We only import VectorStoreService (not AIService). The vector store is
#      used to retrieve context chunks from ChromaDB for RAG-path questions —
#      the same chunks the API internally uses. AIService is no longer needed
#      here because answers now come from the live HTTP API instead.
#
# NOTE: Pydantic validates settings at import time, so os.environ must be
#       populated (done in Section 2) before this import runs.
# ===========================================================================

from app.services.vector_store import VectorStoreService  # noqa: E402


# ===========================================================================
# SECTION 5: TEST DATASET — 21 questions across 6 categories
#
# Each entry has these fields:
#   "topic"       : display label used in the report
#   "question"    : the message sent to the API
#   "ground_truth": the expected correct answer (from the HR policies PDF or
#                   known DB state) — used by context_precision and context_recall
#   "db_path"     : True if the question triggers a database lookup instead of
#                   ChromaDB retrieval (leave queries, org structure). These
#                   questions get empty contexts and are only scored on
#                   faithfulness and answer_relevancy.
#   "employee_id" : integer employee ID to pass in the API request for
#                   personalized responses. None for non-personalized questions.
#
# CATEGORIES:
#   1–12  : Pure HR policy questions (RAG path, all 4 metrics)
#   13–14 : Live leave queries (DB path, 2 metrics only)
#   15–16 : Org structure queries (DB path, 2 metrics only)
#   17–19 : Personalized policy questions with employee_id (RAG path, all 4 metrics)
#   20–21 : Explicit format tests — table and metric response types (RAG path, all 4 metrics)
# ===========================================================================

TEST_DATASET = [
    # -----------------------------------------------------------------------
    # Category 1: Leave policy (RAG path)
    # -----------------------------------------------------------------------
    {
        "topic": "Leave Policy",
        "question": "When can employees start using their annual leave after being hired?",
        "ground_truth": (
            "Employees begin to accrue annual leave immediately upon hire, "
            "but may not use annual leave until after 90 days of employment."
        ),
        "db_path": False,
        "employee_id": None,
    },
    {
        "topic": "Leave Policy",
        "question": "How much vacation time do employees earn based on years of service?",
        "ground_truth": (
            "Employees earn 1.5 pro-rated days per month for less than 2 years of service, "
            "1.75 pro-rated days per month for years 2 through 6, and 2 pro-rated days per "
            "month for 7 or more years of service."
        ),
        "db_path": False,
        "employee_id": None,
    },
    {
        "topic": "Leave Policy",
        "question": "What happens to unused vacation time when an employee is terminated?",
        "ground_truth": (
            "Any earned but unused vacation will be paid at the time of termination."
        ),
        "db_path": False,
        "employee_id": None,
    },
    {
        "topic": "Leave Policy",
        "question": "How far in advance must employees request vacation approval?",
        "ground_truth": (
            "Employees should request approval in writing at least two weeks in advance "
            "before taking vacation."
        ),
        "db_path": False,
        "employee_id": None,
    },
    # -----------------------------------------------------------------------
    # Category 2: Expense policy (RAG path)
    # -----------------------------------------------------------------------
    {
        "topic": "Expense",
        "question": "What expenses does the organization not reimburse employees for while traveling?",
        "ground_truth": (
            "The organization does not reimburse for personal activities while traveling "
            "or other expenses such as entertainment, liquor, dry cleaning, etc."
        ),
        "db_path": False,
        "employee_id": None,
    },
    {
        "topic": "Expense",
        "question": "How many business days does an employee have to submit a travel expense report after returning from a trip?",
        "ground_truth": (
            "Employees should submit a travel expense report containing receipts within "
            "7 business days of completion of travel."
        ),
        "db_path": False,
        "employee_id": None,
    },
    # -----------------------------------------------------------------------
    # Category 3: Working hours (RAG path)
    # -----------------------------------------------------------------------
    {
        "topic": "Working Hours",
        "question": "What are the standard working hours at the organization?",
        "ground_truth": (
            "Standard working hours are 8:00 a.m. to 4:30 p.m., Monday through Friday, "
            "with an unpaid meal period of thirty minutes."
        ),
        "db_path": False,
        "employee_id": None,
    },
    {
        "topic": "Working Hours",
        "question": "When does the organization's workweek begin and end?",
        "ground_truth": (
            "The workweek begins at 12:00 a.m. Saturday and ends at 11:59 p.m. Friday."
        ),
        "db_path": False,
        "employee_id": None,
    },
    # -----------------------------------------------------------------------
    # Category 4: Benefits and holidays (RAG path)
    # -----------------------------------------------------------------------
    {
        "topic": "Benefits",
        "question": "Which employees are eligible for paid holidays?",
        "ground_truth": (
            "Regular full-time and regular part-time employees are eligible for paid holidays."
        ),
        "db_path": False,
        "employee_id": None,
    },
    {
        "topic": "Benefits",
        "question": "If a paid holiday falls during an employee's scheduled vacation, what happens to their vacation day?",
        "ground_truth": (
            "If a paid holiday falls during an employee's scheduled vacation period, "
            "holiday pay will be provided and the employee will still have a vacation day to use."
        ),
        "db_path": False,
        "employee_id": None,
    },
    # -----------------------------------------------------------------------
    # Category 5: Code of conduct (RAG path)
    # -----------------------------------------------------------------------
    {
        "topic": "Code of Conduct",
        "question": "What forms of harassment does the organization prohibit in the workplace?",
        "ground_truth": (
            "The organization prohibits any form of harassment based on race, color, religion, "
            "sex, national origin, age, disability, veteran status, pregnancy, marital status, "
            "medical condition, sexual orientation, or any other status protected by Federal "
            "and state law."
        ),
        "db_path": False,
        "employee_id": None,
    },
    {
        "topic": "Code of Conduct",
        "question": "What is the organization's policy on outside employment?",
        "ground_truth": (
            "Employees may hold outside employment as long as it does not interfere with "
            "performance or ability to meet the job requirements at the organization."
        ),
        "db_path": False,
        "employee_id": None,
    },
    # -----------------------------------------------------------------------
    # NEW — Category 6: Live leave queries (DB path, db_path=True)
    #
    # WHY ADDED: These questions contain keywords like "on leave" or "absent"
    #   that trigger AIService._handle_leave_query(), which queries PostgreSQL
    #   directly instead of ChromaDB. The previous script couldn't test these
    #   (it had no database connection). With the live API running these are
    #   fully testable. Contexts will be empty; only faithfulness and
    #   answer_relevancy apply.
    # -----------------------------------------------------------------------
    {
        "topic": "Live Leave",
        "question": "Which employees are currently on leave?",
        "ground_truth": (
            "The response should list employees who are currently on approved leave, "
            "including their name, department, leave type, and leave dates."
        ),
        "db_path": True,   # triggers database lookup, not ChromaDB
        "employee_id": None,
    },
    {
        "topic": "Live Leave",
        "question": "Show me all employees who are absent today.",
        "ground_truth": (
            "The response should list employees currently marked as on leave or with "
            "approved leave requests active today, shown in a table format."
        ),
        "db_path": True,
        "employee_id": None,
    },
    # -----------------------------------------------------------------------
    # NEW — Category 7: Org structure queries (DB path, db_path=True)
    #
    # WHY ADDED: These questions trigger the org-chart/tree code path in the
    #   API, which returns a type="tree" response built from the database.
    #   The previous script never exercised this branch. Contexts are empty
    #   (no ChromaDB involved); only faithfulness and answer_relevancy apply.
    # -----------------------------------------------------------------------
    {
        "topic": "Org Structure",
        "question": "Show me the organizational chart of the company.",
        "ground_truth": (
            "The response should present a hierarchical tree showing departments "
            "and the employees within each department."
        ),
        "db_path": True,
        "employee_id": None,
    },
    {
        "topic": "Org Structure",
        "question": "Who reports to whom in the organization?",
        "ground_truth": (
            "The response should show the reporting hierarchy across all departments, "
            "listing managers and their direct reports."
        ),
        "db_path": True,
        "employee_id": None,
    },
    # -----------------------------------------------------------------------
    # NEW — Category 8: Personalized policy questions (RAG path, with employee_id)
    #
    # WHY ADDED: When employee_id is passed in the API request, the server
    #   fetches the employee's details from the database and injects them into
    #   the prompt as context (name, position, department, hire date, status).
    #   The model then personalizes its answer — e.g. calculating vacation
    #   entitlement based on actual hire date. The previous script had no
    #   employee context at all. ChromaDB is still used for retrieval, so
    #   all 4 RAGAS metrics apply.
    # -----------------------------------------------------------------------
    {
        "topic": "Personalized Leave",
        "question": "How many vacation days am I entitled to based on my years of service?",
        "ground_truth": (
            "The entitlement depends on years of service: 1.5 days/month for under 2 years, "
            "1.75 days/month for 2–6 years, and 2 days/month for 7 or more years."
        ),
        "db_path": False,
        "employee_id": TEST_EMPLOYEE_ID,   # API will inject this employee's hire date into prompt
    },
    {
        "topic": "Personalized Status",
        "question": "What is my current employment status and how long have I been with the company?",
        "ground_truth": (
            "The response should state the employee's current status (active, inactive, or "
            "on leave) and calculate their tenure from their hire date to today."
        ),
        "db_path": False,
        "employee_id": TEST_EMPLOYEE_ID,
    },
    {
        "topic": "Personalized Eligibility",
        "question": "Am I eligible for paid holidays?",
        "ground_truth": (
            "Regular full-time and regular part-time employees are eligible for paid holidays. "
            "The answer should confirm eligibility based on the employee's employment type."
        ),
        "db_path": False,
        "employee_id": TEST_EMPLOYEE_ID,
    },
    # -----------------------------------------------------------------------
    # NEW — Category 9: Explicit response format tests (RAG path)
    #
    # WHY ADDED: The API can return four response types: text, table, metric,
    #   tree. The previous tests only exercised "text". These questions
    #   explicitly request table and metric formats to verify that
    #   extract_answer_text() handles those branches correctly and that RAGAS
    #   can score them. ChromaDB retrieval is used, so all 4 metrics apply.
    # -----------------------------------------------------------------------
    {
        "topic": "Format: Table",
        "question": "Can you show me a table of all leave types and their descriptions?",
        "ground_truth": (
            "The response should list leave types such as annual leave, sick leave, "
            "and personal leave, along with a brief description of each."
        ),
        "db_path": False,
        "employee_id": None,
    },
    {
        "topic": "Format: Metric",
        "question": "How many paid holidays does the organization observe per year?",
        "ground_truth": (
            "The organization observes paid holidays each year; the exact number "
            "should be stated clearly as a numeric value."
        ),
        "db_path": False,
        "employee_id": None,
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
#
# NOTE: context_precision and context_recall are only scored for RAG-path
#   questions (db_path=False). Aggregate scores for these two metrics are
#   computed only over questions where they are not NaN.
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
# WHY: The HR Assistant API returns a JSON string inside the "response" field,
#      not plain text. For example:
#          '{"type": "text", "data": "The leave policy states...", "meta": {...}}'
#      RAGAS expects a plain English string as the answer. This function peels
#      off the JSON wrapper and returns only the human-readable content.
#
# HOW: It parses the JSON, reads the "type" field to know the response shape,
#      then extracts and formats the "data" field accordingly:
#   - "text"   → the data value is already a plain string, return as-is
#   - "table"  → data contains columns + rows; flatten into a readable text table
#   - "metric" → data contains a label + number; format as "Label: 42"
#   - "tree"   → org-chart hierarchy; convert to string representation
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
#      This function builds those lists by calling the live FastAPI server for
#      answers and querying ChromaDB directly for contexts.
#
# HOW: For each question it makes up to two calls:
#
#   Step A — Contexts (RAG-path questions only):
#     vs.query(question, n_results=5) asks ChromaDB to find the 5 most
#     relevant text chunks from the HR policies PDF. For DB-path questions
#     (db_path=True), contexts is set to an empty list because ChromaDB is
#     bypassed — the API queries PostgreSQL directly for those.
#
#   Step B — Answer (always from the live HTTP API):
#     httpx.post(API_BASE_URL + "/api/chat", json=payload) sends the question
#     to the running FastAPI server. The response is {"response": "<JSON>"},
#     so we unwrap one extra layer compared to the old direct-service approach.
#     employee_id is included in the payload for personalized questions.
#
#   Step C — Extract plain text:
#     extract_answer_text() strips the inner JSON wrapper to get plain English.
#
#   time.sleep(0.5) paces requests to avoid hitting OpenAI rate limits on
#   the server side.
#
# RETURNS: A plain dict with four keys, each holding a list of 21 items.
#          This is the exact shape the HuggingFace Dataset class expects.
# ===========================================================================

from langsmith import traceable as _traceable  # no-op when LANGCHAIN_TRACING_V2 is not set


@_traceable(name="build_ragas_dataset")
def build_ragas_dataset(vs: VectorStoreService) -> dict:
    # @traceable records this entire function as a named run in LangSmith.
    # Every LangChain call inside the loop (ChromaDB similarity search) appears
    # as a child span under it, giving a per-question view in the dashboard.
    # When LANGCHAIN_TRACING_V2 is not set the decorator is a no-op.
    import httpx

    # langsmith.trace creates a named child span for each question in LangSmith.
    # This replaces the generic "row 0, row 1" labels with the actual question
    # text and topic. It is a no-op when LANGCHAIN_TRACING_V2 is not set.
    from langsmith import trace as ls_trace

    questions, answers, contexts_list, ground_truths = [], [], [], []
    n = len(TEST_DATASET)

    for i, item in enumerate(TEST_DATASET):
        q          = item["question"]
        is_db_path = item.get("db_path", False)
        emp_id     = item.get("employee_id")
        path_label = "DB" if is_db_path else "RAG"

        # Build a meaningful LangSmith span name: "[01/21] [RAG] Leave Policy — When can..."
        # This is what appears in the LangSmith dashboard instead of "row 0", "row 1", etc.
        span_name = f"[{i + 1:02d}/{n}] [{path_label}] {item['topic']} — {q[:50]}"

        print(f"  [{i + 1:02d}/{n}] [{path_label}] [{item['topic']}] {q[:60]}...")

        with ls_trace(
            name=span_name,
            # inputs appear in the LangSmith "Inputs" panel for this span
            inputs={
                "question":    q,
                "topic":       item["topic"],
                "path":        path_label,
                "employee_id": emp_id,
            },
            # tags let you filter runs in LangSmith by topic or path type
            tags=[item["topic"], path_label],
        ):
            # Step A: Retrieve context chunks from ChromaDB for RAG-path questions.
            #         DB-path questions (leave/org) bypass ChromaDB entirely — the API
            #         queries PostgreSQL for those, so contexts is intentionally empty.
            if is_db_path:
                contexts = []
            else:
                contexts = vs.query(q, n_results=5)

            # Step B: Call the live FastAPI server to get the generated answer.
            #         The payload matches the ChatRequest schema: message + optional employee_id.
            payload = {"message": q, "employee_id": emp_id}
            try:
                resp = httpx.post(
                    f"{API_BASE_URL}/api/chat",
                    json=payload,
                    timeout=30.0,   # 30 s — generous for slow model responses
                )
                resp.raise_for_status()
                # The API wraps the AI response in {"response": "<JSON string>"}
                raw = resp.json()["response"]
            except httpx.HTTPError as exc:
                print(f"    WARNING: API call failed ({exc}). Recording empty answer.")
                raw = json.dumps({"type": "text", "data": ""})

            # Step C: Strip the JSON wrapper to get plain text for RAGAS.
            answer = extract_answer_text(raw)

        questions.append(q)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(item["ground_truth"])

        time.sleep(0.5)  # brief pause to respect OpenAI rate limits on the server side

    return {
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts_list,   # empty list for DB-path questions
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
#   - df[key].mean() computes the average score — NaN values (from DB-path
#     questions where context metrics don't apply) are automatically skipped
#     by pandas .mean(), so the aggregate only counts scoreable questions.
#   - A PATH column ("RAG" or "DB ") is shown for each row so it's immediately
#     clear why context_precision and context_recall show N/A for DB questions.
#   - pd.isna() guards against NaN values RAGAS produces when a metric
#     couldn't be computed (DB-path questions, or any API error).
#
# RETURNS: agg — a dict of {metric_name: average_score} used later when saving
#          the JSON results file.
# ===========================================================================

def print_report(result, run_date: str, model: str, num_questions: int) -> dict:
    import pandas as pd

    width = 75
    print("\n" + "=" * width)
    print("RAGAS Evaluation Report — HR Assistant (Live API Mode)")
    print(f"Run Date : {run_date}")
    print(f"Model    : {model}")
    print(f"API      : {API_BASE_URL}")
    print(f"Questions: {num_questions}  ({sum(1 for x in TEST_DATASET if not x.get('db_path'))} RAG-path, "
          f"{sum(1 for x in TEST_DATASET if x.get('db_path'))} DB-path)")
    print("=" * width)

    # Convert RAGAS result into a DataFrame (one row per question, one column per metric).
    df = result.to_pandas()

    metric_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    # Compute average scores. pandas .mean() skips NaN automatically, so DB-path
    # questions with empty contexts don't drag down the context metric averages.
    agg = {}
    for key in metric_keys:
        if key in df.columns:
            agg[key] = df[key].mean()

    # Print the aggregate summary with pass/fail verdict.
    print("\nAGGREGATE SCORES (NaN rows excluded from context metric averages):")
    for metric, score in agg.items():
        threshold = THRESHOLDS.get(metric, 0.0)
        if pd.isna(score):
            print(f"  {metric:<22}: N/A")
        else:
            status = "PASS" if score >= threshold else "FAIL"
            print(f"  {metric:<22}: {score:.4f}  [target >= {threshold:.2f}]  {status}")

    # Print the per-question breakdown table.
    # PATH column shows RAG or DB so it's clear why some rows have N/A context scores.
    print("\nPER-QUESTION BREAKDOWN:")
    header = (
        f"  {'#':>2}  {'Path':<4}  {'Topic':<20}  {'Q (truncated)':<38}  "
        + "  ".join(f"{k[:8]:>8}" for k in metric_keys)
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for idx, row in df.iterrows():
        item    = TEST_DATASET[idx]
        path    = "DB " if item.get("db_path") else "RAG"
        topic   = item["topic"][:20]
        q_short = item["question"][:38]
        scores_str = "  ".join(
            f"{row[k]:>8.4f}" if k in df.columns and not pd.isna(row[k]) else f"{'N/A':>8}"
            for k in metric_keys
        )
        print(f"  {idx + 1:>2}  {path:<4}  {topic:<20}  {q_short:<38}  {scores_str}")

    print("=" * width)
    print("  NOTE: context_precision and context_recall show N/A for DB-path questions.")
    print("        This is expected — those questions bypass ChromaDB retrieval entirely.")
    return agg


# ===========================================================================
# FUNCTION: main
#
# WHY: Orchestrates all the steps in the correct order:
#        health check → init ChromaDB → build dataset → evaluate → report → save
#      Keeping this in a function (rather than at module level) means the
#      evaluation only runs when you explicitly execute the script, not when
#      another file imports it.
#
# HOW: Six numbered steps.
# ===========================================================================

def main():
    import httpx

    print("\n=== HR Assistant — RAGAS Evaluation (Live API Mode) ===\n")

    # Print LangSmith status so the user knows whether tracing is active.
    if LANGSMITH_ENABLED:
        project = os.environ.get("LANGCHAIN_PROJECT", "hr-assistant-ragas-eval")
        print(f"LangSmith tracing : ENABLED  (project: '{project}')")
        print(f"  Dashboard        : https://smith.langchain.com")
        print(f"  Note: API answer-generation calls are traced only if the")
        print(f"        FastAPI server also has LANGCHAIN_TRACING_V2=true.\n")
    else:
        print("LangSmith tracing : DISABLED (add LANGCHAIN_API_KEY to .env to enable)\n")

    # ------------------------------------------------------------------
    # Step 1: Health check — verify the FastAPI server is reachable.
    #
    # WHY: If the server isn't running, every API call in build_ragas_dataset()
    #      would fail with a connection error after a 30-second timeout,
    #      wasting several minutes. Failing fast here gives a clear message
    #      telling the user exactly what to do.
    # ------------------------------------------------------------------
    print(f"Checking API health at {API_BASE_URL}/health ...")
    try:
        health = httpx.get(f"{API_BASE_URL}/health", timeout=5.0)
        health.raise_for_status()
        print(f"API is up: {health.json()}\n")
    except Exception as exc:
        print(f"\nERROR: Cannot reach the FastAPI server at {API_BASE_URL}.")
        print(f"  Detail: {exc}")
        print(f"\nPlease start the backend first:")
        print(f"  cd hr_assistant/hr_assistant_api")
        print(f"  python run.py\n")
        return

    # ------------------------------------------------------------------
    # Step 2: Initialise ChromaDB for context retrieval (RAG-path only).
    #
    # VectorStoreService connects to the existing ChromaDB on disk.
    # load_pdf() checks if documents are already stored; if the ChromaDB
    # folder already has 203 chunks, it skips re-ingestion immediately.
    # We only need this for RAG-path questions — DB-path questions get
    # empty contexts and don't touch ChromaDB at all.
    # ------------------------------------------------------------------
    print("Initialising ChromaDB for context retrieval...")
    vs = VectorStoreService(persist_directory=CHROMA_DIR, openai_api_key=OPENAI_API_KEY)
    vs.load_pdf(PDF_PATH)

    # Read the model name from .env (previously we got it from AIService.model).
    model_name = env.get("OPENAI_MODEL_NAME", "unknown")
    print(f"ChromaDB ready. Judge model: {model_name}\n")

    # ------------------------------------------------------------------
    # Step 3: Call the live API for every question and collect contexts.
    #
    # build_ragas_dataset() makes one HTTP POST per question to the live
    # FastAPI server, and one ChromaDB query per RAG-path question.
    # Returns a plain dict ready to be wrapped in a HuggingFace Dataset.
    # ------------------------------------------------------------------
    print(f"Building evaluation dataset ({len(TEST_DATASET)} questions via live API)...")
    data = build_ragas_dataset(vs)
    print(f"\nDataset built: {len(data['question'])} samples.\n")

    # ------------------------------------------------------------------
    # Step 4: Feed the dataset into RAGAS for evaluation.
    #
    # Dataset.from_dict() wraps the plain dict into a HuggingFace Dataset.
    #
    # Metrics from ragas.metrics are instantiated without arguments and the
    # LLM + embeddings are passed to evaluate() once.
    #
    # raise_exceptions=False → RAGAS records NaN instead of crashing when
    # a metric can't be scored (e.g. empty contexts for DB-path questions).
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
    # NOT the model running inside the API that generated those answers.
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    judge_llm     = llm_factory(model_name, client=openai_client)

    # AnswerRelevancy measures semantic similarity between question and answer,
    # so it needs an embeddings model in addition to the LLM.
    judge_emb = LangchainEmbeddingsWrapper(
        LCOpenAIEmbeddings(
            model=env.get("EMBEDDING_MODEL_NAME", "text-embedding-3-small"),
            openai_api_key=OPENAI_API_KEY,
        )
    )

    # experiment_name labels this entire RAGAS evaluation run in LangSmith.
    # Each run gets a timestamp so you can compare runs over time (e.g. before
    # and after changing chunk size or switching models).
    # Format: "hr-ragas-20260515_163224" — visible at the top of the trace.
    experiment_name = f"hr-ragas-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    ds = Dataset.from_dict(data)
    result = evaluate(
        ds,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
        llm=judge_llm,
        embeddings=judge_emb,
        raise_exceptions=False,
        experiment_name=experiment_name,  # names the run in LangSmith instead of a random UUID
    )
    print("Evaluation complete.\n")

    # ------------------------------------------------------------------
    # Step 5: Print the human-readable report to the console.
    # ------------------------------------------------------------------
    run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    agg = print_report(result, run_date, model_name, len(data["question"]))

    # ------------------------------------------------------------------
    # Step 6: Save the full results to a timestamped JSON file.
    #
    # Each entry in per_question includes the path type (RAG or DB) so the
    # JSON file is self-explanatory for later analysis or sharing.
    # pd.isna() converts NaN to None so the JSON stays valid.
    # ------------------------------------------------------------------
    import pandas as pd
    df = result.to_pandas()
    per_question = []
    metric_keys  = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    for idx, row in df.iterrows():
        item = TEST_DATASET[idx]
        per_question.append({
            "index":       idx + 1,
            "path":        "DB" if item.get("db_path") else "RAG",
            "topic":       item["topic"],
            "question":    item["question"],
            "employee_id": item.get("employee_id"),
            "answer":      data["answer"][idx],
            "ground_truth": item["ground_truth"],
            **{
                k: (None if pd.isna(row[k]) else round(float(row[k]), 4))
                for k in metric_keys
                if k in df.columns
            },
        })

    output = {
        "run_date":         run_date,
        "model":            model_name,
        "embedding_model":  env.get("EMBEDDING_MODEL_NAME", "unknown"),
        "api_base_url":     API_BASE_URL,
        "num_questions":    len(data["question"]),
        "rag_path_count":   sum(1 for x in TEST_DATASET if not x.get("db_path")),
        "db_path_count":    sum(1 for x in TEST_DATASET if x.get("db_path")),
        "aggregate_scores": {
            k: (None if pd.isna(v) else round(v, 4)) for k, v in agg.items()
        },
        "thresholds":       THRESHOLDS,
        "per_question":     per_question,
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
