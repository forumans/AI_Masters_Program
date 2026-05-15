"""
RAGAS evaluation for the HR Assistant — live API mode.

Architecture
------------
This script is fully independent of the hr_assistant project:
  - Reads ALL config from evaluation_projects/.env (single source of truth)
  - Connects to ChromaDB directly via langchain-chroma (no hr_assistant code)
  - Gets answers via HTTP calls to the live FastAPI server
  - No sys.path manipulation, no hr_assistant imports, no shared .env files

Prerequisites before running:
  1. PostgreSQL must be running.
  2. FastAPI backend: cd hr_assistant/hr_assistant_api && python run.py
  3. evaluation_projects/.env must be populated (see .env for all keys).

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
import time
from datetime import datetime
from pathlib import Path


# ===========================================================================
# SECTION 1: PATH SETUP
#
# WHY: We need the absolute path of this script to locate evaluation_projects/.env
#      and to resolve the CHROMA_PERSIST_DIR relative to the repo root.
# ===========================================================================

SCRIPT_DIR = Path(__file__).resolve().parent   # .../evaluation_projects/
REPO_ROOT  = SCRIPT_DIR.parent                 # .../GenAI_Masters_Program/


# ===========================================================================
# SECTION 2: LOAD EVALUATION CONFIG — single .env, no hr_assistant reads
#
# WHY: evaluation_projects/.env is the ONE place that controls the evaluation.
#      No other config file is read. This makes the evaluation project fully
#      independent — you can run it against any hr_assistant deployment by
#      just changing the values in evaluation_projects/.env.
#
# HOW: dotenv_values() returns a plain dict without modifying os.environ.
#      We push only the LangSmith keys into os.environ later (Section 3b)
#      because LangChain reads those at import time from the environment.
# ===========================================================================

from dotenv import dotenv_values  # noqa: E402

eval_env = dotenv_values(SCRIPT_DIR / ".env")


# ===========================================================================
# SECTION 3: CONSTANTS — all sourced from evaluation_projects/.env
#
# OPENAI_API_KEY      : used to embed questions when querying ChromaDB
# CHROMA_DIR          : absolute path to the ChromaDB folder on disk
# CHROMA_COLLECTION   : name of the ChromaDB collection holding policy chunks
# EMBEDDING_MODEL_NAME: embedding model used to query ChromaDB (must match
#                       the model used when the collection was originally built)
# API_BASE_URL        : the running FastAPI server — answers come from here
# TEST_EMPLOYEE_ID    : employee ID for personalized questions (Category 8)
# ANSWER_MODEL_NAME   : display-only label for the model inside the API
# ===========================================================================

OPENAI_API_KEY       = eval_env.get("OPENAI_API_KEY", "")
EMBEDDING_MODEL_NAME = eval_env.get("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
CHROMA_COLLECTION    = eval_env.get("CHROMA_COLLECTION_NAME", "hr_policies_collection")
CHROMA_DIR           = str(REPO_ROOT / eval_env.get(
    "CHROMA_PERSIST_DIR", "hr_assistant/hr_assistant_api/chroma_data"
))
API_BASE_URL         = eval_env.get("API_BASE_URL", "http://localhost:5000")
TEST_EMPLOYEE_ID     = eval_env.get("TEST_EMPLOYEE_ID", 1)
ANSWER_MODEL_NAME    = eval_env.get("ANSWER_MODEL_NAME", "unknown")


# ===========================================================================
# SECTION 3b: LANGSMITH TRACING CONFIGURATION
#
# WHY: LangSmith is an observability platform. When enabled, it captures every
#      LLM call, embedding call, and retrieval step made by LangChain components
#      and logs them to a web dashboard — useful for diagnosing low RAGAS scores.
#
# HOW: LangChain reads the four LANGCHAIN_* env vars at import time. We must
#      set them in os.environ BEFORE any LangChain import happens. All four
#      keys come from evaluation_projects/.env — the hr_assistant project is
#      not involved.
#
# SETUP: Add to evaluation_projects/.env:
#   LANGCHAIN_API_KEY=ls__...   ← from https://smith.langchain.com → API Keys
#   LANGCHAIN_PROJECT=hr-assistant-ragas-eval
#
# WHAT GETS TRACED: RAGAS judge LLM calls, embedding calls, ChromaDB queries.
# WHAT IS NOT TRACED: answer generation inside the FastAPI server (traced only
#   if you also add LANGCHAIN_TRACING_V2=true to hr_assistant_api/.env and
#   restart the server).
# ===========================================================================

_langsmith_key = eval_env.get("LANGCHAIN_API_KEY", "")
LANGSMITH_ENABLED = bool(_langsmith_key)

if LANGSMITH_ENABLED:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]     = _langsmith_key
    os.environ["LANGCHAIN_PROJECT"]     = eval_env.get(
        "LANGCHAIN_PROJECT", "hr-assistant-ragas-eval"
    )
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")


# ===========================================================================
# SECTION 3c: LIVE API CONFIGURATION
#
# API_BASE_URL and TEST_EMPLOYEE_ID are set above from eval_env.
# They are documented here for clarity.
#
# API_BASE_URL    : the running FastAPI backend — all answers come from HTTP
#                   calls to this URL; no hr_assistant Python code is invoked
# TEST_EMPLOYEE_ID: passed as employee_id in the ChatRequest payload for
#                   Category 8 personalized questions
# ===========================================================================


# ===========================================================================
# SECTION 3d: JUDGE MODEL CONFIGURATION (provider-driven from eval_env)
#
# WHY: The judge model scores answers independently of the model that generated
#      them. All settings live in evaluation_projects/.env so you can swap
#      providers (Gemini ↔ OpenAI) without touching any code.
#
# HOW: Set in evaluation_projects/.env:
#
#   # Gemini (avoids self-grading bias vs OpenAI-generated answers)
#   JUDGE_LLM_PROVIDER=gemini
#   JUDGE_LLM_MODEL=gemini-2.0-flash
#   JUDGE_EMB_MODEL=models/text-embedding-004
#   GEMINI_API_KEY=AIza...
#
#   # OpenAI (simpler — no extra key needed)
#   JUDGE_LLM_PROVIDER=openai
#   JUDGE_LLM_MODEL=gpt-4o-mini
#   JUDGE_EMB_MODEL=text-embedding-3-small
#
# DEFAULTS: falls back to Gemini if JUDGE_LLM_PROVIDER is not set.
# ===========================================================================

JUDGE_LLM_PROVIDER = eval_env.get("JUDGE_LLM_PROVIDER", "gemini").lower().strip()
GOOGLE_API_KEY     = eval_env.get("GOOGLE_API_KEY") or eval_env.get("GEMINI_API_KEY", "")

_PROVIDER_DEFAULTS = {
    "gemini": {"llm": "gemini-2.0-flash",  "emb": "models/text-embedding-004"},
    "openai": {"llm": "gpt-4o-mini",       "emb": "text-embedding-3-small"},
}

JUDGE_LLM_MODEL = eval_env.get(
    "JUDGE_LLM_MODEL",
    _PROVIDER_DEFAULTS.get(JUDGE_LLM_PROVIDER, _PROVIDER_DEFAULTS["gemini"])["llm"],
)
JUDGE_EMB_MODEL = eval_env.get(
    "JUDGE_EMB_MODEL",
    _PROVIDER_DEFAULTS.get(JUDGE_LLM_PROVIDER, _PROVIDER_DEFAULTS["gemini"])["emb"],
)


# ===========================================================================
# SECTION 4: CHROMADB CONNECTION
#
# WHY: RAGAS needs the retrieved chunks ("contexts") for context_precision
#      and context_recall. The live API doesn't expose these in its response,
#      so we query ChromaDB directly. ChromaDB is a shared data asset — the
#      evaluation connects to it the same way any other client would, without
#      importing any hr_assistant Python code.
#
# HOW: We use langchain-chroma directly (already installed in the venv).
#      The embedding model and collection name must match what was used when
#      the HR policies PDF was originally ingested — set in evaluation_projects/.env.
#      The connection is read-only (similarity_search only, no writes).
#
# NOTE: DB-path questions (leave/org queries) bypass ChromaDB entirely —
#       the API queries PostgreSQL for those. Those questions get empty
#       contexts and are scored only on faithfulness and answer_relevancy.
# ===========================================================================

from langchain_chroma import Chroma                   # noqa: E402
from langchain_openai import OpenAIEmbeddings         # noqa: E402


def _connect_chroma() -> Chroma:
    """Open a read-only connection to the shared ChromaDB vector store."""
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=OpenAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            openai_api_key=OPENAI_API_KEY,
        ),
        collection_name=CHROMA_COLLECTION,
    )


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
#   "employee_id" : integer employee ID for personalized responses, None otherwise
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
        "ground_truth": "Any earned but unused vacation will be paid at the time of termination.",
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
        "ground_truth": "The workweek begins at 12:00 a.m. Saturday and ends at 11:59 p.m. Friday.",
        "db_path": False,
        "employee_id": None,
    },
    # -----------------------------------------------------------------------
    # Category 4: Benefits and holidays (RAG path)
    # -----------------------------------------------------------------------
    {
        "topic": "Benefits",
        "question": "Which employees are eligible for paid holidays?",
        "ground_truth": "Regular full-time and regular part-time employees are eligible for paid holidays.",
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
    # NEW — Category 6: Live leave queries (DB path)
    # Trigger the database lookup path in the API — contexts are empty.
    # -----------------------------------------------------------------------
    {
        "topic": "Live Leave",
        "question": "Which employees are currently on leave?",
        "ground_truth": (
            "The response should list employees who are currently on approved leave, "
            "including their name, department, leave type, and leave dates."
        ),
        "db_path": True,
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
    # NEW — Category 7: Org structure queries (DB path)
    # Trigger the tree/hierarchy path in the API — contexts are empty.
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
    # NEW — Category 8: Personalized policy questions (RAG path + employee_id)
    # The API injects the employee's DB record into the prompt context.
    # -----------------------------------------------------------------------
    {
        "topic": "Personalized Leave",
        "question": "How many vacation days am I entitled to based on my years of service?",
        "ground_truth": (
            "The entitlement depends on years of service: 1.5 days/month for under 2 years, "
            "1.75 days/month for 2-6 years, and 2 days/month for 7 or more years."
        ),
        "db_path": False,
        "employee_id": TEST_EMPLOYEE_ID,
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
    # Verify extract_answer_text() handles table and metric response types.
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
# Minimum acceptable score per metric. Scores at or above → PASS, below → FAIL.
# context_precision and context_recall are only scored for RAG-path questions;
# pandas .mean() skips NaN rows automatically so DB-path questions don't drag
# down the aggregate for those two metrics.
# ===========================================================================

THRESHOLDS = {
    "faithfulness":      0.80,
    "answer_relevancy":  0.80,
    "context_precision": 0.70,
    "context_recall":    0.75,
}


# ===========================================================================
# FUNCTION: extract_answer_text
#
# WHY: The API returns a JSON string in the "response" field, not plain text:
#          '{"type": "text", "data": "...", "meta": {...}}'
#      RAGAS expects a plain English string. This function unwraps the JSON.
#
# HOW: Reads the "type" field and extracts "data" accordingly:
#   "text"   → return data string directly
#   "table"  → flatten columns + rows into a readable plain-text table
#   "metric" → format as "Label: value"
#   "tree"   → convert to string (fallback)
# ===========================================================================

def extract_answer_text(raw_response: str) -> str:
    try:
        parsed = json.loads(raw_response)
        response_type = parsed.get("type", "text")
        data = parsed.get("data", "")

        if response_type == "text":
            return str(data)

        if response_type == "table" and isinstance(data, dict):
            columns = data.get("columns", [])
            rows    = data.get("rows", [])
            lines   = [" | ".join(str(c) for c in columns)]
            for row in rows:
                lines.append(" | ".join(str(v) for v in row))
            return "\n".join(lines)

        if response_type == "metric" and isinstance(data, dict):
            return f"{data.get('label', '')}: {data.get('value', '')}"

        return str(data)

    except (json.JSONDecodeError, AttributeError):
        return raw_response


# ===========================================================================
# FUNCTION: build_ragas_dataset
#
# WHY: Builds the four parallel lists RAGAS needs: question, answer, contexts,
#      ground_truth — one entry per test case.
#
# HOW: For each question:
#   Step A — Contexts (RAG-path only):
#     chroma.similarity_search(question, k=5) returns the top-5 matching
#     chunks from ChromaDB. DB-path questions get empty contexts because
#     the API queries PostgreSQL directly for those — ChromaDB is not involved.
#
#   Step B — Answer (always from the live HTTP API):
#     httpx.post(API_BASE_URL + "/api/chat", json=payload) — one HTTP call per
#     question. The response is {"response": "<JSON string>"}, so we unwrap one
#     extra layer before calling extract_answer_text().
#
#   Step C — Extract plain text for RAGAS scoring.
#
# @traceable creates a named LangSmith span per question (visible in the
# dashboard instead of generic "row 0, row 1" labels). No-op when LangSmith
# is not configured.
# ===========================================================================

from langsmith import traceable as _traceable  # noqa: E402


@_traceable(name="build_ragas_dataset")
def build_ragas_dataset(chroma: Chroma) -> dict:
    from langsmith import trace as ls_trace
    import httpx

    questions, answers, contexts_list, ground_truths = [], [], [], []
    n = len(TEST_DATASET)

    for i, item in enumerate(TEST_DATASET):
        q          = item["question"]
        is_db_path = item.get("db_path", False)
        emp_id     = item.get("employee_id")
        path_label = "DB" if is_db_path else "RAG"

        span_name = f"[{i + 1:02d}/{n}] [{path_label}] {item['topic']} — {q[:50]}"
        print(f"  [{i + 1:02d}/{n}] [{path_label}] [{item['topic']}] {q[:60]}...")

        with ls_trace(
            name=span_name,
            inputs={"question": q, "topic": item["topic"], "path": path_label, "employee_id": emp_id},
            tags=[item["topic"], path_label],
        ):
            # Step A: context retrieval — RAG-path only.
            # DB-path questions (leave/org) bypass ChromaDB; the API queries
            # PostgreSQL for those, so contexts is intentionally empty.
            if is_db_path:
                contexts = []
            else:
                docs     = chroma.similarity_search(q, k=5)
                contexts = [doc.page_content for doc in docs]

            # Step B: call the live API for the generated answer.
            payload = {"message": q, "employee_id": emp_id}
            try:
                resp = httpx.post(
                    f"{API_BASE_URL}/api/chat",
                    json=payload,
                    timeout=30.0,
                )
                resp.raise_for_status()
                raw = resp.json()["response"]
            except httpx.HTTPError as exc:
                print(f"    WARNING: API call failed ({exc}). Recording empty answer.")
                raw = json.dumps({"type": "text", "data": ""})

            # Step C: strip the JSON wrapper to get plain text for RAGAS.
            answer = extract_answer_text(raw)

        questions.append(q)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(item["ground_truth"])

        time.sleep(0.5)

    return {
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts_list,
        "ground_truth": ground_truths,
    }


# ===========================================================================
# FUNCTION: _build_judge
#
# WHY: Constructs the RAGAS judge LLM and embeddings based on the provider
#      configured in evaluation_projects/.env. Centralising this here keeps
#      main() clean and makes adding new providers a one-place change.
#
# SUPPORTED PROVIDERS:
#   "gemini" → ChatGoogleGenerativeAI + GoogleGenerativeAIEmbeddings
#              requires GEMINI_API_KEY (or GOOGLE_API_KEY) in eval .env
#   "openai" → ChatOpenAI + OpenAIEmbeddings
#              uses OPENAI_API_KEY already in eval .env
# ===========================================================================

def _build_judge():
    import warnings
    # LangchainLLMWrapper and LangchainEmbeddingsWrapper are deprecated in favour
    # of ragas.llms.llm_factory, which currently only supports OpenAI clients.
    # We suppress these warnings until RAGAS adds llm_factory support for Gemini.
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    if JUDGE_LLM_PROVIDER == "gemini":
        if not GOOGLE_API_KEY:
            print("ERROR: JUDGE_LLM_PROVIDER=gemini but GEMINI_API_KEY is missing from evaluation_projects/.env.")
            print("  Get a free key at https://aistudio.google.com → Get API key")
            raise SystemExit(1)
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        llm = LangchainLLMWrapper(
            ChatGoogleGenerativeAI(model=JUDGE_LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0)
        )
        emb = LangchainEmbeddingsWrapper(
            GoogleGenerativeAIEmbeddings(model=JUDGE_EMB_MODEL, google_api_key=GOOGLE_API_KEY)
        )

    elif JUDGE_LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings as LCOpenAIEmbeddings
        llm = LangchainLLMWrapper(
            ChatOpenAI(model=JUDGE_LLM_MODEL, openai_api_key=OPENAI_API_KEY, temperature=0)
        )
        emb = LangchainEmbeddingsWrapper(
            LCOpenAIEmbeddings(model=JUDGE_EMB_MODEL, openai_api_key=OPENAI_API_KEY)
        )

    else:
        print(f"ERROR: Unknown JUDGE_LLM_PROVIDER '{JUDGE_LLM_PROVIDER}' in evaluation_projects/.env.")
        print("  Supported values: gemini, openai")
        raise SystemExit(1)

    return llm, emb


# ===========================================================================
# FUNCTION: print_report
#
# WHY: Formats the raw RAGAS result into a human-friendly console table with
#      aggregate scores (pass/fail) and a per-question breakdown.
#
# HOW:
#   result.to_pandas() → DataFrame: one row per question, one column per metric.
#   df[key].mean()     → average score; NaN rows (DB-path questions) are
#                        automatically skipped by pandas so they don't skew
#                        the context metric averages.
#   PATH column        → "RAG" or "DB" explains why some rows show N/A.
# ===========================================================================

def print_report(result, run_date: str, num_questions: int) -> dict:
    import pandas as pd

    width = 75
    print("\n" + "=" * width)
    print("RAGAS Evaluation Report — HR Assistant (Live API Mode)")
    print(f"Run Date    : {run_date}")
    print(f"Answer model: {ANSWER_MODEL_NAME}  |  Judge: {JUDGE_LLM_MODEL} ({JUDGE_LLM_PROVIDER})")
    print(f"API         : {API_BASE_URL}")
    print(f"Questions   : {num_questions}  "
          f"({sum(1 for x in TEST_DATASET if not x.get('db_path'))} RAG-path, "
          f"{sum(1 for x in TEST_DATASET if x.get('db_path'))} DB-path)")
    print("=" * width)

    df = result.to_pandas()
    metric_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    agg = {k: df[k].mean() for k in metric_keys if k in df.columns}

    print("\nAGGREGATE SCORES (NaN rows excluded from context metric averages):")
    for metric, score in agg.items():
        if pd.isna(score):
            print(f"  {metric:<22}: N/A")
        else:
            threshold = THRESHOLDS.get(metric, 0.0)
            status = "PASS" if score >= threshold else "FAIL"
            print(f"  {metric:<22}: {score:.4f}  [target >= {threshold:.2f}]  {status}")

    print("\nPER-QUESTION BREAKDOWN:")
    header = (
        f"  {'#':>2}  {'Path':<4}  {'Topic':<20}  {'Q (truncated)':<38}  "
        + "  ".join(f"{k[:8]:>8}" for k in metric_keys)
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for idx, row in df.iterrows():
        item    = TEST_DATASET[idx]
        path    = "DB" if item.get("db_path") else "RAG"
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
# FUNCTION: main — orchestrates all steps
#
# Steps:
#   1. Health check    — fail fast if the FastAPI server is not running
#   2. ChromaDB init   — connect directly, no hr_assistant code
#   3. Build dataset   — HTTP calls to API + direct ChromaDB queries
#   4. RAGAS evaluate  — LLM-as-judge scoring
#   5. Print report    — console output
#   6. Save JSON       — timestamped results file
# ===========================================================================

def main():
    import httpx

    print("\n=== HR Assistant — RAGAS Evaluation (Live API Mode) ===\n")

    if LANGSMITH_ENABLED:
        project = os.environ.get("LANGCHAIN_PROJECT", "hr-assistant-ragas-eval")
        print(f"LangSmith tracing : ENABLED  (project: '{project}')")
        print(f"  Dashboard       : https://smith.langchain.com\n")
    else:
        print("LangSmith tracing : DISABLED (add LANGCHAIN_API_KEY to evaluation_projects/.env)\n")

    # ------------------------------------------------------------------
    # Step 1: Health check — fail fast with a clear message.
    # ------------------------------------------------------------------
    print(f"Checking API at {API_BASE_URL}/health ...")
    try:
        health = httpx.get(f"{API_BASE_URL}/health", timeout=5.0)
        health.raise_for_status()
        print(f"API is up: {health.json()}\n")
    except Exception as exc:
        print(f"\nERROR: Cannot reach {API_BASE_URL}. Start the backend first:")
        print(f"  cd hr_assistant/hr_assistant_api && python run.py")
        print(f"  Detail: {exc}\n")
        return

    # ------------------------------------------------------------------
    # Step 2: Connect to ChromaDB directly — no hr_assistant imports.
    # Uses CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME
    # and OPENAI_API_KEY from evaluation_projects/.env.
    # ------------------------------------------------------------------
    print(f"Connecting to ChromaDB at {CHROMA_DIR} ...")
    chroma = _connect_chroma()
    count  = chroma._collection.count()
    print(f"ChromaDB ready: {count} chunks in '{CHROMA_COLLECTION}'\n")

    # ------------------------------------------------------------------
    # Step 3: Build the evaluation dataset.
    # ------------------------------------------------------------------
    print(f"Building evaluation dataset ({len(TEST_DATASET)} questions via live API)...")
    data = build_ragas_dataset(chroma)
    print(f"\nDataset built: {len(data['question'])} samples.\n")

    # ------------------------------------------------------------------
    # Step 4: Run RAGAS evaluation (LLM-as-judge).
    # ------------------------------------------------------------------
    print("Running RAGAS evaluation (LLM-as-judge)...")
    import warnings
    from datasets import Dataset
    from ragas import evaluate, RunConfig
    # Suppress deprecation warnings from ragas.metrics — the classic API still
    # works correctly. ragas.metrics.collections (the new API) requires an
    # OpenAI-specific llm_factory that does not support Gemini, so we stay on
    # ragas.metrics until RAGAS adds multi-provider support for collections.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
        from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

    judge_llm, judge_emb = _build_judge()
    print(f"Judge LLM        : {JUDGE_LLM_MODEL} ({JUDGE_LLM_PROVIDER})")
    print(f"Judge Embeddings : {JUDGE_EMB_MODEL} ({JUDGE_LLM_PROVIDER})\n")

    # RunConfig controls how RAGAS executes the 84 judge calls
    # (21 questions × 4 metrics). The default runs them all concurrently,
    # which overwhelms Gemini's rate limits and causes TimeoutError.
    # max_workers=2 serialises requests to stay within rate limits.
    # timeout=120 gives each call 2 minutes before marking it as failed.
    # max_retries=3 retries transient errors (rate limit, network blip).
    run_config = RunConfig(
        timeout=120,
        max_retries=3,
        max_wait=60,
        max_workers=2,
    )

    experiment_name = f"hr-ragas-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ds = Dataset.from_dict(data)
    result = evaluate(
        ds,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
        llm=judge_llm,
        embeddings=judge_emb,
        raise_exceptions=False,
        experiment_name=experiment_name,
        run_config=run_config,
    )
    print("Evaluation complete.\n")

    # ------------------------------------------------------------------
    # Step 5: Print report.
    # ------------------------------------------------------------------
    run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    agg = print_report(result, run_date, len(data["question"]))

    # ------------------------------------------------------------------
    # Step 6: Save JSON results.
    # ------------------------------------------------------------------
    import pandas as pd
    df = result.to_pandas()
    metric_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    per_question = []
    for idx, row in df.iterrows():
        item = TEST_DATASET[idx]
        per_question.append({
            "index":        idx + 1,
            "path":         "DB" if item.get("db_path") else "RAG",
            "topic":        item["topic"],
            "question":     item["question"],
            "employee_id":  item.get("employee_id"),
            "answer":       data["answer"][idx],
            "ground_truth": item["ground_truth"],
            **{
                k: (None if pd.isna(row[k]) else round(float(row[k]), 4))
                for k in metric_keys if k in df.columns
            },
        })

    output = {
        "run_date":         run_date,
        "answer_model":     ANSWER_MODEL_NAME,
        "judge_llm":        JUDGE_LLM_MODEL,
        "judge_embeddings": JUDGE_EMB_MODEL,
        "api_base_url":     API_BASE_URL,
        "chroma_dir":       CHROMA_DIR,
        "num_questions":    len(data["question"]),
        "rag_path_count":   sum(1 for x in TEST_DATASET if not x.get("db_path")),
        "db_path_count":    sum(1 for x in TEST_DATASET if x.get("db_path")),
        "aggregate_scores": {
            k: (None if pd.isna(v) else round(v, 4)) for k, v in agg.items()
        },
        "thresholds":       THRESHOLDS,
        "per_question":     per_question,
    }

    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = SCRIPT_DIR / f"ragas_results_{timestamp}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
