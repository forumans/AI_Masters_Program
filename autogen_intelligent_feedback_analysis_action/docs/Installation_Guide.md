# Installation Guide

## Prerequisites

- Python 3.10 or higher
- Internet access if you want to use OpenAI-backed AutoGen behavior
- An OpenAI API key for LLM-backed execution

The project can still run in direct mode without working AutoGen configuration, but LLM-backed agent behavior will fall back to rule-based logic.

## 1. Create and Activate a Virtual Environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

The current `requirements.txt` already targets newer compatible package versions, including:
- `pyautogen`
- `autogen-agentchat`
- `openai`
- `pandas`
- `streamlit`

## 3. Configure Environment Variables

Create `.env` from the example if your project includes one, or define the variables manually.

Recommended variables:

```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_NAME=gpt-4o-mini
AUTOGEN_USE_DOCKER=false
AUTOGEN_MAX_CONSECUTIVE_AUTO_REPLY=10
AUTOGEN_TIMEOUT=120
```

## 4. Configure AutoGen Model Access

Preferred approach:

```bash
copy config\OAI_CONFIG_LIST.example config\OAI_CONFIG_LIST
```

Then edit `config/OAI_CONFIG_LIST`:

```json
[
  {
    "model": "gpt-4o-mini",
    "api_key": "your-openai-api-key-here"
  }
]
```

Notes:
- The system looks for `config/OAI_CONFIG_LIST` automatically.
- You can also provide `OAI_CONFIG_LIST` through an environment variable.
- If no config file is found, the code can build a minimal config from `OPENAI_API_KEY` and `OPENAI_MODEL_NAME`.

## 5. Verify Input Data

Make sure these files exist under `data/`:

```text
data/
├── app_store_reviews.csv
└── support_emails.csv
```

Example review row:

```csv
review_id,platform,rating,review_text,user_name,date,app_version
REV001,Google Play,1,"App crashes when I try to sync",john_doe89,2026-05-01,3.2.1
```

Example support email row:

```csv
email_id,subject,body,sender_email,timestamp,priority
EMAIL001,App Crash Report,"Hi Support Team, the app crashes during sync",user@email.com,2026-05-01T10:00:00,high
```

## 6. Run the Tests

```bash
pytest tests -q
```

This verifies the current implementation for:
- direct mode
- AutoGen-mode summary orchestration
- agent fallbacks
- chat payload parsing

## 7. Run the Application

### Main CLI

```bash
python src/main.py
```

### Demo Script

```bash
python run_autogen_demo.py
```

### Streamlit Dashboard

```bash
streamlit run src/ui/dashboard.py
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'autogen'`

Install dependencies again:

```bash
pip install -r requirements.txt
```

### AutoGen config is missing

Symptoms:
- group chat is inactive
- specialist agents fall back to rule-based logic

Fix:
- create `config/OAI_CONFIG_LIST`
- or set `OPENAI_API_KEY` and `OPENAI_MODEL_NAME`

### `FileNotFoundError` for input CSVs

Ensure the following exist:
- `data/app_store_reviews.csv`
- `data/support_emails.csv`

### AutoGen mode does not appear fully conversational

That is expected in the current architecture. AutoGen mode is hybrid:
- group chat coordinates and summarizes
- the concrete batch pipeline runs through local Python classes

## Verification Checklist

You are set up correctly when:
- `pytest tests -q` passes
- `python src/main.py` completes successfully
- output files appear in the configured output directory
- `processing_summary.json` is generated after a run
