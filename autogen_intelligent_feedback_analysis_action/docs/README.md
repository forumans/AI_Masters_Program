# AutoGen Intelligent User Feedback Analysis and Action System

This project processes user feedback from app store reviews and support emails, classifies each item, performs specialized analysis for bugs and feature requests, generates structured tickets, and reviews output quality.

The current implementation is a hybrid AutoGen design:
- Specialist components use AutoGen `AssistantAgent` and `UserProxyAgent` when configuration is available.
- The main processing pipeline runs through local Python agent classes for reliability.
- AutoGen group chat is used for orchestration prompts and run summaries in `use_autogen=True` mode.
- If AutoGen dependencies or configuration are unavailable, the system falls back to direct processing and rule-based logic where needed.

## Project Structure

```text
autogen_intelligent_feedback_analysis_action/
├── config/
│   └── OAI_CONFIG_LIST.example
├── data/
│   ├── app_store_reviews.csv
│   ├── support_emails.csv
│   ├── expected_classifications.csv
│   ├── analyzed_feedback.csv
│   ├── generated_tickets.csv
│   └── quality_reviews.csv
├── docs/
│   ├── API_Reference.md
│   ├── AutoGen_Architecture.md
│   ├── Installation_Guide.md
│   └── README.md
├── src/
│   ├── agents/
│   │   ├── csv_reader_agent.py
│   │   ├── feedback_classifier_agent.py
│   │   ├── bug_analysis_agent.py
│   │   ├── feature_extractor_agent.py
│   │   ├── ticket_creator_agent.py
│   │   └── quality_critic_agent.py
│   ├── orchestration/
│   │   └── autogen_manager.py
│   ├── ui/
│   │   └── dashboard.py
│   ├── autogen_support.py
│   └── main.py
├── tests/
│   ├── test_agents.py
│   └── test_orchestration.py
├── requirements.txt
└── run_autogen_demo.py
```

## Implemented Functionality

- Read and validate feedback from `app_store_reviews.csv` and `support_emails.csv`
- Combine both sources into a common feedback dataset
- Classify feedback into `Bug`, `Feature Request`, `Praise`, `Complaint`, and `Spam`
- Analyze bug reports for severity, category, device info, reproduction steps, and error details
- Analyze feature requests for category, priority, impact, complexity, target users, and benefits
- Create structured tickets with IDs, titles, descriptions, priority, labels, and effort estimates
- Review ticket quality for completeness, clarity, relevance, accuracy, and actionability
- Save output files and a processing summary for each run

## Processing Modes

### AutoGen Mode

Run with `use_autogen=True`.

What happens:
- The system validates and loads data locally.
- A group chat session is used to coordinate the run and generate a structured summary.
- The actual feedback pipeline is executed through the local agent classes.
- Structured output files are written to the configured output directory.

### Direct Mode

Run with `use_autogen=False`.

What happens:
- The system executes the full pipeline sequentially with no group chat dependency.
- Specialist agents still attempt AutoGen-based single-task reasoning first.
- If AutoGen is unavailable or a response cannot be parsed, each specialist falls back to rule-based processing.

## Output Files

Depending on the data and run mode, the system writes:

- `classified_feedback.csv`
- `bug_analysis.csv`
- `feature_extraction.csv`
- `generated_tickets.csv`
- `quality_reviews.csv`
- `metrics.json`
- `processing_summary.json`

## Running the System

Run from the `autogen_intelligent_feedback_analysis_action` directory.

### Demo

```bash
python run_autogen_demo.py
```

### Main Application

```bash
python src/main.py
```

### Streamlit Dashboard

```bash
streamlit run src/ui/dashboard.py
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`
- `OPENAI_MODEL_NAME`
- `AUTOGEN_USE_DOCKER`
- `AUTOGEN_MAX_CONSECUTIVE_AUTO_REPLY`
- `AUTOGEN_TIMEOUT`

### AutoGen Config File

You can provide model configuration through:

- `config/OAI_CONFIG_LIST`
- `OAI_CONFIG_LIST` environment variable

If no config file is found, the system can also build a minimal config from `OPENAI_API_KEY` and `OPENAI_MODEL_NAME`.

## Testing

Run:

```bash
pytest tests -q
```

Current automated coverage includes:
- agent fallback behavior
- group chat wiring
- direct-mode processing
- AutoGen-mode orchestration summary handling
- chat JSON extraction

## Notes

- `CSVReaderAgent` is a regular Python data component, not a conversational AutoGen agent.
- The AutoGen implementation is intentionally hybrid. Group chat helps coordinate and summarize, while the local pipeline does the concrete data processing.
- When AutoGen is unavailable, the project still functions through direct execution and rule-based fallbacks.
