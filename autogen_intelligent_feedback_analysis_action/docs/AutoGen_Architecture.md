# AutoGen Architecture Documentation

## Overview

The system uses a hybrid architecture:
- local Python classes perform the end-to-end feedback analysis pipeline
- AutoGen agents provide LLM-backed specialist reasoning when available
- AutoGen group chat coordinates the run and produces summaries in AutoGen mode

This design keeps the pipeline runnable even when AutoGen configuration is missing or an LLM response is unusable.

## Core Architecture

### Local Processing Pipeline

The main execution path lives in [`src/orchestration/autogen_manager.py`](../src/orchestration/autogen_manager.py).

Shared pipeline stages:
1. Read feedback data
2. Validate dataset structure
3. Classify all feedback items
4. Run bug analysis for items classified as `Bug`
5. Run feature extraction for items classified as `Feature Request`
6. Merge analysis results
7. Create structured tickets
8. Review ticket quality
9. Save CSV and JSON outputs

### Specialist Components

#### CSVReaderAgent
- Reads CSV inputs
- Normalizes both sources into a common schema
- Validates required fields
- This is not a conversational AutoGen agent

#### FeedbackClassifierAgent
- Uses AutoGen classification when available
- Falls back to keyword-based classification on failure
- Produces category, confidence, and reasoning

#### BugAnalysisAgent
- Uses AutoGen analysis when available
- Falls back to rule-based severity/category extraction
- Produces severity, bug category, device info, reproduction steps, and error message

#### FeatureExtractorAgent
- Uses AutoGen analysis when available
- Falls back to rule-based feature extraction
- Produces feature category, priority, impact, complexity, target users, and benefits

#### TicketCreatorAgent
- Uses AutoGen ticket generation when available
- Falls back to rule-based ticket construction
- Produces structured ticket records with IDs, labels, assignee, and effort

#### QualityCriticAgent
- Uses AutoGen quality review when available
- Falls back to rule-based scoring
- Produces quality scores, issues, suggestions, and manual-review flags

## AutoGen Orchestration

### Group Chat Participants

The group chat defined in `AutoGenFeedbackAnalysisSystem` includes:
- `coordinator`
- `data_processor`
- `feedback_classifier`
- `bug_analyzer`
- `feature_extractor`
- `ticket_creator`
- `quality_reviewer`
- `user_proxy`

### What AutoGen Mode Does

When `process_feedback(use_autogen=True)` is used:
1. The system loads and validates the feedback data locally.
2. A group chat stage is used to coordinate the run at a planning level.
3. The shared Python pipeline performs classification, analysis, ticketing, and quality review.
4. A second group chat stage generates a structured summary of the run.
5. Results are saved along with `processing_summary.json`.

### What AutoGen Mode Does Not Do

The current implementation does not use group chat as the sole execution runtime for every pipeline step. The specialized local classes still perform the concrete data processing. This is a deliberate reliability tradeoff.

## Direct Mode

When `process_feedback(use_autogen=False)` is used:
- the shared pipeline runs sequentially
- no group chat is required
- each specialist still tries AutoGen single-task reasoning first if configured
- failures fall back to deterministic rule-based logic

## Fallback Strategy

Fallback order:
1. AutoGen group chat mode
2. Direct pipeline mode
3. Rule-based specialist logic

This happens at two levels:
- orchestration level: AutoGen mode falls back to direct mode
- specialist level: LLM-backed agents fall back to rules if configuration or parsing fails

## Configuration Loading

`src/autogen_support.py` provides shared AutoGen support:
- checks whether AutoGen imports are available
- loads config from `config/OAI_CONFIG_LIST`
- supports `OAI_CONFIG_LIST` from environment
- builds a minimal config from `OPENAI_API_KEY` and `OPENAI_MODEL_NAME` when possible

## Output Model

The orchestration layer returns a result dictionary with:
- `status`
- `mode`
- `processing_time`
- `total_processed`
- `successful`
- `failed`
- `classification_accuracy`
- `output_files`
- `chat_summary`
- `chat_history`
- `category_distribution`

## Saved Files

The pipeline can produce:
- `classified_feedback.csv`
- `bug_analysis.csv`
- `feature_extraction.csv`
- `generated_tickets.csv`
- `quality_reviews.csv`
- `metrics.json`
- `processing_summary.json`

## Testing Coverage

Current automated tests validate:
- agent fallback behavior
- ticket validation behavior
- direct-mode output generation
- group chat participant wiring
- AutoGen-mode structured summary handling
- JSON extraction from chat content

## Design Tradeoff

This project now favors a dependable hybrid pattern over a fully chat-driven runtime. That means:
- better reliability for batch processing
- cleaner fallback behavior
- easier automated testing
- less “pure” conversational execution than a fully autonomous AutoGen workflow
