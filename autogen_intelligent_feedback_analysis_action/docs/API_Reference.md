# API Reference

## AutoGenFeedbackAnalysisSystem

Main orchestration class defined in [`src/orchestration/autogen_manager.py`](../src/orchestration/autogen_manager.py).

### Constructor

```python
AutoGenFeedbackAnalysisSystem(
    data_dir: str = "data",
    output_dir: str = "data",
    confidence_threshold: float = 0.7,
)
```

### `process_feedback`

```python
process_feedback(use_autogen: bool = True) -> Dict
```

Runs the feedback pipeline.

Behavior:
- `use_autogen=True`: uses group chat planning/summary plus the shared local pipeline
- `use_autogen=False`: runs the shared local pipeline directly

Typical return shape:

```python
{
    "status": "success",
    "mode": "autogen",
    "processing_time": 1.42,
    "total_processed": 25,
    "successful": 25,
    "failed": 0,
    "classification_accuracy": 0.84,
    "output_files": {
        "classified_feedback": "data/classified_feedback.csv",
        "generated_tickets": "data/generated_tickets.csv",
        "quality_reviews": "data/quality_reviews.csv",
        "metrics": "data/metrics.json",
    },
    "chat_summary": "Processed 25 feedback items...",
    "chat_history": [...],
    "category_distribution": {
        "Bug": 10,
        "Feature Request": 8,
        "Complaint": 4,
        "Praise": 3,
    },
}
```

### `get_system_status`

```python
get_system_status() -> Dict
```

Returns basic system metadata:

```python
{
    "system_type": "AutoGen",
    "agents_initialized": True,
    "group_chat_active": True,
    "data_directory": "data",
    "output_directory": "data",
    "confidence_threshold": 0.7,
    "timestamp": "2026-05-12T18:00:00",
}
```

## Agent Classes

## CSVReaderAgent

File: [`src/agents/csv_reader_agent.py`](../src/agents/csv_reader_agent.py)

Primary methods:

```python
read_app_store_reviews() -> pd.DataFrame
read_support_emails() -> pd.DataFrame
combine_feedback_data() -> pd.DataFrame
validate_data(df: pd.DataFrame) -> bool
get_data_summary(df: pd.DataFrame) -> Dict
```

Notes:
- normalizes review and support-email inputs into a shared schema
- this component is local Python logic, not a conversational AutoGen agent

## FeedbackClassifierAgent

File: [`src/agents/feedback_classifier_agent.py`](../src/agents/feedback_classifier_agent.py)

Primary methods:

```python
classify_feedback(feedback_text: str) -> Dict
classify_batch(feedback_df: pd.DataFrame) -> pd.DataFrame
evaluate_classification(classified_df: pd.DataFrame, expected_df: pd.DataFrame) -> Dict
```

Single-item return shape:

```python
{
    "category": "Bug",
    "confidence": 0.85,
    "reasoning": "User reports app crashing",
}
```

## BugAnalysisAgent

File: [`src/agents/bug_analysis_agent.py`](../src/agents/bug_analysis_agent.py)

Primary methods:

```python
analyze_bug_report(feedback_text: str, feedback_id: str = "") -> Dict
analyze_batch(bug_df: pd.DataFrame) -> pd.DataFrame
get_severity_distribution(analyzed_df: pd.DataFrame) -> Dict
```

Single-item return shape:

```python
{
    "feedback_id": "REV001",
    "severity": "High",
    "category": "crash",
    "device_info": "Android 12",
    "reproduction_steps": ["Open app", "Start sync"],
    "error_message": "App crashes when syncing",
    "confidence": 0.82,
    "reasoning": "Rule-based or AutoGen-based analysis",
}
```

## FeatureExtractorAgent

File: [`src/agents/feature_extractor_agent.py`](../src/agents/feature_extractor_agent.py)

Primary methods:

```python
extract_feature_info(feedback_text: str, feedback_id: str = "") -> Dict
extract_batch(feature_df: pd.DataFrame) -> pd.DataFrame
get_feature_statistics(extracted_df: pd.DataFrame) -> Dict
```

Single-item return shape:

```python
{
    "feedback_id": "EMAIL001",
    "category": "Functionality",
    "priority": "High",
    "impact_score": 0.8,
    "complexity": "Medium",
    "target_users": "General users",
    "benefits": ["Improve workflow"],
    "confidence": 0.8,
    "reasoning": "Rule-based or AutoGen-based extraction",
}
```

## TicketCreatorAgent

File: [`src/agents/ticket_creator_agent.py`](../src/agents/ticket_creator_agent.py)

Primary methods:

```python
create_ticket(feedback_data: Dict, analysis_data: Dict | None = None) -> Dict
create_batch_tickets(feedback_df: pd.DataFrame, analysis_df: pd.DataFrame | None = None) -> pd.DataFrame
save_tickets(tickets_df: pd.DataFrame, output_path: str) -> bool
```

Single-item return shape:

```python
{
    "ticket_id": "TK-20260512-ABCD1234",
    "title": "[Bug] App crashes during sync",
    "description": "...",
    "type": "Bug",
    "priority": "High",
    "status": "Open",
    "assignee": "Developer",
    "labels": ["bug", "user-feedback"],
    "estimated_effort": "5",
    "reproduction_steps": "1. Open app",
    "expected_outcome": "",
    "feedback_id": "REV001",
    "source_type": "app_store_review",
    "created_at": "2026-05-12T18:00:00",
    "updated_at": "2026-05-12T18:00:00",
    "confidence": 0.8,
}
```

## QualityCriticAgent

File: [`src/agents/quality_critic_agent.py`](../src/agents/quality_critic_agent.py)

Primary methods:

```python
review_ticket_quality(ticket_data: Dict) -> Dict
review_batch_tickets(tickets_df: pd.DataFrame) -> pd.DataFrame
get_quality_metrics(reviews_df: pd.DataFrame) -> Dict
save_quality_reviews(reviews_df: pd.DataFrame, output_path: str) -> bool
```

Single-item return shape:

```python
{
    "ticket_id": "TK-20260512-ABCD1234",
    "overall_score": 0.84,
    "quality_level": "Good",
    "completeness_score": 0.9,
    "accuracy_score": 0.8,
    "clarity_score": 0.8,
    "relevance_score": 0.9,
    "actionability_score": 0.7,
    "issues": [],
    "suggestions": [],
    "needs_manual_review": False,
    "reasoning": "Rule-based or AutoGen-based review",
}
```

## AutoGen Support

File: [`src/autogen_support.py`](../src/autogen_support.py)

Shared helpers:

```python
resolve_config_path(env_or_file: str = "OAI_CONFIG_LIST", base_dir: str | None = None) -> Path | None
load_config_list(env_or_file: str = "OAI_CONFIG_LIST", base_dir: str | None = None) -> List[Dict[str, str]]
autogen_is_ready(config_list: list | None) -> bool
```

Purpose:
- centralize AutoGen import handling
- locate config files
- build minimal model configuration from environment variables

## Testing Entry Point

Run:

```bash
pytest tests -q
```

The current tests cover the implemented orchestration and fallback behavior rather than live OpenAI responses.
