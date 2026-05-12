import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from agents.feedback_classifier_agent import FeedbackClassifierAgent
from agents.ticket_creator_agent import TicketCreatorAgent


def test_feedback_classifier_falls_back_to_rules_when_autogen_is_unavailable(tmp_path):
    agent = FeedbackClassifierAgent(model_dir=str(tmp_path / "models"))

    result = agent.classify_feedback("The app crashes every time I press sync.")

    assert result["category"] == "Bug"
    assert result["confidence"] > 0
    assert "reasoning" in result


def test_ticket_creator_enhances_invalid_ticket_payload():
    agent = TicketCreatorAgent(auto_approve=False)

    ticket = agent._validate_and_enhance_ticket(
        ticket_data={
            "title": "Broken output",
            "type": "Unknown Type",
            "priority": "Urgent",
            "status": "Queued",
        },
        feedback_data={"id": "REV-1", "source_type": "app_store_review"},
        analysis_data={"confidence": 0.88},
    )

    assert ticket["ticket_id"].startswith("TK-")
    assert ticket["type"] == "Investigation"
    assert ticket["priority"] == "Medium"
    assert ticket["status"] == "Open"
    assert ticket["feedback_id"] == "REV-1"
    assert ticket["confidence"] == 0.88
