import json
import os
import sys
from types import SimpleNamespace

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import orchestration.autogen_manager as autogen_manager_module
from orchestration.autogen_manager import AutoGenFeedbackAnalysisSystem


def _write_sample_feedback_files(base_dir):
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(
        [
            {
                "review_id": "REV001",
                "platform": "Google Play",
                "rating": 1,
                "review_text": "The app crashes when I try to sync my account.",
                "user_name": "user1",
                "date": "2026-05-01",
                "app_version": "1.0.0",
            }
        ]
    ).to_csv(os.path.join(data_dir, "app_store_reviews.csv"), index=False)

    pd.DataFrame(
        [
            {
                "email_id": "EMAIL001",
                "subject": "Feature request",
                "body": "Please add dark mode and scheduled exports for teams.",
                "sender_email": "user@example.com",
                "timestamp": "2026-05-02T10:00:00",
                "priority": "high",
            }
        ]
    ).to_csv(os.path.join(data_dir, "support_emails.csv"), index=False)

    return data_dir, output_dir


class DummyAssistantAgent:
    def __init__(self, name, llm_config=None, system_message=""):
        self.name = name
        self.llm_config = llm_config or {}
        self.system_message = system_message


class DummyUserProxyAgent:
    def __init__(self, name, human_input_mode="NEVER", max_consecutive_auto_reply=1, code_execution_config=False):
        self.name = name
        self.human_input_mode = human_input_mode
        self.max_consecutive_auto_reply = max_consecutive_auto_reply
        self.code_execution_config = code_execution_config

    def initiate_chat(self, target, message, max_turns=2):
        payload = {
            "summary": "AutoGen summary from test",
            "highlights": ["Coordinated classification and ticketing"],
            "risks": [],
        }
        return SimpleNamespace(
            chat_history=[
                {"role": "user", "content": message},
                {"role": "assistant", "content": json.dumps(payload)},
            ]
        )


class DummyGroupChat:
    def __init__(self, agents, messages=None, max_round=30):
        self.agents = agents
        self.messages = messages or []
        self.max_round = max_round


class DummyGroupChatManager:
    def __init__(self, groupchat, llm_config=None):
        self.groupchat = groupchat
        self.llm_config = llm_config or {}


def test_process_direct_generates_expected_outputs(tmp_path):
    data_dir, output_dir = _write_sample_feedback_files(tmp_path)
    system = AutoGenFeedbackAnalysisSystem(data_dir=data_dir, output_dir=output_dir)

    result = system.process_feedback(use_autogen=False)

    assert result["status"] == "success"
    assert result["mode"] == "direct"
    assert result["total_processed"] == 2
    assert os.path.exists(result["output_files"]["classified_feedback"])
    assert os.path.exists(result["output_files"]["generated_tickets"])
    assert os.path.exists(result["output_files"]["quality_reviews"])
    assert os.path.exists(os.path.join(output_dir, "processing_summary.json"))


def test_group_chat_setup_includes_feedback_classifier(tmp_path, monkeypatch):
    data_dir, output_dir = _write_sample_feedback_files(tmp_path)

    monkeypatch.setattr(autogen_manager_module, "AssistantAgent", DummyAssistantAgent)
    monkeypatch.setattr(autogen_manager_module, "UserProxyAgent", DummyUserProxyAgent)
    monkeypatch.setattr(autogen_manager_module, "GroupChat", DummyGroupChat)
    monkeypatch.setattr(autogen_manager_module, "GroupChatManager", DummyGroupChatManager)
    monkeypatch.setattr(
        autogen_manager_module,
        "load_config_list",
        lambda base_dir=None: [{"model": "gpt-4o-mini", "api_key": "test-key"}],
    )
    monkeypatch.setattr(autogen_manager_module, "autogen_is_ready", lambda config_list: True)

    system = AutoGenFeedbackAnalysisSystem(data_dir=data_dir, output_dir=output_dir)
    participant_names = [agent.name for agent in system.group_chat.agents]

    assert "feedback_classifier" in participant_names


def test_process_with_autogen_uses_structured_summary(tmp_path, monkeypatch):
    data_dir, output_dir = _write_sample_feedback_files(tmp_path)

    monkeypatch.setattr(autogen_manager_module, "AssistantAgent", DummyAssistantAgent)
    monkeypatch.setattr(autogen_manager_module, "UserProxyAgent", DummyUserProxyAgent)
    monkeypatch.setattr(autogen_manager_module, "GroupChat", DummyGroupChat)
    monkeypatch.setattr(autogen_manager_module, "GroupChatManager", DummyGroupChatManager)
    monkeypatch.setattr(
        autogen_manager_module,
        "load_config_list",
        lambda base_dir=None: [{"model": "gpt-4o-mini", "api_key": "test-key"}],
    )
    monkeypatch.setattr(autogen_manager_module, "autogen_is_ready", lambda config_list: True)

    system = AutoGenFeedbackAnalysisSystem(data_dir=data_dir, output_dir=output_dir)
    result = system.process_feedback(use_autogen=True)

    assert result["status"] == "success"
    assert result["mode"] == "autogen"
    assert result["chat_summary"] == "AutoGen summary from test"
    assert len(result["chat_history"]) >= 2
    assert os.path.exists(result["output_files"]["metrics"])


def test_extract_results_from_chat_parses_json_code_block(tmp_path):
    data_dir, output_dir = _write_sample_feedback_files(tmp_path)
    system = AutoGenFeedbackAnalysisSystem(data_dir=data_dir, output_dir=output_dir)

    extracted = system._extract_results_from_chat(
        [
            {
                "role": "assistant",
                "content": '```json\n{"summary": "Structured response"}\n```',
            }
        ]
    )

    assert extracted["summary"] == "Structured response"
    assert extracted["structured_payloads"][0]["summary"] == "Structured response"
