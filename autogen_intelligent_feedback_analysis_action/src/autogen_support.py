"""
Utilities for working with AutoGen in environments where configuration or the
AutoGen package itself may be missing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

try:
    from autogen import (  # type: ignore
        AssistantAgent,
        GroupChat,
        GroupChatManager,
        UserProxyAgent,
        config_list_from_json as _autogen_config_list_from_json,
    )

    AUTOGEN_AVAILABLE = True
    AUTOGEN_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on local environment
    AssistantAgent = None
    GroupChat = None
    GroupChatManager = None
    UserProxyAgent = None
    _autogen_config_list_from_json = None
    AUTOGEN_AVAILABLE = False
    AUTOGEN_IMPORT_ERROR = exc


def resolve_config_path(env_or_file: str = "OAI_CONFIG_LIST", base_dir: Optional[str] = None) -> Optional[Path]:
    """
    Resolve an AutoGen config file path from common project locations.
    """
    candidate_strings = []

    env_value = os.getenv(env_or_file)
    if env_value and not env_value.strip().startswith("["):
        candidate_strings.append(env_value)

    search_roots = [Path.cwd()]
    if base_dir:
        search_roots.append(Path(base_dir).resolve())

    module_root = Path(__file__).resolve().parent
    search_roots.extend([module_root, module_root.parent])

    for root in search_roots:
        candidate_strings.extend(
            [
                str(root / env_or_file),
                str(root / "config" / env_or_file),
                str(root / "config" / f"{env_or_file}.json"),
            ]
        )

    seen = set()
    for candidate in candidate_strings:
        if candidate in seen:
            continue
        seen.add(candidate)
        path = Path(candidate)
        if path.exists() and path.is_file():
            return path

    return None


def _load_config_from_env_or_defaults() -> List[Dict[str, str]]:
    """
    Build a minimal config list from environment variables when no config file
    is available.
    """
    env_value = os.getenv("OAI_CONFIG_LIST")
    if env_value and env_value.strip().startswith("["):
        parsed = json.loads(env_value)
        if isinstance(parsed, list):
            return parsed

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return []

    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    return [{"model": model_name, "api_key": api_key}]


def load_config_list(env_or_file: str = "OAI_CONFIG_LIST", base_dir: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Load a usable AutoGen config list from file or environment.
    """
    fallback_config = _load_config_from_env_or_defaults()

    config_path = resolve_config_path(env_or_file=env_or_file, base_dir=base_dir)
    if config_path:
        with config_path.open("r", encoding="utf-8") as config_file:
            file_config = json.load(config_file)
        if isinstance(file_config, list):
            return file_config

    return fallback_config


def autogen_is_ready(config_list: Optional[List[Dict[str, str]]]) -> bool:
    """
    Return True when AutoGen is importable and configuration is available.
    """
    return AUTOGEN_AVAILABLE and bool(config_list)
