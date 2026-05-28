"""
Prompt Configuration Loading Module

Reads all prompt templates from prompts.yaml, supporting:
- Dynamic variable injection (via Python str.format_map)
- Integration with LangChain ChatPromptTemplate
- Hot reload without process restart
"""
import os
import threading
from typing import Dict, Optional

import yaml
from langchain_core.prompts import ChatPromptTemplate

from smart_router_agent_EN.config.config import BASE_DIR

DEFAULT_PROMPTS_PATH = os.path.join(BASE_DIR, "config", "prompts.yaml")


class PromptLoader:
    """
    Unified prompt loader. Reads templates from prompts.yaml with hot reload support.
    Thread-safe: uses RLock to protect the _templates dictionary.
    """

    def __init__(self, prompts_path: str = DEFAULT_PROMPTS_PATH):
        self._path = prompts_path
        self._lock = threading.RLock()
        self._templates: Dict[str, str] = {}
        self.reload()

    def reload(self):
        """Re-read prompts.yaml from disk"""
        with self._lock:
            with open(self._path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self._templates = {k: str(v) for k, v in data.items()}
        print(f"  [PromptLoader] Loaded {len(self._templates)} prompt templates")

    def get_raw(self, name: str) -> str:
        """Get raw template string"""
        with self._lock:
            if name not in self._templates:
                raise KeyError(f"Prompt '{name}' not found in {self._path}")
            return self._templates[name]

    def format(self, name: str, **kwargs) -> str:
        """Get template and inject variables, returning the final text"""
        raw = self.get_raw(name)
        return raw.format_map(_SafeFormatDict(kwargs))

    def as_chat_prompt(self, name: str) -> ChatPromptTemplate:
        """
        Convert template to LangChain ChatPromptTemplate.
        {variable} in YAML will be recognized as input_variables.
        """
        raw = self.get_raw(name)
        return ChatPromptTemplate.from_messages([("human", raw)])

    @property
    def template_names(self):
        with self._lock:
            return list(self._templates.keys())


class _SafeFormatDict(dict):
    """Returns the placeholder itself for missing keys during format_map, avoiding KeyError"""
    def __missing__(self, key):
        return "{" + key + "}"
