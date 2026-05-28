"""
Prompt 配置加载模块

从 prompts.yaml 读取所有 Prompt 模板，支持：
- 动态变量注入（通过 Python str.format_map）
- 与 LangChain ChatPromptTemplate 集成
- 热加载（reload）不重启进程
"""
import os
import threading
from typing import Dict, Optional

import yaml
from langchain_core.prompts import ChatPromptTemplate

from smart_router_agent.config.config import BASE_DIR

DEFAULT_PROMPTS_PATH = os.path.join(BASE_DIR, "config", "prompts.yaml")


class PromptLoader:
    """
    统一 Prompt 加载器。从 prompts.yaml 读取模板，支持热加载。
    线程安全：使用 RLock 保护 _templates 字典。
    """

    def __init__(self, prompts_path: str = DEFAULT_PROMPTS_PATH):
        self._path = prompts_path
        self._lock = threading.RLock()
        self._templates: Dict[str, str] = {}
        self.reload()

    def reload(self):
        """从磁盘重新读取 prompts.yaml"""
        with self._lock:
            with open(self._path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self._templates = {k: str(v) for k, v in data.items()}
        print(f"  [PromptLoader] 已加载 {len(self._templates)} 个 Prompt 模板")

    def get_raw(self, name: str) -> str:
        """获取原始模板字符串"""
        with self._lock:
            if name not in self._templates:
                raise KeyError(f"Prompt '{name}' not found in {self._path}")
            return self._templates[name]

    def format(self, name: str, **kwargs) -> str:
        """获取模板并注入变量，返回最终文本"""
        raw = self.get_raw(name)
        return raw.format_map(_SafeFormatDict(kwargs))

    def as_chat_prompt(self, name: str) -> ChatPromptTemplate:
        """
        将模板转为 LangChain ChatPromptTemplate。
        YAML 中的 {variable} 会被识别为 input_variables。
        """
        raw = self.get_raw(name)
        return ChatPromptTemplate.from_messages([("human", raw)])

    @property
    def template_names(self):
        with self._lock:
            return list(self._templates.keys())


class _SafeFormatDict(dict):
    """format_map 时对缺失 key 返回占位符本身，避免 KeyError"""
    def __missing__(self, key):
        return "{" + key + "}"
