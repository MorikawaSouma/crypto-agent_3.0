from abc import ABC, abstractmethod
from typing import Dict, Any
from ..llm_client import DeepSeekClient
from ..mcp import MCPMessage

class BaseAgent(ABC):
    def __init__(self, name: str, llm_client: DeepSeekClient):
        self.name = name
        self.llm = llm_client
        
    def _clean_think_content(self, response: str) -> str:
        """
        Remove <think>...</think> content from reasoning models
        """
        import re
        # Remove <think>...</think> (dotall to match newlines)
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        return cleaned.strip()

    @abstractmethod
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the given data and return a decision/score
        """
        pass
    
    def receive_message(self, message: MCPMessage):
        """
        Handle incoming MCP messages
        """
        pass
