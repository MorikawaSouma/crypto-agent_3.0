from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from ..llm_client import DeepSeekClient
from ..mcp import MCPMessage
from .memory import MemoryManager

class BaseAgent(ABC):
    def __init__(self, name: str, llm_client: DeepSeekClient):
        self.name = name
        self.llm = llm_client
        self.memory = MemoryManager(name)
        


        
    def reset_memory(self):
        """Clear agent's memory"""
        self.memory.clear_memory()

    def _clean_think_content(self, response: str) -> str:
        """
        Remove <think>...</think> content from reasoning models
        """
        import re
        # Remove <think>...</think> (dotall to match newlines)
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        return cleaned.strip()

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Helper to parse JSON response
        """
        import json
        import re
        
        cleaned = self._clean_think_content(response)
        
        # Extract JSON block if needed
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group(0)
            
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            print(f"Error parsing JSON for {self.name}: {cleaned[:100]}...")
            return {"action": "HOLD", "score": 50, "reasoning": "Parse Error"}

    @abstractmethod
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the given data and return a decision/score
        """
        pass
    
    def reflect(self, context: str, decision: str, outcome: str, reasoning: str):
        """
        Save a reflection on the outcome of a decision.
        """
        self.memory.add_memory(context, decision, outcome, reasoning)

    def receive_message(self, message: MCPMessage):
        """
        Handle incoming MCP messages
        """
        pass
