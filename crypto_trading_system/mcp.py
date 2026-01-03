from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import datetime
import json

@dataclass
class MCPMessage:
    message_type: str
    sender: str
    receiver: str
    payload: Dict[str, Any]
    priority: str = "normal"
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.datetime.utcnow().isoformat() + "Z"
            
    def to_json(self):
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls(**data)
