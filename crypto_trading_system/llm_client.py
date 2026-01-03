import requests
import json
import random
import time
from .config import Config

class DeepSeekClient:
    def __init__(self, mock=False):
        self.api_key = Config.DEEPSEEK_API_KEY
        self.base_url = Config.DEEPSEEK_BASE_URL
        self.model = Config.MODEL_NAME
        self.mock = mock
        
    def query(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        """
        Send a query to the DeepSeek model using requests
        """
        if self.mock:
            return self._get_mock_response(system_prompt)

        # Construct endpoint
        endpoint = f"{self.base_url}/chat/completions"
        if "chat/completions" in self.base_url:
            endpoint = self.base_url

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": 4096,
            "stream": False
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Use local proxy as requested by user
        proxies = {
            "http": "http://127.0.0.1:7897",
            "https": "http://127.0.0.1:7897"
        }
        
        print(f"DEBUG: Querying {self.model} via requests...", flush=True)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    endpoint,
                    json=data,
                    headers=headers,
                    proxies=proxies,
                    timeout=120 # Standard timeout for chat models
                )
                
                response.raise_for_status()
                resp_json = response.json()
                
                if 'choices' not in resp_json or not resp_json['choices']:
                    print(f"DEBUG: No choices in response: {resp_json}", flush=True)
                    continue
                    
                content = resp_json['choices'][0]['message'].get('content', '')
                return content
                
            except Exception as e:
                print(f"DEBUG: Request failed attempt {attempt+1}: {e}", flush=True)
                time.sleep(2)
        
        return ""

    def _get_mock_response(self, system_prompt: str) -> str:
        """Generate mock JSON response based on agent type inferred from system prompt"""
        action = random.choice(["BUY", "SELL", "HOLD"])
        score = random.randint(40, 90)
        
        if "沃伦·巴菲特" in system_prompt:
            return json.dumps({
                "score": score,
                "reasoning": "Mocked Buffett Analysis: Value investing principles applied.",
                "action": action
            })
        elif "乔治·索罗斯" in system_prompt:
            return json.dumps({
                "score": score,
                "position_advice": random.uniform(0.1, 1.0),
                "reasoning": "Mocked Soros Analysis: Reflexivity observed.",
                "action": action
            })
        elif "雷·达里欧" in system_prompt:
            return json.dumps({
                "weight_suggestion": score,
                "reasoning": "Mocked Dalio Analysis: Risk parity adjustment.",
                "action": "REBALANCE" if action != "HOLD" else "HOLD"
            })
        elif "吉姆·西蒙斯" in system_prompt:
            return json.dumps({
                "score": score,
                "confidence": random.uniform(0.5, 0.9),
                "reasoning": "Mocked Simons Analysis: Statistical anomaly detected.",
                "action": action
            })
        elif "情绪分析专家" in system_prompt:
            return json.dumps({
                "score": score,
                "sentiment_stage": random.choice(["Accumulation", "Markup", "Distribution", "Panic"]),
                "reasoning": "Mocked Sentiment Analysis: Social volume high.",
                "action": action
            })
        else:
            return "{}"
