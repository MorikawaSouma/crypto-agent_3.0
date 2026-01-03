from .base_agent import BaseAgent
from typing import Dict, Any
import json

class WarrenBuffettAgent(BaseAgent):
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = """
        你是沃伦·巴菲特风格的价值投资分析师。
        关注：内在价值vs市场价格、安全边际、长期增长潜力、护城河强度。
        决策风格：保守、注重安全边际、长期主义。
        """
        
        user_prompt = f"""
        请基于以下数据：
        
        项目基本面：{data.get('fundamentals', 'N/A')}
        市场数据：{data.get('market_data_summary', 'N/A')}
        估值指标：{data.get('valuation', 'N/A')}
        
        从价值投资角度评估该资产。
        
        请严格以JSON格式输出，不要包含Markdown格式：
        {{
            "score": 0-100的整数,
            "reasoning": "详细理由",
            "action": "BUY/SELL/HOLD"
        }}
        """
        
        response = self.llm.query(system_prompt, user_prompt)
        # print("DEBUG: Mocking LLM response for Buffett")
        # response = '{"score": 80, "reasoning": "Mocked reasoning", "action": "BUY"}'
        return self._parse_response(response)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        try:
            # Clean think tags first
            response = self._clean_think_content(response)
            # Clean up potential markdown code blocks
            clean_response = response.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_response)
        except Exception as e:
            print(f"[{self.name}] Error parsing JSON: {e}")
            return {"score": 50, "reasoning": "Error parsing LLM response", "action": "HOLD"}

class GeorgeSorosAgent(BaseAgent):
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = """
        你是乔治·索罗斯风格的宏观对冲分析师。
        关注：市场反身性理论、宏观经济趋势、市场情绪极端点、流动性分析。
        决策风格：激进、擅长抓住大趋势、逆向思维。
        """
        
        user_prompt = f"""
        请分析：
        
        市场趋势：{data.get('trend_data', 'N/A')}
        宏观环境：{data.get('macro_data', 'N/A')}
        情绪指标：{data.get('sentiment_data', 'N/A')}
        流动性：{data.get('liquidity_data', 'N/A')}
        
        运用反身性理论，识别当前市场主导趋势和潜在转折点。
        
        请严格以JSON格式输出，不要包含Markdown格式：
        {{
            "score": 0-100的整数,
            "position_advice": "0.0-1.0 (仓位比例建议)",
            "reasoning": "详细理由",
            "action": "BUY/SELL/HOLD"
        }}
        """
        
        response = self.llm.query(system_prompt, user_prompt)
        return self._parse_response(response)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        try:
            response = self._clean_think_content(response)
            clean_response = response.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_response)
        except Exception as e:
            print(f"[{self.name}] Error parsing JSON: {e}")
            return {"score": 50, "position_advice": 0.0, "reasoning": "Error", "action": "HOLD"}

class RayDalioAgent(BaseAgent):
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = """
        你是雷·达里欧风格的全天候策略分析师。
        关注：经济周期位置、风险分散度、相关性矩阵、尾部风险。
        决策风格：平衡、风险分散、系统化。
        """
        
        user_prompt = f"""
        请评估：
        
        当前经济周期：{data.get('economic_cycle', 'N/A')}
        资产相关性：{data.get('correlation_matrix', 'N/A')}
        风险指标：{data.get('risk_metrics', 'N/A')}
        组合配置：{data.get('portfolio_allocation', 'N/A')}
        
        从风险平价角度分析。
        
        请严格以JSON格式输出，不要包含Markdown格式：
        {{
            "weight_suggestion": 0-100的整数,
            "reasoning": "详细理由",
            "action": "REBALANCE/HOLD"
        }}
        """
        
        response = self.llm.query(system_prompt, user_prompt)
        return self._parse_response(response)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        try:
            response = self._clean_think_content(response)
            clean_response = response.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_response)
        except Exception as e:
            print(f"[{self.name}] Error parsing JSON: {e}")
            return {"weight_suggestion": 0, "reasoning": "Error", "action": "HOLD"}

class JimSimonsAgent(BaseAgent):
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = """
        你是吉姆·西蒙斯风格的量化交易大师。
        关注：统计显著性、因子有效性、均值回归概率、异常模式识别。
        决策风格：完全数据驱动、冷酷、数学模型导向，不关心基本面故事。
        """
        
        user_prompt = f"""
        请分析以下量化特征：
        
        统计特征：{data.get('statistical_features', 'N/A')}
        技术因子：{data.get('technical_factors', 'N/A')}
        波动率模型：{data.get('volatility_model', 'N/A')}
        
        请从纯数学和统计套利角度评估。
        
        请严格以JSON格式输出，不要包含Markdown格式：
        {{
            "score": 0-100的整数,
            "confidence": 0.0-1.0 (模型置信度),
            "reasoning": "详细理由(基于统计学术语)",
            "action": "BUY/SELL/HOLD"
        }}
        """
        
        response = self.llm.query(system_prompt, user_prompt)
        return self._parse_response(response)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        try:
            response = self._clean_think_content(response)
            clean_response = response.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_response)
        except Exception as e:
            print(f"[{self.name}] Error parsing JSON: {e}")
            return {"score": 50, "confidence": 0.0, "reasoning": "Error", "action": "HOLD"}

class CryptoSentimentAgent(BaseAgent):
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = """
        你是加密货币市场的情绪分析专家，擅长捕捉FOMO（错失恐惧）和FUD（恐惧、不确定和怀疑）。
        关注：社交媒体热度、恐慌贪婪指数、大户异动、Reddit/Twitter舆情。
        决策风格：反向指标（极端贪婪时卖出）或趋势跟随（情绪爆发初期）。
        """
        
        user_prompt = f"""
        请分析当前市场情绪：
        
        社交媒体数据：{data.get('social_metrics', 'N/A')}
        恐慌贪婪指数：{data.get('fear_greed', 'N/A')}
        新闻情绪：{data.get('news_sentiment', 'N/A')}
        链上大户行为：{data.get('whale_activity', 'N/A')}
        
        请判断当前市场情绪阶段（吸筹/拉升/派发/恐慌）。
        
        请严格以JSON格式输出，不要包含Markdown格式：
        {{
            "score": 0-100的整数 (分数越高越看涨),
            "sentiment_stage": "阶段描述",
            "reasoning": "详细理由",
            "action": "BUY/SELL/HOLD"
        }}
        """
        
        response = self.llm.query(system_prompt, user_prompt)
        return self._parse_response(response)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        try:
            response = self._clean_think_content(response)
            clean_response = response.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_response)
        except Exception as e:
            print(f"[{self.name}] Error parsing JSON: {e}")
            return {"score": 50, "sentiment_stage": "Unknown", "reasoning": "Error", "action": "HOLD"}
