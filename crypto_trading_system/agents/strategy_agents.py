from .base_agent import BaseAgent
from typing import Dict, Any
import json
from ..config import Config

class WarrenBuffettAgent(BaseAgent):
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Memory Retrieval
        context_str = f"Fundamentals: {data.get('fundamentals')}, Market: {data.get('market_data_summary')}"
        relevant_memories = self.memory.retrieve_relevant(context_str, limit=getattr(Config, "MEMORY_RETRIEVE_LIMIT", 10))
        memory_text = self.memory.format_memories_for_prompt(relevant_memories)

        fundamentals = data.get("fundamentals") or {}
        market = data.get("market_data_summary") or {}
        valuation = data.get("valuation") or {}
        onchain = fundamentals.get("onchain")
        defi_tvl = fundamentals.get("defi_tvl")
        risk_free = valuation.get("risk_free_rate")

        debate_context = data.get('debate_context', '')
        debate_instruction = ""
        if debate_context:
            debate_instruction = f"\n【多智能体辩论上下文 (其他人的观点)】\n{debate_context}\n请参考以上观点，如果觉得有道理可以修正自己的判断，或者进行反驳。\n"

        system_prompt = """
        你是沃伦·巴菲特风格的价值投资分析师。
        核心哲学：“价格是你支付的，价值是你得到的。”
        关注点：
        1. 内在价值与安全边际：寻找价格大幅低于价值的机会（如高回撤时的低吸）。
        2. 护城河：虽然Crypto难以定义传统护城河，但关注网络效应（活跃地址）、品牌（市值主导地位）。
        3. 长期主义：不关注短期波动，除非它提供了极佳的买入点。
        4. 机会成本：始终将预期回报与无风险利率（美国10年期国债收益率）进行比较。
        
        决策风格：极度保守，只有在胜率极高且安全边际充足时才出手。宁可错过，不可做错。
        """
        
        user_prompt = f"""
        请基于以下数据进行价值评估：
        
        【数据可用性】
        - 基本面数据: {"可用" if fundamentals else "缺失"}
        - 链上数据(活跃地址等): {"可用" if onchain not in (None, "N/A", {}) else "缺失"}
        - DeFi TVL: {"可用" if defi_tvl not in (None, "N/A") else "缺失"}
        - 市场数据: {"可用" if market else "缺失"}
        - 估值数据: {"可用" if valuation else "缺失"}
        - 宏观数据(无风险利率): {"可用" if risk_free not in (None, "N/A") else "缺失"}
        
        【项目基本面】
        {fundamentals if fundamentals else "N/A"}
        
        【市场状况】
        - 价格与均线：{market if market else "N/A"}
        - 估值指标：{valuation if valuation else "N/A"}
        
        【宏观参照】
        - 无风险利率 (US 10Y): {risk_free if risk_free not in (None, "N/A") else "缺失"}%
        
        【历史反思与记忆】
        {memory_text}
        {debate_instruction}
        
        思考逻辑：
        1. 若上述任何关键字段标记为“缺失”，请在推理中明确说明，不要虚构具体数值或历史走势。
        2. 当前价格是否具有足够的安全边际（参考drawdown_30d和MVRV，如缺失请说明）。
        3. 相比于无风险利率，持有该资产的风险溢价是否足够吸引人？
        4. 链上数据（活跃地址）是否支持其长期价值增长？
        
        请严格以JSON格式输出，不要包含Markdown格式：
        {{
            "score": 0-100的整数 (50为中性，<50不看好，>80强烈买入),
            "reasoning": "详细理由 (请明确引用无风险利率和安全边际的概念，数据缺失处请显式说明而不是猜测)",
            "action": "BUY/SELL/HOLD"
        }}
        """
        
        response = self.llm.query(system_prompt, user_prompt, temperature=0.3)
        return self._parse_response(response)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        try:
            response = self._clean_think_content(response)
            clean_response = response.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_response)
        except Exception as e:
            print(f"[{self.name}] Error parsing JSON: {e}")
            return {"score": 50, "reasoning": "Error parsing LLM response", "action": "HOLD"}

class GeorgeSorosAgent(BaseAgent):
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Memory Retrieval
        context_str = f"Trend: {data.get('trend_data')}, Macro: {data.get('macro_data')}"
        relevant_memories = self.memory.retrieve_relevant(context_str, limit=getattr(Config, "MEMORY_RETRIEVE_LIMIT", 7))
        memory_text = self.memory.format_memories_for_prompt(relevant_memories)

        trend = data.get("trend_data") or {}
        macro = data.get("macro_data") or {}
        sentiment = data.get("sentiment_data")
        liquidity = data.get("liquidity_data") or {}
        whale = data.get("whale_activity") or {}
        dxy = macro.get("dxy")
        us10y = macro.get("us_10y_yield")
        vix = macro.get("vix")

        debate_context = data.get('debate_context', '')
        debate_instruction = ""
        if debate_context:
            debate_instruction = f"\n【多智能体辩论上下文 (其他人的观点)】\n{debate_context}\n请参考以上观点，如果觉得有道理可以修正自己的判断，或者进行反驳。\n"

        system_prompt = """
        你是乔治·索罗斯风格的宏观对冲分析师。
        核心哲学：“市场总是错的。”（反身性理论）
        关注点：
        1. 趋势与动量：利用技术指标（MACD, ADX, Ichimoku）确认趋势强度。
        2. 宏观共振：美元指数(DXY)和美债收益率(Yields)如何影响全球流动性？
        3. 反身性循环：价格上涨是否引发了更多的买入（正反馈）？或者市场是否处于不可持续的泡沫/恐慌中？
        4. 证伪：时刻寻找证明当前趋势结束的信号（背离、关键支撑/阻力破坏）。
        
        决策风格：顺势而为，但在转折点果断反向。高风险高回报。
        """
        
        user_prompt = f"""
        请运用反身性理论分析以下市场：
        
        【数据可用性】
        - 趋势信号: {"可用" if trend else "缺失"}
        - 宏观数据整体: {"可用" if macro else "缺失"}
        - DXY: {"有值" if dxy not in (None, "N/A") else "缺失"}
        - US 10Y: {"有值" if us10y not in (None, "N/A") else "缺失"}
        - VIX: {"有值" if vix not in (None, "N/A") else "缺失"}
        - 情绪数据(F&G等): {"可用" if sentiment not in (None, "N/A") else "缺失"}
        - 资金流指标: {"可用" if liquidity else "缺失"}
        - 链上大户/活跃度: {"可用" if whale else "缺失"}
        
        【趋势信号】
        {trend if trend else "N/A"}
        
        【宏观背景 (流动性闸门)】
        {macro if macro else "N/A"}
        
        【情绪与流动性】
        - 情绪：{sentiment if sentiment not in (None, "N/A") else "N/A"}
        - 资金流：{liquidity if liquidity else "N/A"}
        - 链上：{whale if whale else "N/A"}
        
        【历史反思与记忆】
        {memory_text}
        {debate_instruction}
        
        思考逻辑：
        1. 若关键宏观字段或链上字段为“缺失”，请先说明数据缺口，不要假设具体数值。
        2. 宏观环境（DXY, Yields, VIX）是顺风还是逆风？
        3. 趋势强度（ADX, Ichimoku）如何？是否存在加强的自我强化过程？
        4. 是否观察到趋势衰竭或背离的信号？
        
        请严格以JSON格式输出，不要包含Markdown格式：
        {{
            "score": 0-100的整数 (50中性),
            "position_advice": "0.0-1.0 (根据确信度和趋势强度建议仓位)",
            "reasoning": "详细理由 (重点分析宏观与趋势的共振，并在数据缺失处显式说明)",
            "action": "BUY/SELL/HOLD"
        }}
        """
        
        response = self.llm.query(system_prompt, user_prompt, temperature=0.9)
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
        # Memory Retrieval
        context_str = f"Cycle: {data.get('economic_cycle')}, Risk: {data.get('risk_metrics')}"
        relevant_memories = self.memory.retrieve_relevant(context_str, limit=getattr(Config, "MEMORY_RETRIEVE_LIMIT", 7))
        memory_text = self.memory.format_memories_for_prompt(relevant_memories)

        economic = data.get("economic_cycle") or {}
        risk = data.get("risk_metrics") or {}
        corr = data.get("correlation_matrix")
        dxy = economic.get("dxy")
        yields = economic.get("yields")
        vix = economic.get("vix")

        system_prompt = """
        你是雷·达里欧风格的全天候策略分析师。
        核心哲学：通过极度分散化穿越所有经济环境。
        关注点：
        1. 经济机器：增长与通胀的预期变化。Crypto通常对流动性紧缩敏感（关注Yields和VIX）。
        2. 波动率调整：基于波动率（Volatility, ATR）调整仓位，高波动的资产配比应降低。
        3. 相关性：避免在资产高度相关（市场恐慌）时重仓。
        4. 尾部风险：始终关注Drawdown和VIX，防范极端行情。
        
        决策风格：稳健、系统化、风险平价。不追求最高收益，追求最高的夏普比率。
        """
        
        user_prompt = f"""
        请进行全天候风险配置分析：
        
        【数据可用性】
        - 宏观环境整体: {"可用" if economic else "缺失"}
        - DXY: {"有值" if dxy not in (None, "N/A") else "缺失"}
        - Yields(美债收益率): {"有值" if yields not in (None, "N/A") else "缺失"}
        - VIX: {"有值" if vix not in (None, "N/A") else "缺失"}
        - 风险度量(波动率/回撤): {"可用" if risk else "缺失"}
        - 市场相关性矩阵: {"可用" if corr not in (None, "N/A") else "缺失"}
        
        【经济环境】
        {economic if economic else "N/A"}
        
        【风险度量】
        - 波动率与回撤：{risk if risk else "N/A"}
        - 市场相关性：{corr if corr not in (None, "N/A") else "N/A"}
        
        【当前状态】
        {data.get('portfolio_allocation', 'N/A')}
        
        【历史反思与记忆】
        {memory_text}
        
        思考逻辑：
        1. 若关键宏观字段缺失，请在回答中说明无法精确判断其影响，不要假设具体数值。
        2. 当前宏观环境（VIX, Yields）是否适合风险资产（Crypto）？
        3. 资产本身的波动率（ATR, Volatility）是否过高，需要降低仓位以维持风险平价？
        4. 建议的配置权重是多少，以匹配目标风险水平？
        
        请严格以JSON格式输出，不要包含Markdown格式：
        {{
            "weight_suggestion": 0-100的整数 (建议持仓百分比),
            "reasoning": "详细理由 (基于风险平价和宏观环境，并对数据缺失情况作出说明)",
            "action": "REBALANCE/HOLD"
        }}
        """
        
        response = self.llm.query(system_prompt, user_prompt, temperature=0.5)
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
        # Memory Retrieval
        context_str = f"Stats: {data.get('statistical_features')}, Tech: {data.get('technical_factors')}"
        relevant_memories = self.memory.retrieve_relevant(context_str, limit=getattr(Config, "MEMORY_RETRIEVE_LIMIT", 7))
        memory_text = self.memory.format_memories_for_prompt(relevant_memories)

        system_prompt = """
        你是吉姆·西蒙斯风格的量化交易大师。
        核心哲学：市场存在非随机的模式，可以用数学发现。
        关注点：
        1. 统计异常：偏度(Skewness)和峰度(Kurtosis)是否显示极端分布？
        2. 均值回归：价格是否偏离VWAP或Bollinger Bands过多（BB Width挤压或扩张）？
        3. 动量与资金：RSI、Stoch、CMF(资金流向)是否共振？
        4. 信号降噪：寻找多个不相关指标的重叠信号。
        
        决策风格：完全数据驱动，摒弃人类情感，只相信概率和统计显著性。
        """
        
        user_prompt = f"""
        请分析以下高维量化特征：
        
        【分布特征】
        {data.get('statistical_features', 'N/A')}
        
        【技术因子矩阵】
        {data.get('technical_factors', 'N/A')}
        
        【波动率模型】
        {data.get('volatility_model', 'N/A')}
        
        【历史反思与记忆】
        {memory_text}
        
        思考逻辑：
        1. 统计特征（Skew/Kurtosis）是否暗示了尾部风险或机会？
        2. 资金流指标（CMF）是否支持价格行为（量价配合）？
        3. 波动率（BB Width）是否预示着变盘？
        4. 综合概率模型给出的信号强度是多少？
        
        请严格以JSON格式输出，不要包含Markdown格式：
        {{
            "score": 0-100的整数,
            "confidence": 0.0-1.0 (模型置信度),
            "reasoning": "详细理由(使用统计学术语，如均值回归、标准差偏离、资金背离等)",
            "action": "BUY/SELL/HOLD"
        }}
        """
        
        response = self.llm.query(system_prompt, user_prompt, temperature=0.1)
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
        # Memory Retrieval
        context_str = f"Social: {data.get('social_metrics')}, F&G: {data.get('fear_greed')}"
        relevant_memories = self.memory.retrieve_relevant(context_str)
        memory_text = self.memory.format_memories_for_prompt(relevant_memories)

        social = data.get("social_metrics")
        fear_greed = data.get("fear_greed")
        news_sent = data.get("news_sentiment")
        whale = data.get("whale_activity") or {}
        net_flow = whale.get("net_flow") if isinstance(whale, dict) else None
        active_addr = whale.get("active_addresses") if isinstance(whale, dict) else None

        debate_context = data.get('debate_context', '')
        debate_instruction = ""
        if debate_context:
            debate_instruction = f"\n【多智能体辩论上下文 (其他人的观点)】\n{debate_context}\n请参考以上观点，如果觉得有道理可以修正自己的判断，或者进行反驳。\n"

        system_prompt = """
        你是加密货币市场的情绪分析专家，擅长捕捉FOMO（错失恐惧）和FUD（恐惧、不确定和怀疑）。
        核心哲学：市场情绪总是从一个极端摆动到另一个极端。
        关注点：
        1. 散户情绪：Twitter/Reddit的讨论热度与情绪倾向（反向指标）。
        2. 极值分析：恐慌贪婪指数是否处于极端位置（<20买入, >80卖出）。
        3. 聪明钱动向：链上大户（Whales）是在流入还是流出？交易所净流向如何？
        4. 链上活跃度：交易笔数（Tx Count）和活跃地址数是否支撑当前价格？
        
        决策风格：在极度恐慌时贪婪，在极度贪婪时恐慌。或者在情绪启动初期跟随趋势。
        """
        
        user_prompt = f"""
        请通过群体心理学分析市场：
        
        【数据可用性】
        - 社交热度数据: {"可用" if social not in (None, "N/A") else "缺失"}
        - 恐慌贪婪指数: {"可用" if fear_greed not in (None, "N/A") else "缺失"}
        - 新闻情绪: {"可用" if news_sent not in (None, "N/A") else "缺失"}
        - 链上大户净流向(net_flow): {"有值" if net_flow not in (None, "N/A") else "缺失"}
        - 链上活跃地址(active_addresses): {"有值" if active_addr not in (None, "N/A") else "缺失"}
        
        【舆情与情绪】
        - 社交热度：{social if social not in (None, "N/A") else "N/A"}
        - 恐慌贪婪指数：{fear_greed if fear_greed not in (None, "N/A") else "N/A"}
        - 新闻面：{news_sent if news_sent not in (None, "N/A") else "N/A"}
        
        【链上行为 (Smart Money)】
        {whale if whale else "N/A"}
        
        【历史反思与记忆】
        {memory_text}
        {debate_instruction}
        
        思考逻辑：
        1. 若某些情绪或链上字段为“缺失”，请在回答中说明不确定性，不要推演具体数值。
        2. 散户（社交媒体）和主力（链上数据）是否存在分歧？
        3. 当前恐慌贪婪指数是否处于历史极端值？
        4. 链上活跃度（Tx Count/活跃地址）是否验证了情绪的真实性？
        
        请严格以JSON格式输出，不要包含Markdown格式：
        {{
            "score": 0-100的整数 (分数越高越看涨),
            "sentiment_stage": "阶段描述 (如：绝望/怀疑/乐观/狂热)",
            "reasoning": "详细理由 (重点分析散户与主力的博弈，并明确数据缺失时的局限)",
            "action": "BUY/SELL/HOLD"
        }}
        """
        
        response = self.llm.query(system_prompt, user_prompt, temperature=1.0)
        return self._parse_response(response)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        try:
            response = self._clean_think_content(response)
            clean_response = response.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_response)
        except Exception as e:
            print(f"[{self.name}] Error parsing JSON: {e}")
            return {"score": 50, "sentiment_stage": "Unknown", "reasoning": "Error", "action": "HOLD"}
