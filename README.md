# 已新建文件夹（Just kidding）

Multi-Agent Crypto Trading System 3.0

本系统基于大型语言模型和多智能体架构，结合实时市场、宏观、链上与情绪数据，提供投资分析、交易决策与历史回测。核心目标是让不同风格的策略智能体在投票或辩论两种模式下协同工作，并通过长期记忆与反思避免重复犯错，提升策略稳定性与收益质量。

功能总览
- 多智能体协同：巴菲特（价值）、索罗斯（宏观趋势）、达利欧（风险平价与仓位）、西蒙斯（量化因子）、情绪分析（FOMO/FUD），主控智能体统一融合
- 模式切换：支持并行投票（Parallel Execution）与多轮辩论（Debate），可随时切换
- 长期记忆与反思：集成向量数据库（ChromaDB），在每次决策前检索相似的历史错误案例，决策后写入反思以形成闭环
- 仓位与风险控制：由达利欧智能体输出目标权重，配合波动率逆向、单资产权重上限、现金缓冲、平滑处理
- 动态权重融合：主控智能体使用轻量在线学习模型（SGDRegressor）按上下文动态分配各智能体权重，并依据盈亏进行更新

系统架构
- 主控智能体（MasterAgent）：统一协调数据、模式（投票/辩论）、权重融合与反思反馈。[master_agent.py](file:///d:/agent2/agent2/crypto_trading_system/agents/master_agent.py)
- 策略智能体（Strategy Agents）：分别实现各自风格的分析与输出，支持辩论上下文与记忆检索。[strategy_agents.py](file:///d:/agent2/agent2/crypto_trading_system/agents/strategy_agents.py)
- 基类与记忆（Base & Memory）：所有智能体共享的 LLM 调用、反思接口与向量数据库管理。[base_agent.py](file:///d:/agent2/agent2/crypto_trading_system/agents/base_agent.py)、[memory.py](file:///d:/agent2/agent2/crypto_trading_system/agents/memory.py)
- 特征工程与数据：技术指标与数据获取（交易所、宏观、情绪、链上）。[feature_engineering.py](file:///d:/agent2/agent2/crypto_trading_system/feature_engineering.py)、[data_loader.py](file:///d:/agent2/agent2/crypto_trading_system/data_loader.py)、[real_data_provider.py](file:///d:/agent2/agent2/crypto_trading_system/real_data_provider.py)
- 回测引擎：单资产与多资产回测，集成仓位控制与反思反馈。[agent_backtest.py](file:///d:/agent2/agent2/crypto_trading_system/agent_backtest.py)、[run_backtest.py](file:///d:/agent2/agent2/crypto_trading_system/run_backtest.py)、[run_agent_backtest.py](file:///d:/agent2/agent2/crypto_trading_system/run_agent_backtest.py)
- LLM 客户端：深度求索模型调用（支持 mock）。[llm_client.py](file:///d:/agent2/agent2/crypto_trading_system/llm_client.py)

安装与环境
- 推荐使用 Python 虚拟环境（venv 或 conda）
- 安装依赖：pip install -r requirements.txt；另外安装 chromadb 与 scikit-learn
- 如需代理，设置环境变量或 .env 中的 PROXY_URL（例如 http://127.0.0.1:7897）
- 注意：首次使用 ChromaDB 默认嵌入模型会下载（需网络）

配置说明
- 配置文件：[config.py](file:///d:/agent2/agent2/crypto_trading_system/config.py)
- 主要参数：
  - SYMBOLS：回测或分析的交易对列表
  - START_TIME/END_TIME：分析/回测时间区间（字符串 'YYYY-MM-DD HH:MM:SS'）
  - DEFAULT_LIMIT：当未指定时间范围时，用于拉取最新数据的数量
  - AGENT_MODE：默认 "debate"，可设为 "voting"（并行投票）或 "debate"（多轮辩论）
  - DEBATE_ROUNDS：辩论轮数（建议 2–5），默认 3
  - INITIAL_CAPITAL：初始资金（用于回测）
  - DEEPSEEK_API_KEY/DEEPSEEK_BASE_URL/MODEL_NAME：LLM 配置

运行指南
- 实时分析：
  - 入口：[main.py](file:///d:/agent2/agent2/crypto_trading_system/main.py)
  - 根据 Config 的 SYMBOLS 与时间范围获取数据，输出分析与决策报告（可扩展导出 CSV）
- 单资产回测：
  - 入口：[run_backtest.py](file:///d:/agent2/agent2/crypto_trading_system/run_backtest.py)
  - 回测驱动：[agent_backtest.py:run_single_asset](file:///d:/agent2/agent2/crypto_trading_system/agent_backtest.py#L14-L139)
  - 达利欧智能体提供目标权重，按风险控制与阈值进行调仓，反思在每步比较上一决策盈亏时触发
- 多资产回测：
  - 入口：[run_agent_backtest.py](file:///d:/agent2/agent2/crypto_trading_system/run_agent_backtest.py)
  - 回测驱动：[agent_backtest.py:run_multi_asset](file:///d:/agent2/agent2/crypto_trading_system/agent_backtest.py#L141-L415)
  - 汇总所有资产的最新决策，计算目标组合权重并调仓；反思对每个资产分别触发并反馈到主控

模式与辩论
- 并行投票（voting）：各智能体在独立上下文中输出结果，由主控根据动态权重融合为最终得分与动作
- 多轮辩论（debate）：每轮汇总所有智能体观点的摘要并注入上下文，促使下一轮修正；轮数由 DEBATE_ROUNDS 或运行期 master.set_debate_rounds(n) 控制
- 运行期切换：master.set_mode("voting"|"debate")；若需脚本统一读取配置，可在入口脚本初始化后调用 set_mode(Config.AGENT_MODE)、set_debate_rounds(Config.DEBATE_ROUNDS)

长期记忆与反思
- 决策前：各智能体从向量数据库检索“相似的历史错误案例”，避免在同一模式下重复犯错
- 决策后：当回测判定上一决策为亏损或盈利事件时，主控调用 reflect_on_trade 将结果反馈到智能体的记忆库，并用轻量模型在线更新权重分布
- 相关代码：
  - 反思入口：[master_agent.py:reflect_on_trade](file:///d:/agent2/agent2/crypto_trading_system/agents/master_agent.py#L314-L359)
  - 记忆管理（ChromaDB）：[memory.py](file:///d:/agent2/agent2/crypto_trading_system/agents/memory.py)
  - 智能体检索与上下文注入：[strategy_agents.py](file:///d:/agent2/agent2/crypto_trading_system/agents/strategy_agents.py)

仓位与风险控制
- 达利欧智能体输出权重建议（weight_suggestion），系统进行：
  - 波动率逆向（风险平价风格）：按 1/vol 调整权重
  - 单资产上限（如 40%）：避免集中风险
  - 现金缓冲（如 5%）：提升抗冲击能力
  - 平滑权重（smoothing_alpha）：降低换手率与交易成本
- 单资产与多资产的调仓均有 1% 阈值以避免碎片化交易

动态权重融合
- 主控使用轻量在线学习（SGDRegressor）分配各智能体权重
- 正反馈：当一致观点带来盈利，提高对应智能体权重；负反馈相反
- 支持在回测期间持续学习，适配市场环境变化

快速上手
- 配置 SYMBOLS、时间区间与 AGENT_MODE/DEBATE_ROUNDS
- 运行实时分析或回测入口脚本
- 查看输出的日志、CSV 与图表（多资产脚本会保存结果与图）
- 如需尝试不同辩论深度，可调整 DEBATE_ROUNDS 或运行期调用 master.set_debate_rounds(n)

注意事项
- 本项目仅用于教育与研究，不构成任何投资建议
- 加密市场波动剧烈，自动化策略存在风险；使用时请控制仓位与风险
