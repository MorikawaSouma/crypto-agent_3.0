# Multi-Agent Crypto Trading System (AI-Powered) 3.0

这是一个基于大型语言模型 (LLM) 和多智能体架构 (Multi-Agent) 的加密货币量化交易与回测系统。

系统模拟了顶级投资者的思维模式（如巴菲特、索罗斯、达里欧、西蒙斯），结合 **真实市场数据**、**宏观经济数据**、**链上数据** 和 **实时新闻情绪**，提供深入的投资分析和交易决策。

## 3.0 版本更新

### 1. 多智能体多轮辩论

这一版保留了五个核心智能体：巴菲特、索罗斯、达里欧、西蒙斯和情绪智能体，在此基础上实现了真正的“多轮辩论机制”：

- 对每一个被选中的 Symbol，MasterAgent 会在同一时间截面上构造一份“事实底稿”，包括：
  - 市场价格与技术指标（K 线、波动率、回撤等）；
  - 宏观与情绪数据（VIX、恐慌贪婪指数等）；
  - 链上和 DeFi 相关数据（TVL、活跃地址等，如可用）；
  - 新闻与事件摘要（如可用）。
- 第一轮中，每个智能体只基于自己那一块数据和角色设定，独立给出：
  - 对该币的操作建议：BUY / SELL / HOLD；
  - 一个 0–100 的打分（Ray Dalio 给出的是 `weight_suggestion`）；
  - 完整的文本推理过程。
- 第一轮结束后，MasterAgent 会把这一轮所有人的结论整理成结构化的 round summary，包括：
  - 每个智能体的 action（BUY/SELL/HOLD）；
  - score 或 weight_suggestion；
  - reasoning 的前 150 字摘要。
- 从第二轮开始，MasterAgent 会把 round summary 作为“辩论上下文”注入到每个智能体的输入里，并明确提示：
  > “这是上一轮所有人的观点，请你在本轮中参考这些观点，对自己的判断进行补充、质疑或修正。”

通过多轮迭代，这五个智能体会从“各说各话”逐步收敛到“有共识的状态”。当前默认进行 3 轮讨论（可在 `config.py` 中通过 `DEBATE_ROUNDS` 调整），也可以将 `AGENT_MODE` 切换为 `voting` 回退到单轮投票模式。

### 2. 智能体的记忆和反思机制

3.0 版本为每个智能体接入了独立的“语义记忆”，用 ChromaDB 做本地持久化的向量数据库：

- 每个智能体在初始化时，会把自己的名字传给 `MemoryManager`，后者基于此生成一个安全的 collection 名。可以类比为：
  - 每个 Agent 有一张只属于自己的“记忆表”，互相无法看到彼此的历史记录。
- 每一条记忆由三部分构成：
  - 我们自己拼接的一段文本：包含当时的 context、decision、outcome、reasoning，是一条完整的“故事”；
  - `metadatas`：一个 dict，包含 `timestamp`、`decision`、`outcome`、`agent` 等标签，用于过滤和解释；
  - 对上述文本做 Embedding 得到的一串浮点向量，语义相近的记忆在向量空间里会更接近。
- 在每次智能体做出新决策前，会先基于当前的 context，在自己的记忆库中检索：
  - 最多 `MEMORY_RETRIEVE_LIMIT` 条（配置项可调）；
  - 语义上最接近当前场景、且 outcome 为 `"loss"` 的历史案例；
  - 并将这些“失败教训”注入提示词中，作为反思材料。
- 在回测开始前，可以通过配置决定是否清空所有记忆：
  - 若清空，则回测在“冷启动”状态下进行；
  - 若不清空，则回测与后续实盘会沿用历史记忆，形成长期经验。

这样，每个智能体不再是无状态的 LLM 调用，而是会“记住”自己过去在哪些场景下亏过钱，并在类似场景中倾向于更加保守或调整判断。

### 3. 基于 PyTorch MLP 的动态权重分配

为了让系统不只是简单的“少数服从多数”，3.0 引入了一个基于 PyTorch 的轻量级 MLP 模型，用来在五个智能体之间做动态权重分配。相比早期版本手写的 Q/K 注意力，这一版完全使用官方封装的模块（nn.Linear、nn.ReLU、Adam 等），只关注“输入是什么、loss 怎么写”。

- 输入特征向量 X 包括：
  - 各智能体当前的打分（Buffett / Soros / Dalio / Simons / 情绪）以及 Dalio 的仓位建议；
  - 当前的宏观环境指标（如 VIX、恐慌贪婪指数等）；
  - 过去一段时间价格与成交量的统计特征（均值、标准差、偏度、峰度等）。
- 模型结构是一个两层的前馈网络：
  - 第 1 层：Linear(input_dim → hidden_dim) + ReLU，其中 hidden_dim 会根据 input_dim 自动选在 [16, 128] 的合理区间；
  - 第 2 层：Linear(hidden_dim → 5)，输出对应五个智能体的原始得分；
  - 再除以 temperature（≈√input_dim）做缩放，经 softmax 转成 5 个加总为 1 的权重。
- 从直觉上看，这一层就是在做：“当前这个市场状态下，更应该听谁的？”——在该状态下历史表现更好的智能体，其对应的权重会被模型学得更高。

与此同时，模型会在每次交易平仓（产生 Profit 或 Loss）后进行在线更新：

- 对于一笔盈利的交易，会奖励“当时站在系统同一方向”的智能体，在类似市场状态下提升它们的权重（loss = -平均权重）；
- 对于一笔亏损的交易，会惩罚“当时站在系统同一方向”的智能体，在类似状态下降低它们的权重（loss = 平均权重）。

在足够长的时间维度上，如果某个智能体在“趋势行情”里长期表现更好，它在类似趋势环境下的权重就会被模型自发地调高；反之，在“震荡行情”里表现稳定的智能体，其权重也会在震荡期逐步成为主导，实现了“谁更适合当前市场 regime，谁的声音就更大”。

### 4. 机器学习选股机制

3.0 版本不再手动指定交易标的，而是先通过机器学习做一层“量化选股”，再交给智能体做深度分析：

- 从市值较高的 15 个主流币构成基础 Universe（`Config.UNIVERSE`）；
- 基于日线 OHLCV 构建一套 Alpha158 风格的价量因子（见 `factors.py`），包括多周期动量、波动、成交量结构、技术指标等；
- 使用 LightGBM 回归模型，学习在什么样的因子组合下，**下一天的收益率** 更有可能为正且更大（标签为预测“下一天收益率”）；
- 采用滚动训练（`RollingModelManager`），定期在最新一段历史上重新训练模型，以适应市场环境变化；
- 在回测和实盘信号生成时，模型会在 Universe 上对所有币进行一轮横截面打分，选出 Top-K 作为候选标的，再交给多智能体辩论与权重分配模块。

这样可以保证选股过程有一个量化、可回测的基础，而不是完全依赖主观设定的币种列表。

### 5. 数据问题

为了保障结论的可信度，本版本尽量移除了对硬编码和纯模拟数据的依赖，在宏观、链上、情绪等关键维度上优先使用真实数据源；如果某一类数据无法获取，系统会明确在提示词中标记“数据缺失”，而不是伪造数值：

- 对于缺失的宏观或链上数据，MasterAgent 会直接把“缺失”这一事实传递给各个智能体；
- 智能体在推理过程中需要显式说明“该部分数据不可用”，并在此基础上调整自己的风险偏好和操作建议：
  - 尤其是巴菲特风格的价值智能体，在估值和安全边际关键数据缺失时，会明显倾向于从 BUY 降级为 HOLD；
- 一些免费 API 在品类和频率上存在限制（例如分钟级访问次数限制）；
- 后续版本会考虑接入部分付费数据源，以进一步提升宏观和链上数据的覆盖率与稳定性。

总体来说，3.0 版本在不“造假数据”的前提下，把多智能体辩论、长期记忆、动态权重分配和机器学习选股这几块真正打通，为后续 4.0 版本的策略优化打下基础。

## 安装指南

### 1. 环境准备
推荐使用 Python 3.10+ 和虚拟环境：

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. 安装依赖
```bash
pip install -r crypto_trading_system/requirements.txt
```

### 3. 配置环境变量
复制 `.env.example` 为 `.env` 并填入 API Key：

```ini
# DeepSeek API (核心)
DEEPSEEK_API_KEY=sk-your-key-here

# 辅助数据 API Key (可选)
# CryptoPanic 新闻 API (推荐申请，免费版即可)
CRYPTOPANIC_API_KEY=your-cryptopanic-key
# 如果不填，系统会自动使用 RSS 模式抓取新闻

# 交易所 API (实盘交易需要)
BINANCE_API_KEY=your-binance-key
BINANCE_SECRET=your-binance-secret
```

## 使用方法

### 1. 运行历史回测 (Backtest)
在 `config.py` 中配置回测时间段 (`START_TIME`, `END_TIME`) 和交易币种 (`SYMBOLS`)。

```bash
python -m crypto_trading_system.run_backtest
```
系统将自动拉取该时间段的历史价格、宏观数据和新闻（如有存档），模拟 Agent 辩论过程，并输出资金曲线。

### 2. 运行实时信号 (Live Analysis)
分析当前市场状况，生成多智能体投资报告。

```bash
python -m crypto_trading_system.main
```

### 3. 运行 ML 策略回测
使用 LightGBM 模型进行纯量化回测。

```bash
python -m crypto_trading_system.run_ml_backtest
```

## 项目结构

```
crypto_trading_system/
├── agents/                 # 智能体核心逻辑
│   ├── master_agent.py     # 主控智能体 (辩论主持人)
│   ├── strategy_agents.py  # 具体策略智能体 (Buffett, Soros, etc.)
│   └── memory.py           # ChromaDB 记忆模块
├── real_data_provider.py   # 真实数据源聚合 (DefiLlama, News, Yahoo, etc.)
├── config.py               # 全局配置
├── run_backtest.py         # Agent 回测入口
└── requirements.txt        # 依赖列表
```

## 免责声明

本项目仅供教育和研究使用，不构成任何投资建议。加密货币市场波动巨大，使用自动化交易系统存在资金损失风险。请对自己负责。

## 📄 License

MIT License
