# Affective Substrate — 情感基底系统

给LLM外挂非线性混沌动力学系统，让AI有人性、有情感、有记忆。

## 这是什么

一个生物启发的AI情感架构：用Mackey-Glass混沌方程模拟人体激素系统，通过残差流注入调制LLM的内部状态，配合带情感标签的情景记忆和知识图谱，让AI在跨会话中形成持续的情感状态和对用户的深入理解。

**不是**prompt注入，**不是**RAG，是**动力学系统调制LLM激活**。

## 架构

```
用户输入 → VAD分析 → 混沌动力学(Mackey-Glass) → 残差流注入 → LLM
              ↓              ↑
         情感标签 ← Episodic Memory (带情感标签的事件流)
                        ↑
                   Knowledge Graph (结构化事实)
                        ↑
                   Reflection (定期自动抽象)
```

**双系统对抗**：
- 快系统 x(t)：4种激素（肾上腺素/多巴胺/皮质醇/血清素），Mackey-Glass混沌，秒级
- 慢系统 y(t)：积分-泄漏，副交感调节，分钟级
- 储备池 r(t)：300单元回声状态网络，工作记忆

**三层Steering合成**：
```
steering = h_t(人格底色) + Σ记忆情感标签(情境激活) + VAD(即时反应)
```

## 核心组件

| 组件 | 文件 | 功能 |
|------|------|------|
| 混沌动力学 | `core.py` | Mackey-Glass + 慢系统 + 储备池 |
| VAD分析器 | `vad_analyzer.py` | 文本→4维情感向量 |
| 情景记忆 | `episodic_memory.py` | SQLite，带情感标签的事件流 |
| 知识图谱 | `knowledge_graph.py` | 实体-关系图，多跳推理 |
| 记忆管理器 | `memory_manager.py` | 编排所有组件，Steering合成 |
| 残差流注入 | `activation_steering_server.py` | Hook注入LLM中间层 |
| Hopfield记忆 | `hopfield_memory.py` | Modern Hopfield关联召回 |
| 幂律延迟 | `power_law_delay.py` | 分布式时延核 |
| 轻量服务 | `server.py` | Ollama API + Prompt注入模式 |
| 客户端 | `client.py` | 交互式终端 + 可视化 |

## 4种"激素"

| 激素 | τ(衰减) | 时间尺度 | 生物对应 |
|------|---------|----------|----------|
| 肾上腺素 | 10步 | 秒级 | 快速兴奋/惊吓 |
| 多巴胺 | 20步 | 十秒级 | 快感/奖励 |
| 皮质醇 | 30步 | 半分钟 | 压力/焦虑 |
| 血清素 | 50步 | 分钟级 | 慢速满足/平静 |

## 两种运行模式

### 模式A：残差流注入（效果最好）

- 直接加载模型（PyTorch）
- 通过forward hook在第15/18/21/24层注入steering vector
- 需要GPU显存

### 模式B：System Prompt注入（通用）

- 通过Ollama API调用
- 情感状态编码为文本注入system prompt
- 任何模型都能用

## 安装

```bash
pip install numpy matplotlib
# 模式A额外需要:
pip install torch transformers
```

## 快速开始

### 模式B（轻量，推荐先试）

```bash
# 启动Ollama
ollama serve

# 启动情感服务
python server.py

# 另一个终端，启动客户端
python client.py
```

### 模式A（残差流注入）

```bash
# 启动注入服务器（需要GPU）
python activation_steering_server.py

# 客户端连接
python client.py --server http://localhost:8000
```

## 文件结构

```
affective-substrate/
├── core.py                      # Mackey-Glass混沌动力学
├── vad_analyzer.py              # VAD情绪分析器
├── episodic_memory.py           # 情景记忆（SQLite）
├── knowledge_graph.py           # 知识图谱（SQLite）
├── memory_manager.py            # 记忆管理器（编排器）
├── activation_steering_server.py # 残差流注入服务器
├── hook.py                      # ResidualSteering基础设施
├── hopfield_memory.py           # Modern Hopfield关联记忆
├── power_law_delay.py           # 幂律延迟核
├── server.py                    # 轻量Ollama服务
├── client.py                    # 交互式客户端
├── visualize.py                 # 可视化工具
├── demo_visualize.py            # 演示可视化
├── full_visualization.py        # 完整可视化
├── test_full.py                 # 完整测试
├── test_memory.py               # 记忆测试
├── test_run.py                  # 运行测试
├── verify_steering.py           # Steering验证
├── check_*.py                   # 各种检查脚本
├── ARCHITECTURE.md              # 详细架构文档
├── data/
│   └── nrc_va_lexicon.json      # NRC情感词典（11516词条）
└── README.md
```

## 详细文档

- [ARCHITECTURE.md](ARCHITECTURE.md) — 完整架构、数学方程、数据流、每个组件的详解

## 状态

Phase 1-3 测试全部通过（情景记忆/Reflection/知识图谱）。

**已完成**：混沌动力学、VAD分析、情景记忆、知识图谱、记忆管理、残差流注入

**待做**：Mac Mini MPS适配、Hermes Skill封装、VRM表情联动、LLM-based实体提取

## 灵感来源

- Mackey & Glass (1977) — 时滞微分方程建模生理控制
- Anthropic Persona Vectors (2024-2025) — 激活工程
- Zou et al. Representation Engineering (2023) — 表征工程方法论
- Stanford Generative Agents (2023) — 情景记忆 + Reflection
- Microsoft GraphRAG / HippoRAG — 结构化图检索
- MemGPT / Letta — Agent自管理记忆
- Ramsauer 2020 — Modern Hopfield网络
- Lisa Feldman Barrett《情绪的构建》— "情绪 = 内感受 + 概念化"
