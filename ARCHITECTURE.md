# Affective Substrate — 架构与流程文档

## 1. 核心思想

给普通LLM外挂一个**非线性混沌动力学系统**，模拟生物内分泌系统，让LLM的神经元激活"活"起来——有人性、有情感、有记忆惯性。

不是prompt注入，不是RAG，而是**动力学系统调制LLM残差流**。

### 设计哲学

- **h_t（荷尔蒙状态）不是主记忆**——它只是情感状态调节器，"对这个人的整体感觉"
- **主记忆是Episodic Stream + Knowledge Graph + Reflection**——结构化、可检索、可反思
- **两套存储并行**：h_t走steering通道（不进prompt），事件记忆走prompt通道（按需检索）
- **三层Steering合成**：人格底色(h_t) + 情境激活(记忆情感标签) + 即时反应(VAD)

### 生物类比

| 工程组件 | 生物对应 | 功能 |
|----------|----------|------|
| 快系统 x(t) | 交感神经系统 | 快速应激、兴奋、警觉 |
| 慢系统 y(t) | 副交感神经系统 | 慢速调节、镇静、恢复 |
| 储备池 r(t) | 工作记忆 | 分布式时延、短期整合 |
| Episodic Memory | 海马体 | 情景记忆编码与检索 |
| Emotional Tags | 杏仁核 | 情感标记（杏仁核-海马耦合） |
| Knowledge Graph | 语义记忆网络 | 结构化事实与关系 |
| Reflection | 前额叶皮层 | 自动抽象与反思 |
| Steering注入 | 神经调制系统 | 全局状态调制 |

---

## 2. 系统架构总览

```
用户输入
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│                    Memory Manager (memory_manager.py)        │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ VAD      │  │ Affective    │  │ Episodic Memory       │  │
│  │ Analyzer │→ │ Substrate    │←→│ (带情感标签)           │  │
│  │          │  │ (core.py)    │  │                       │  │
│  └──────────┘  └──────┬───────┘  └───────────────────────┘  │
│                       │              ┌───────────────────┐   │
│                       │              │ Knowledge Graph   │   │
│                       │              │ (实体+关系)        │   │
│                       │              └───────────────────┘   │
│                       ▼                                      │
│              ┌─────────────────┐                             │
│              │ 3-Layer Steering│                             │
│              │ Vector合成      │                             │
│              └────────┬────────┘                             │
└───────────────────────┼─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Activation Steering Server                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Residual     │  │ Hopfield     │  │ Model Manager    │   │
│  │ Steering     │  │ Memory       │  │ (Qwen3.5-9B)    │   │
│  │ (hook.py)    │  │              │  │                  │   │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘   │
│         │                 │                    │             │
│         └─────────────────┼────────────────────┘             │
│                           ▼                                  │
│              Layers [15,18,21,24] 注入                       │
│              h' = h + injection                               │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
                   LLM 输出文本
                        │
                        ▼
                   VAD 分析 → 反馈给动力系统（闭环）
```

---

## 3. 核心组件详解

### 3.1 混沌动力学系统 (core.py)

#### 数学方程

**快系统（Mackey-Glass时滞微分方程）：**

```
dx_i/dt = β_i · x_i(t-τ_i) / (1 + x_i(t-τ_i)^n_i) − γ_i(y_i) · x_i + ξ_i(t)
```

- `β_i`：增益系数
- `τ_i`：时延（不同激素不同时间尺度）
- `n_i`：非线性阶数（饱和效应）
- `γ_i(y_i) = γ_0i + κ_i · y_i`：自适应阻尼（慢系统调节）
- `ξ_i(t)`：外部反馈（来自LLM输出的VAD分析）

**慢系统（积分-泄漏）：**

```
dy_i/dt = ε_i · (|x_i| − y_i),   ε_i << 1/τ_i
```

- 跟踪|x|的慢变包络
- 通过`γ(y) = γ₀ + κ·y`调节快系统的阻尼
- 概念等价于神经科学里的突触缩放(Turrigiano)

**储备池（回声状态网络）：**

```
dr/dt = −r/τ_r + tanh(W_r·r + W_x·x)
```

- 300个储备单元，谱半径0.9
- 分布式时延，工作记忆

#### 4种"激素"

| 变量 | 模拟 | τ(延迟) | β(增益) | n | 时间尺度 | 生物对应 |
|------|------|---------|---------|---|----------|----------|
| x[0] | 肾上腺素 | 10 | 0.2 | 10 | 秒级 | 快速兴奋/惊吓 |
| x[1] | 多巴胺 | 20 | 0.15 | 10 | 十秒级 | 快感/奖励 |
| x[2] | 皮质醇 | 30 | 0.1 | 12 | 半分钟 | 压力/焦虑 |
| x[3] | 血清素 | 50 | 0.08 | 8 | 分钟级 | 慢速满足/平静 |

#### 关键特性

- **情绪惯性**：刺激结束后按τ缓慢衰减，不是即刻归零
- **非线性饱和**：`(1+x^n)`分母产生适应性，持续强刺激不会无限飙升
- **混沌区间**：Lyapunov指数λ ≈ 0.01~0.1，"温和混沌"——丰富但不发散
- **双时间尺度分离**：快系统秒级波动，慢系统分钟级调节

---

### 3.2 VAD情绪分析器 (vad_analyzer.py)

将文本映射为4维情感向量 `[arousal, valence, dominance, stress]` ∈ [0,1]。

#### 数据源

1. **NRC情感词典**：11516词条（6491英文+5025中文），预处理为VAD连续值
2. **中文EmoBank CVAW**：5512中文词的V/A评分
3. **内置回退词典**：~60个常用情感词

#### 分析流程

```
文本 → 最长匹配分词(4→1字) → 词典查分
                                ↓
否定词翻转(窗口3词) → 程度副词放大/缩小 → emoji映射(40+个)
                                ↓
标点符号信号(！+0.15, ... -0.05)
                                ↓
聚合: mean(valence) + mean(arousal) + punctuation_boost
                                ↓
派生维度:
  dominance = 0.3 + 0.4*v + 0.2*(1-|a-0.5|*2)
  stress = 0.2 + 0.4*a*(1-v) + 0.2*(1-v)
```

---

### 3.3 情景记忆 (episodic_memory.py)

SQLite存储的带情感标签的事件流。

#### 记忆记录结构

```python
MemoryRecord:
  id: int
  user_id: str
  content: str              # 事件内容
  timestamp: float          # 写入时间
  importance: float         # 重要性评分 [0,1]
  emotional_tag: Dict       # {valence, arousal, dominance, stress} — 写入时冻结
  entities: List[str]       # 提取的实体
  session_id: str
  access_count: int         # 被检索次数
  last_accessed: float
  memory_type: str          # "episodic" / "reflection" / "event"
  parent_id: Optional[int]
```

#### 检索算法

四维加权评分：

```
score = 0.3 × recency + 0.3 × importance + 0.3 × relevance + 0.1 × emotional_congruence
```

- **Recency**：指数衰减 `exp(-0.693 × age_hours / 720)`，半衰期30天
- **Importance**：存储时评估的重要性值
- **Relevance**：查询词与内容的关键词重叠率
- **Emotional Congruence**：当前h_t向量与记忆emotional_tag的余弦相似度 → [0,1]

#### 心境一致性检索 (Mood-Congruent Recall)

当前情感状态偏置检索结果：悲伤时更容易回忆起悲伤的事。

**反心境机制 (Counter-Mood)**：检测到连续N轮valence < 0.3或 > 0.7时，翻转emotional_bias，防止抑郁回忆循环。

#### 记忆衰减

不常被检索的记忆重要性逐渐降低：`importance *= 0.995`（每轮）

---

### 3.4 知识图谱 (knowledge_graph.py)

SQLite存储的实体-关系图。

#### 数据结构

```
Entity: id, name, entity_type, properties, mention_count
Relation: source_id, target_id, relation_type, confidence
```

#### 规则提取模式

- `"我的猫叫X"` → (user, 拥有, X:pet)
- `"喜欢/讨厌X"` → (user, likes/dislikes, X:concept)
- `"在X工作/上学"` → (user, works_at/studies_at, X:org)
- `"朋友叫X"` → (user, 朋友, X:person)

#### 多跳推理

BFS遍历指定关系类型序列：`user → owns → 小花 → is → 猫` → "你的猫小花"

---

### 3.5 记忆管理器 (memory_manager.py)

整合所有组件的编排器。

#### 每轮处理流程 (process_turn)

```
1. VAD分析用户输入
   ↓
2. 恢复AffectiveSubstrate状态，步进动力学
   feedback = [arousal→肾上腺素, valence→多巴胺, stress→皮质醇, valence-stress→血清素]
   ↓
3. 心境一致性检索（带反心境检测）
   emotional_bias = h_t[:4]
   if 连续valence < 0.3 或 > 0.7: emotional_bias = -emotional_bias
   → 检索top-k记忆
   ↓
4. 知识图谱查询
   关键词提取 → 实体搜索 → 事实检索(最多5条)
   ↓
5. 三层Steering向量合成
   ↓
6. 检查是否触发Reflection
   ↓
7. 自动存储情景记忆 + 提取KG实体关系
```

#### 三层Steering向量合成

```python
# Layer 1: 人格底色（荷尔蒙状态 → 4个正交方向）
directions = gram_schmidt_random(4, d_model)  # 4个正交单位向量
a_s = A * tanh(s * (h_t - baseline))  # 快系统调制系数
a_p = A * tanh(s * (h_y - baseline))  # 慢系统调制系数
steering_personality = a_s @ directions - a_p @ directions

# Layer 2: 情境激活（记忆情感标签加权和）
steering_memory = Σ (tag_vec @ directions × importance) / n_memories

# Layer 3: 即时反应（当前VAD）
steering_immediate = vad_vec @ directions × 0.3

# 合成
steering = personality + 0.5 × memory + immediate
steering = clip(steering, norm_limit=√d_model × 0.15)
```

#### Reflection机制

每N轮触发一次（默认N=10）：

1. 取最近20条情景记忆
2. 计算平均valence/arousal/stress
3. 趋势检测：`np.polyfit`线性拟合valences
   - slope > 0.05 → "情绪改善中"
   - slope < -0.05 → "情绪恶化中"
4. 提取高频实体top-5
5. 生成摘要，存为`memory_type="reflection"`
6. 重置计数器

---

### 3.6 残差流注入 (hook.py + activation_steering_server.py)

#### Steering向量计算

```python
α_s = A · tanh(s · (x - x̄))  # 快系统调制
α_p = A · tanh(s · (y - ȳ))  # 慢系统调制
injection = α_s @ directions - α_p @ directions + P @ r
injection = clip(injection, norm=√d_model × 0.15)
```

- `A = 0.1`：最大调制幅度
- `s = 2.0`：灵敏度
- `directions`：4个Gram-Schmidt正交化的情感方向向量
- `P`：储备池→d_model投影矩阵

#### Hook注入

在Qwen3.5-9B的第15/18/21/24层注册forward hook：

```python
def hook_fn(module, input, output):
    if _active:
        output[0] += injection.unsqueeze(0).unsqueeze(0)
    return output
```

注入强度监控：`‖injection‖ / ‖h‖ < 0.2`，超过则裁剪。

#### Hopfield记忆召回

Modern Hopfield网络（Ramsauer 2020）：

```python
recall = softmax(ξ · sim(r_current, r_k) · importance_k) @ (r_k, x_k, y_k)
```

将recall向量通过投影矩阵P注入残差流——LLM"感觉到"被回忆激活，但不知道回忆了什么。

---

### 3.7 两种运行模式

#### 模式A：残差流注入（activation_steering_server.py）

- 直接加载模型（PyTorch + CUDA/MPS）
- 通过forward hook注入steering vector到残差流
- 效果最好，但需要GPU显存
- 注入层：15/18/21/24（中间层，语义已形成但还有时间传播）

#### 模式B：System Prompt注入（server.py）

- 通过Ollama API调用模型
- 将情感状态编码为文本注入system prompt
- 任何模型都能用，无需GPU
- 效果弱于直接注入，但兼容性最好

```python
# 模式B示例
context = f"[情感状态：激动度{arousal:.0%}，愉悦度{valence:.0%}，压力{stress:.0%}]"
system_prompt = base_prompt + context
```

---

## 4. 辅助组件

### 4.1 幂律延迟核 (power_law_delay.py)

替代固定τ的延迟缓冲，使用幂律核 `K(s) = α·s^{-α-1}`：

- 对数间隔采样点，Jacobian校正权重
- 更接近生物神经元的突触延迟分布
- `ExponentialDelayBuffer`：轻量替代，单个运行平均

### 4.2 客户端 (client.py)

交互式终端客户端：

- 维护本地AffectiveSubstrate实例
- 终端柱状图可视化激素状态
- matplotlib四面板图：快系统/慢系统/相图/VAD历史
- 命令：`quit`/`plot`/`state`

### 4.3 可视化 (visualize.py / demo_visualize.py / full_visualization.py)

- 激素时间序列图
- 相空间轨迹（x-y平面）
- VAD时间序列
- 情感标签热力图

---

## 5. 数据流：一次完整的对话轮次

```
用户: "今天考试考砸了，心情很差"
│
├─→ VAD分析: [arousal=0.3, valence=0.2, dominance=0.3, stress=0.8]
│
├─→ 动力系统步进:
│   feedback = [0.3→肾上腺素, 0.2→多巴胺, 0.8→皮质醇, -0.6→血清素]
│   x[0]+=dx, x[1]+=dx, x[2]+=dx, x[3]+=dx
│   y[i] += ε(|x[i]| - y[i])
│   r += dr
│
├─→ 心境一致性检索:
│   emotional_bias = [x[0], x[1], x[2], x[3]] = [0.3, 0.15, 0.6, 0.2]
│   → 检索到: "上次考试前很紧张"(valence=0.3, stress=0.7)
│   → 检索到: "期中考试考得很好"(valence=0.8, stress=0.4)
│   → 按emotional_congruence排序
│
├─→ 知识图谱查询:
│   提取关键词"考试" → 无匹配实体
│
├─→ 三层Steering合成:
│   personality: h_t → 方向投影
│   memory: 2条记忆的情感标签加权 → 方向投影
│   immediate: [0.3, 0.2, 0.3, 0.8] → 方向投影 × 0.3
│   steering = personality + 0.5×memory + immediate
│   steering = clip(steering, norm=9.6)
│
├─→ 注入LLM残差流:
│   layers[15].output[0] += steering
│   layers[18].output[0] += steering
│   layers[21].output[0] += steering
│   layers[24].output[0] += steering
│
├─→ LLM生成回复:
│   "考试没考好确实很难受，但一次考试不能定义你..."
│
├─→ 后处理:
│   ├─ 存入情景记忆(content, importance=0.7, emotional_tag={v:0.2,a:0.3,d:0.3,s:0.8})
│   ├─ 提取KG实体(考试→concept)
│   ├─ 更新h_t状态（持久化到JSON）
│   └─ 检查reflection触发条件
│
└─→ 闭环完成
```

---

## 6. 与VTuber/VRM的整合

荷尔蒙状态直接映射到面部表情：

| 激素 | VRM表情 | 映射 |
|------|---------|------|
| x[0] 肾上腺素 | 眼睛大小 | x[0]大→睁大眼 |
| x[1] 多巴胺 | 微笑程度 | x[1]大→微笑 |
| x[2] 皮质醇 | 眉头紧锁 | x[2]大→皱眉 |
| x[3] 血清素 | 放松程度 | x[3]大→放松表情 |

---

## 7. 持久化

| 数据 | 存储方式 | 路径 |
|------|----------|------|
| h_t状态（荷尔蒙） | JSON (per-user) | `states/{user_id}.json` |
| 情景记忆 | SQLite | `episodic_memory.db` |
| 知识图谱 | SQLite | `knowledge_graph.db` |
| NRC词典 | JSON | `data/nrc_va_lexicon.json` |

---

## 8. 测试状态

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 1 | Episodic存储/检索/用户隔离/衰减 | ✅ 通过 |
| Phase 2 | Reflection触发/情感趋势/Mood-congruent/Counter-mood | ✅ 通过 |
| Phase 3 | KG实体提取/关系/多跳推理 | ✅ 通过 |

---

## 9. 待做

- [ ] 适配Mac Mini MPS + Qwen3.6-35B-A3B
- [ ] 封装为Hermes Skill
- [ ] VRM表情联动（情绪→VRM表情映射）
- [ ] 多用户支持
- [ ] LLM-based实体提取（替代规则方法）
- [ ] 在线评估（语义多样性/时间自相关/人工A/B）
