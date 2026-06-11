"""
hopfield_memory.py
===================
L2: Modern Hopfield 片段记忆

核心机制：
  - 显著时刻（高 |dx/dt|）自动触发存储
  - 当前状态查询记忆库（Ramsauer 2020 形式）
  - recall 项加到储备池动力学上作为偏置
  - LLM 不知道"想起了什么"，只感受到状态变化

数学：
  存储: M ← M ∪ {(r_k, x_k, y_k)} 当 |ẋ_k| > θ
  查询: recall = Σ_k softmax(ξ⟨r(t), r_k⟩) · (r_k, x_k, y_k)
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MemoryEntry:
    """单条记忆"""
    r: np.ndarray       # 储备池状态快照
    x: np.ndarray       # 快系统状态
    y: np.ndarray       # 慢系统状态
    turn: int           # 存储时的对话轮次
    text_snapshot: str  # 当时的文本片段（可选，用于调试）
    importance: float   # 重要性评分


class HopfieldMemory:
    """
    Modern Hopfield 片段记忆
    
    用法：
        mem = HopfieldMemory(n_reservoir=300, n_hormones=4)
        
        # 每轮检查是否触发存储
        if mem.should_store(dx, turn):
            mem.store(r, x, y, turn, text)
        
        # 查询回忆
        recall_r, recall_x, recall_y = mem.recall(r_current)
    """
    
    def __init__(
        self,
        n_reservoir: int = 300,
        n_hormones: int = 4,
        max_memories: int = 200,
        storage_threshold: float = 0.3,    # |dx/dt| 阈值
        temperature: float = 10.0,          # softmax 温度 ξ
        decay_rate: float = 0.001,          # 记忆衰减率
        importance_decay: float = 0.995,    # 重要性衰减
    ):
        self.n_reservoir = n_reservoir
        self.n_hormones = n_hormones
        self.max_memories = max_memories
        self.storage_threshold = storage_threshold
        self.temperature = temperature
        self.decay_rate = decay_rate
        self.importance_decay = importance_decay
        
        self.memories: List[MemoryEntry] = []
        self.total_stored = 0
        self.total_recalled = 0
    
    def should_store(self, dx: np.ndarray, turn: int) -> bool:
        """
        判断是否应该存储当前状态
        
        触发条件：
          1. |dx/dt| 超过阈值（显著变化）
          2. 距离上次存储至少3轮（避免冗余）
        """
        dx_norm = np.linalg.norm(dx)
        
        if dx_norm < self.storage_threshold:
            return False
        
        # 距离上次存储至少3轮
        if self.memories and (turn - self.memories[-1].turn) < 3:
            return False
        
        return True
    
    def store(
        self, 
        r: np.ndarray, 
        x: np.ndarray, 
        y: np.ndarray, 
        turn: int,
        text: str = "",
        importance: float = None,
    ):
        """存储一条记忆"""
        dx_norm = np.linalg.norm(x)  # 用当前 |x| 作为重要性估计
        if importance is None:
            importance = dx_norm
        
        entry = MemoryEntry(
            r=r.copy(),
            x=x.copy(),
            y=y.copy(),
            turn=turn,
            text_snapshot=text[:200] if text else "",
            importance=importance,
        )
        
        self.memories.append(entry)
        self.total_stored += 1
        
        # 超过容量时，移除最不重要的记忆
        if len(self.memories) > self.max_memories:
            # 按重要性排序，移除最弱的
            self.memories.sort(key=lambda m: m.importance, reverse=True)
            self.memories = self.memories[:self.max_memories]
    
    def recall(self, r_current: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        查询记忆库，返回加权回忆
        
        Modern Hopfield (Ramsauer 2020):
          recall = Σ_k softmax(ξ · ⟨r, r_k⟩) · (r_k, x_k, y_k)
        
        Returns:
            recall_r: 储备池回忆 [n_reservoir]
            recall_x: 快系统回忆 [n_hormones]
            recall_y: 慢系统回忆 [n_hormones]
            weights: 注意力权重 [n_memories]（用于调试）
        """
        if not self.memories:
            zr = np.zeros(self.n_reservoir)
            zx = np.zeros(self.n_hormones)
            zy = np.zeros(self.n_hormones)
            return zr, zx, zy, np.array([])
        
        # 计算注意力权重
        k = len(self.memories)
        similarities = np.zeros(k)
        for i, mem in enumerate(self.memories):
            # 余弦相似度
            sim = np.dot(r_current, mem.r) / (
                np.linalg.norm(r_current) * np.linalg.norm(mem.r) + 1e-8
            )
            # 用重要性加权
            similarities[i] = self.temperature * sim * mem.importance
        
        # softmax
        similarities -= np.max(similarities)  # 数值稳定
        weights = np.exp(similarities)
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights /= weight_sum
        else:
            weights = np.ones(k) / k
        
        # 加权求和
        recall_r = np.zeros(self.n_reservoir)
        recall_x = np.zeros(self.n_hormones)
        recall_y = np.zeros(self.n_hormones)
        
        for i, mem in enumerate(self.memories):
            recall_r += weights[i] * mem.r
            recall_x += weights[i] * mem.x
            recall_y += weights[i] * mem.y
        
        self.total_recalled += 1
        
        return recall_r, recall_x, recall_y, weights
    
    def decay(self):
        """记忆衰减——不常被回忆的记忆重要性降低"""
        for mem in self.memories:
            mem.importance *= self.importance_decay
    
    def get_stats(self) -> dict:
        """获取记忆统计"""
        if not self.memories:
            return {
                "n_memories": 0,
                "total_stored": self.total_stored,
                "total_recalled": self.total_recalled,
            }
        
        importances = [m.importance for m in self.memories]
        turns = [m.turn for m in self.memories]
        
        return {
            "n_memories": len(self.memories),
            "total_stored": self.total_stored,
            "total_recalled": self.total_recalled,
            "avg_importance": float(np.mean(importances)),
            "max_importance": float(np.max(importances)),
            "min_importance": float(np.min(importances)),
            "turn_range": [min(turns), max(turns)],
        }
    
    def get_recent(self, n: int = 5) -> List[dict]:
        """获取最近 n 条记忆"""
        recent = self.memories[-n:]
        return [{
            "turn": m.turn,
            "importance": m.importance,
            "x": m.x.tolist(),
            "y": m.y.tolist(),
            "text": m.text_snapshot[:100],
        } for m in recent]
