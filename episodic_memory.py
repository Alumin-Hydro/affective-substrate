"""
episodic_memory.py
==================
Episodic Stream — 情景记忆流

每条记忆：
  - content: 事件描述文本
  - timestamp: 创建时间
  - importance: 重要性评分 (0-1)
  - emotional_tag: 4维情感标签 {valence, arousal, dominance, stress}
  - entities: 提及的实体列表
  - session_id: 所属会话
  - access_count: 被检索次数（遗忘衰减用）
  - last_accessed: 最近检索时间

存储: SQLite（轻量、单文件、无外部依赖）
检索: recency × importance × relevance 三维加权
"""

import sqlite3
import json
import time
import os
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass
class MemoryRecord:
    """单条情景记忆"""
    id: Optional[int] = None
    user_id: str = ""
    content: str = ""
    timestamp: float = 0.0
    importance: float = 0.5
    emotional_tag: Dict = field(default_factory=lambda: {"valence": 0.5, "arousal": 0.3, "dominance": 0.5, "stress": 0.3})
    entities: List[str] = field(default_factory=list)
    session_id: str = ""
    access_count: int = 0
    last_accessed: float = 0.0
    memory_type: str = "episodic"  # episodic / reflection / event
    parent_id: Optional[int] = None  # reflection的来源ID


class EpisodicMemory:
    """
    SQLite-backed 情景记忆流
    
    用法:
        mem = EpisodicMemory("memory.db")
        mem.store(user_id="u1", content="用户提到猫去世了", 
                  importance=0.9, emotional_tag={...})
        results = mem.retrieve(user_id="u1", query="宠物", top_k=5)
    """
    
    def __init__(self, db_path: str = "episodic_memory.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                importance REAL DEFAULT 0.5,
                emotional_tag TEXT DEFAULT '{}',
                entities TEXT DEFAULT '[]',
                session_id TEXT DEFAULT '',
                access_count INTEGER DEFAULT 0,
                last_accessed REAL DEFAULT 0,
                memory_type TEXT DEFAULT 'episodic',
                parent_id INTEGER DEFAULT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user ON memories(user_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_type ON memories(memory_type)
        """)
        conn.commit()
        conn.close()
    
    def store(
        self,
        user_id: str,
        content: str,
        importance: float = 0.5,
        emotional_tag: Optional[Dict] = None,
        entities: Optional[List[str]] = None,
        session_id: str = "",
        memory_type: str = "episodic",
        parent_id: Optional[int] = None,
    ) -> int:
        """存储一条记忆，返回ID"""
        if emotional_tag is None:
            emotional_tag = {"valence": 0.5, "arousal": 0.3, "dominance": 0.5, "stress": 0.3}
        if entities is None:
            entities = self._extract_entities(content)
        
        now = time.time()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            INSERT INTO memories 
            (user_id, content, timestamp, importance, emotional_tag, entities, 
             session_id, access_count, last_accessed, memory_type, parent_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
        """, (
            user_id, content, now, importance,
            json.dumps(emotional_tag, ensure_ascii=False),
            json.dumps(entities, ensure_ascii=False),
            session_id, now, memory_type, parent_id,
        ))
        mem_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return mem_id
    
    def retrieve(
        self,
        user_id: str,
        query: str = "",
        top_k: int = 5,
        emotional_bias: Optional[np.ndarray] = None,
        recency_weight: float = 0.3,
        importance_weight: float = 0.3,
        relevance_weight: float = 0.3,
        emotional_weight: float = 0.1,
        memory_types: Optional[List[str]] = None,
        time_decay_hours: float = 720.0,  # 30天半衰期
    ) -> List[MemoryRecord]:
        """
        检索记忆，多维加权排序
        
        Args:
            user_id: 用户ID
            query: 查询文本（用于关键词匹配）
            top_k: 返回条数
            emotional_bias: 当前情感状态向量 [4]，用于mood-congruent检索
            recency_weight: 时间近因权重
            importance_weight: 重要性权重
            relevance_weight: 相关性权重
            emotional_weight: 情感匹配权重
            memory_types: 过滤记忆类型
            time_decay_hours: 时间衰减半衰期(小时)
        """
        conn = sqlite3.connect(self.db_path)
        
        sql = "SELECT * FROM memories WHERE user_id = ?"
        params = [user_id]
        
        if memory_types:
            placeholders = ",".join("?" * len(memory_types))
            sql += f" AND memory_type IN ({placeholders})"
            params.extend(memory_types)
        
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        
        if not rows:
            return []
        
        now = time.time()
        records = []
        scores = []
        
        for row in rows:
            rec = MemoryRecord(
                id=row[0], user_id=row[1], content=row[2], timestamp=row[3],
                importance=row[4],
                emotional_tag=json.loads(row[5]) if row[5] else {},
                entities=json.loads(row[6]) if row[6] else [],
                session_id=row[7], access_count=row[8], last_accessed=row[9],
                memory_type=row[10], parent_id=row[11],
            )
            records.append(rec)
            
            # === Scoring ===
            
            # 1. Recency: 指数衰减
            age_hours = (now - rec.timestamp) / 3600
            recency = np.exp(-0.693 * age_hours / time_decay_hours)  # 半衰期衰减
            
            # 2. Importance: 直接用
            imp = rec.importance
            
            # 3. Relevance: 关键词匹配（简单TF）
            relevance = 0.0
            if query:
                query_words = set(re.findall(r'[\w\u4e00-\u9fff]+', query.lower()))
                content_words = set(re.findall(r'[\w\u4e00-\u9fff]+', rec.content.lower()))
                if query_words:
                    overlap = len(query_words & content_words)
                    relevance = overlap / len(query_words)
            
            # 4. Emotional congruence: 余弦相似度
            emo_score = 0.0
            if emotional_bias is not None and rec.emotional_tag:
                tag_vec = np.array([
                    rec.emotional_tag.get('valence', 0.5),
                    rec.emotional_tag.get('arousal', 0.3),
                    rec.emotional_tag.get('dominance', 0.5),
                    rec.emotional_tag.get('stress', 0.3),
                ])
                # 余弦相似度
                norm = np.linalg.norm(emotional_bias) * np.linalg.norm(tag_vec)
                if norm > 0:
                    emo_score = float(np.dot(emotional_bias, tag_vec) / norm)
                    emo_score = (emo_score + 1) / 2  # 归一化到 [0, 1]
            
            # 加权总分
            score = (recency_weight * recency +
                     importance_weight * imp +
                     relevance_weight * relevance +
                     emotional_weight * emo_score)
            
            scores.append(score)
        
        # 排序取top_k
        indices = np.argsort(scores)[::-1][:top_k]
        results = [records[i] for i in indices]
        
        # 更新access_count
        if results:
            conn = sqlite3.connect(self.db_path)
            for rec in results:
                conn.execute(
                    "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                    (now, rec.id)
                )
            conn.commit()
            conn.close()
        
        return results
    
    def get_emotional_tag(self, mem_id: int) -> Optional[Dict]:
        """获取单条记忆的情感标签"""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT emotional_tag FROM memories WHERE id = ?", (mem_id,)).fetchone()
        conn.close()
        if row and row[0]:
            return json.loads(row[0])
        return None
    
    def get_recent(self, user_id: str, n: int = 10, memory_type: str = None) -> List[MemoryRecord]:
        """获取最近N条记忆"""
        conn = sqlite3.connect(self.db_path)
        sql = "SELECT * FROM memories WHERE user_id = ?"
        params = [user_id]
        if memory_type:
            sql += " AND memory_type = ?"
            params.append(memory_type)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(n)
        
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        
        return [MemoryRecord(
            id=row[0], user_id=row[1], content=row[2], timestamp=row[3],
            importance=row[4],
            emotional_tag=json.loads(row[5]) if row[5] else {},
            entities=json.loads(row[6]) if row[6] else [],
            session_id=row[7], access_count=row[8], last_accessed=row[9],
            memory_type=row[10], parent_id=row[11],
        ) for row in rows]
    
    def count(self, user_id: str = None, memory_type: str = None) -> int:
        """统计记忆数量"""
        conn = sqlite3.connect(self.db_path)
        sql = "SELECT COUNT(*) FROM memories WHERE 1=1"
        params = []
        if user_id:
            sql += " AND user_id = ?"
            params.append(user_id)
        if memory_type:
            sql += " AND memory_type = ?"
            params.append(memory_type)
        count = conn.execute(sql, params).fetchone()[0]
        conn.close()
        return count
    
    def decay(self, user_id: str, decay_factor: float = 0.995):
        """记忆重要性衰减——不常被检索的记忆逐渐淡化"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE memories 
            SET importance = importance * ? 
            WHERE user_id = ? AND memory_type = 'episodic'
        """, (decay_factor, user_id))
        conn.commit()
        conn.close()
    
    def _extract_entities(self, text: str) -> List[str]:
        """简单的实体提取（基于规则，后续可升级为NER）"""
        entities = []
        # 中文人名（2-4字，常见姓氏开头）
        zh_names = re.findall(r'(?:我|你|他|她|它)(?:的)?(?:朋友|同学|老师|爸|妈|哥|姐|弟|妹|猫|狗|宠物)?', text)
        entities.extend(zh_names)
        # 英文大写开头的词
        en_names = re.findall(r'\b[A-Z][a-z]+\b', text)
        entities.extend(en_names)
        return list(set(entities))
    
    def get_all_for_reflection(self, user_id: str, since_timestamp: float = 0) -> List[MemoryRecord]:
        """获取指定时间之后的所有episodic记忆（供reflection使用）"""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT * FROM memories 
            WHERE user_id = ? AND memory_type = 'episodic' AND timestamp > ?
            ORDER BY timestamp ASC
        """, (user_id, since_timestamp)).fetchall()
        conn.close()
        
        return [MemoryRecord(
            id=row[0], user_id=row[1], content=row[2], timestamp=row[3],
            importance=row[4],
            emotional_tag=json.loads(row[5]) if row[5] else {},
            entities=json.loads(row[6]) if row[6] else [],
            session_id=row[7], access_count=row[8], last_accessed=row[9],
            memory_type=row[10], parent_id=row[11],
        ) for row in rows]
    
    def store_reflection(self, user_id: str, content: str, importance: float,
                         emotional_tag: Dict, source_ids: List[int], session_id: str = "") -> int:
        """存储一条reflection记忆"""
        conn = sqlite3.connect(self.db_path)
        now = time.time()
        cursor = conn.execute("""
            INSERT INTO memories 
            (user_id, content, timestamp, importance, emotional_tag, entities, 
             session_id, access_count, last_accessed, memory_type, parent_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, 'reflection', NULL)
        """, (
            user_id, content, now, importance,
            json.dumps(emotional_tag, ensure_ascii=False),
            json.dumps([], ensure_ascii=False),
            session_id, now,
        ))
        ref_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return ref_id
