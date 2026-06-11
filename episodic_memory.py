"""
episodic_memory.py
==================
Episodic Stream — 情景记忆流 (v2)

改进:
  - entity_id: 全局唯一ID (ENT_XXXX)
  - status: verified / draft / suspected_broken / archived
  - confidence_score: 置信度 (0-1)
  - tags: 标签列表
  - related_entities: 关联实体ID列表
  - IDF-aware关键词检索
  - 别名查询扩展
  - 可选embedding检索

存储: SQLite
检索: recency × importance × relevance(IDF) × emotional_congruence × embedding
"""

import sqlite3
import json
import time
import os
import re
import math
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
import numpy as np


# === Entity ID 生成器 ===

_entity_counter_path = os.path.join(os.path.dirname(__file__), "data", ".entity_counter")


def _next_entity_id() -> str:
    """生成全局唯一实体ID: ENT_0001, ENT_0002, ..."""
    counter = 0
    if os.path.exists(_entity_counter_path):
        with open(_entity_counter_path, 'r') as f:
            counter = int(f.read().strip())
    counter += 1
    os.makedirs(os.path.dirname(_entity_counter_path), exist_ok=True)
    with open(_entity_counter_path, 'w') as f:
        f.write(str(counter))
    return f"ENT_{counter:04d}"


@dataclass
class MemoryRecord:
    """单条情景记忆 (v2)"""
    id: Optional[int] = None
    entity_id: str = ""           # 全局唯一ID ENT_XXXX
    user_id: str = ""
    content: str = ""
    timestamp: float = 0.0
    importance: float = 0.5
    confidence_score: float = 1.0  # 置信度
    status: str = "verified"       # verified / draft / suspected_broken / archived
    emotional_tag: Dict = field(default_factory=lambda: {"valence": 0.5, "arousal": 0.3, "dominance": 0.5, "stress": 0.3})
    entities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    related_entities: List[str] = field(default_factory=list)  # 关联entity_id列表
    session_id: str = ""
    access_count: int = 0
    last_accessed: float = 0.0
    memory_type: str = "episodic"  # episodic / reflection / event
    parent_id: Optional[int] = None
    version: str = ""              # 版本时间戳

    def to_frontmatter(self) -> str:
        """导出为YAML frontmatter格式"""
        lines = [
            "---",
            f'title: "{self.content[:60]}"',
            f'entity_id: "{self.entity_id}"',
            f'version: "{self.version}"',
            f'status: "{self.status}"',
            f'confidence_score: {self.confidence_score}',
            f'tags: {json.dumps(self.tags, ensure_ascii=False)}',
            f'related_entities: {json.dumps(self.related_entities, ensure_ascii=False)}',
            "---",
        ]
        return "\n".join(lines)


class EpisodicMemory:
    """
    SQLite-backed 情景记忆流 (v2)
    
    改进:
    - entity_id全局唯一
    - IDF-aware关键词检索
    - 别名查询扩展
    - status/confidence治理
    """
    
    def __init__(self, db_path: str = "episodic_memory.db", alias_dict=None):
        self.db_path = db_path
        self.alias_dict = alias_dict  # AliasDict实例
        self._idf_cache: Dict[str, float] = {}  # IDF缓存
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id TEXT UNIQUE NOT NULL,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                importance REAL DEFAULT 0.5,
                confidence_score REAL DEFAULT 1.0,
                status TEXT DEFAULT 'verified',
                emotional_tag TEXT DEFAULT '{}',
                entities TEXT DEFAULT '[]',
                tags TEXT DEFAULT '[]',
                related_entities TEXT DEFAULT '[]',
                session_id TEXT DEFAULT '',
                access_count INTEGER DEFAULT 0,
                last_accessed REAL DEFAULT 0,
                memory_type TEXT DEFAULT 'episodic',
                parent_id INTEGER DEFAULT NULL,
                version TEXT DEFAULT ''
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user ON memories(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON memories(memory_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_id ON memories(entity_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON memories(status)")
        
        # 迁移旧表：如果缺列则添加
        cursor = conn.execute("PRAGMA table_info(memories)")
        existing_cols = {row[1] for row in cursor.fetchall()}
        
        migrations = {
            'entity_id': 'TEXT UNIQUE',
            'confidence_score': 'REAL DEFAULT 1.0',
            'status': "TEXT DEFAULT 'verified'",
            'tags': "TEXT DEFAULT '[]'",
            'related_entities': "TEXT DEFAULT '[]'",
            'version': "TEXT DEFAULT ''",
        }
        for col, col_type in migrations.items():
            if col not in existing_cols:
                try:
                    conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {col_type}")
                except sqlite3.OperationalError:
                    pass  # 已存在
        
        conn.commit()
        conn.close()
    
    def _compute_idf(self, conn):
        """计算全局IDF（逆文档频率）"""
        if self._idf_cache:
            return
        
        rows = conn.execute("SELECT content FROM memories").fetchall()
        if not rows:
            return
        
        doc_count = len(rows)
        word_doc_freq: Dict[str, int] = {}
        
        for (content,) in rows:
            words = set(re.findall(r'[\w\u4e00-\u9fff]+', content.lower()))
            for w in words:
                word_doc_freq[w] = word_doc_freq.get(w, 0) + 1
        
        for word, freq in word_doc_freq.items():
            self._idf_cache[word] = math.log((doc_count + 1) / (freq + 1)) + 1  # 平滑IDF
    
    def _invalidate_idf_cache(self):
        self._idf_cache.clear()
    
    def store(
        self,
        user_id: str,
        content: str,
        importance: float = 0.5,
        confidence_score: float = 1.0,
        status: str = "verified",
        emotional_tag: Optional[Dict] = None,
        entities: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        related_entities: Optional[List[str]] = None,
        session_id: str = "",
        memory_type: str = "episodic",
        parent_id: Optional[int] = None,
    ) -> str:
        """存储一条记忆，返回entity_id"""
        if emotional_tag is None:
            emotional_tag = {"valence": 0.5, "arousal": 0.3, "dominance": 0.5, "stress": 0.3}
        if entities is None:
            entities = self._extract_entities(content)
        if tags is None:
            tags = []
        if related_entities is None:
            related_entities = []
        
        entity_id = _next_entity_id()
        now = time.time()
        version = time.strftime("%Y-%m-%d", time.localtime(now))
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            INSERT INTO memories 
            (entity_id, user_id, content, timestamp, importance, confidence_score, status,
             emotional_tag, entities, tags, related_entities,
             session_id, access_count, last_accessed, memory_type, parent_id, version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
        """, (
            entity_id, user_id, content, now, importance, confidence_score, status,
            json.dumps(emotional_tag, ensure_ascii=False),
            json.dumps(entities, ensure_ascii=False),
            json.dumps(tags, ensure_ascii=False),
            json.dumps(related_entities, ensure_ascii=False),
            session_id, now, memory_type, parent_id, version,
        ))
        conn.commit()
        conn.close()
        
        self._invalidate_idf_cache()
        return entity_id
    
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
        time_decay_hours: float = 720.0,
        status_filter: Optional[str] = "verified",
    ) -> List[MemoryRecord]:
        """
        检索记忆，IDF-aware多维加权排序
        
        改进:
        - IDF-aware关键词: 高频词降权，低频词升权
        - 别名查询扩展: 查询中的别名自动扩展为规范名+候选拼接
        - status过滤: 默认只检索verified状态
        """
        conn = sqlite3.connect(self.db_path)
        
        # 计算IDF
        self._compute_idf(conn)
        
        sql = "SELECT * FROM memories WHERE user_id = ?"
        params = [user_id]
        
        if status_filter:
            sql += " AND status = ?"
            params.append(status_filter)
        
        if memory_types:
            placeholders = ",".join("?" * len(memory_types))
            sql += f" AND memory_type IN ({placeholders})"
            params.extend(memory_types)
        
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        
        if not rows:
            return []
        
        # 查询扩展（别名）
        expanded_query = query
        if self.alias_dict and query:
            expanded_query = self.alias_dict.expand_query(query)
        
        now = time.time()
        records = []
        scores = []
        
        for row in rows:
            rec = MemoryRecord(
                id=row[0], entity_id=row[1], user_id=row[2], content=row[3],
                timestamp=row[4], importance=row[5],
                confidence_score=row[6] if len(row) > 6 else 1.0,
                status=row[7] if len(row) > 7 else "verified",
                emotional_tag=json.loads(row[8]) if row[8] else {},
                entities=json.loads(row[9]) if row[9] else [],
                tags=json.loads(row[10]) if len(row) > 10 and row[10] else [],
                related_entities=json.loads(row[11]) if len(row) > 11 and row[11] else [],
                session_id=row[12] if len(row) > 12 else "",
                access_count=row[13] if len(row) > 13 else 0,
                last_accessed=row[14] if len(row) > 14 else 0,
                memory_type=row[15] if len(row) > 15 else "episodic",
                parent_id=row[16] if len(row) > 16 else None,
                version=row[17] if len(row) > 17 else "",
            )
            records.append(rec)
            
            # === Scoring ===
            
            # 1. Recency: 指数衰减
            age_hours = (now - rec.timestamp) / 3600
            recency = np.exp(-0.693 * age_hours / time_decay_hours)
            
            # 2. Importance: 直接用 × confidence
            imp = rec.importance * rec.confidence_score
            
            # 3. Relevance: IDF-aware关键词匹配
            relevance = 0.0
            if expanded_query:
                query_words = re.findall(r'[\w\u4e00-\u9fff]+', expanded_query.lower())
                content_words = re.findall(r'[\w\u4e00-\u9fff]+', rec.content.lower())
                content_word_set = set(content_words)
                
                if query_words:
                    idf_weighted_overlap = 0.0
                    total_idf = 0.0
                    for w in query_words:
                        idf = self._idf_cache.get(w, 1.0)  # 未知词IDF=1
                        total_idf += idf
                        if w in content_word_set:
                            idf_weighted_overlap += idf
                    
                    if total_idf > 0:
                        relevance = idf_weighted_overlap / total_idf
            
            # 4. Emotional congruence: 余弦相似度
            emo_score = 0.0
            if emotional_bias is not None and rec.emotional_tag:
                tag_vec = np.array([
                    rec.emotional_tag.get('valence', 0.5),
                    rec.emotional_tag.get('arousal', 0.3),
                    rec.emotional_tag.get('dominance', 0.5),
                    rec.emotional_tag.get('stress', 0.3),
                ])
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
    
    def get_by_entity_id(self, entity_id: str) -> Optional[MemoryRecord]:
        """按entity_id精确查询"""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT * FROM memories WHERE entity_id = ?", (entity_id,)).fetchone()
        conn.close()
        if row:
            return self._row_to_record(row)
        return None
    
    def update_status(self, entity_id: str, status: str, confidence_score: float = None):
        """更新记忆状态（治理用）"""
        conn = sqlite3.connect(self.db_path)
        if confidence_score is not None:
            conn.execute("UPDATE memories SET status = ?, confidence_score = ? WHERE entity_id = ?",
                         (status, confidence_score, entity_id))
        else:
            conn.execute("UPDATE memories SET status = ? WHERE entity_id = ?", (status, entity_id))
        conn.commit()
        conn.close()
    
    def archive_stale(self, user_id: str, max_age_days: int = 30, min_importance: float = 0.3):
        """归档过期低重要性记忆"""
        cutoff = time.time() - max_age_days * 86400
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE memories SET status = 'archived'
            WHERE user_id = ? AND status = 'verified' 
            AND timestamp < ? AND importance < ?
            AND memory_type = 'episodic'
        """, (user_id, cutoff, min_importance))
        count = conn.total_changes
        conn.commit()
        conn.close()
        return count
    
    def _row_to_record(self, row) -> MemoryRecord:
        """数据库行 → MemoryRecord"""
        return MemoryRecord(
            id=row[0], entity_id=row[1], user_id=row[2], content=row[3],
            timestamp=row[4], importance=row[5],
            confidence_score=row[6] if len(row) > 6 else 1.0,
            status=row[7] if len(row) > 7 else "verified",
            emotional_tag=json.loads(row[8]) if row[8] else {},
            entities=json.loads(row[9]) if row[9] else [],
            tags=json.loads(row[10]) if len(row) > 10 and row[10] else [],
            related_entities=json.loads(row[11]) if len(row) > 11 and row[11] else [],
            session_id=row[12] if len(row) > 12 else "",
            access_count=row[13] if len(row) > 13 else 0,
            last_accessed=row[14] if len(row) > 14 else 0,
            memory_type=row[15] if len(row) > 15 else "episodic",
            parent_id=row[16] if len(row) > 16 else None,
            version=row[17] if len(row) > 17 else "",
        )
    
    def get_emotional_tag(self, entity_id: str) -> Optional[Dict]:
        """获取单条记忆的情感标签"""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT emotional_tag FROM memories WHERE entity_id = ?", (entity_id,)).fetchone()
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
        return [self._row_to_record(row) for row in rows]
    
    def count(self, user_id: str = None, memory_type: str = None, status: str = None) -> int:
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
        if status:
            sql += " AND status = ?"
            params.append(status)
        count = conn.execute(sql, params).fetchone()[0]
        conn.close()
        return count
    
    def decay(self, user_id: str, decay_factor: float = 0.995):
        """记忆重要性衰减——不常被检索的记忆逐渐淡化"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE memories 
            SET importance = importance * ? 
            WHERE user_id = ? AND memory_type = 'episodic' AND status = 'verified'
        """, (decay_factor, user_id))
        conn.commit()
        conn.close()
    
    def _extract_entities(self, text: str) -> List[str]:
        """简单的实体提取（规则方法）"""
        entities = []
        zh_names = re.findall(r'(?:我|你|他|她|它)(?:的)?(?:朋友|同学|老师|爸|妈|哥|姐|弟|妹|猫|狗|宠物)?', text)
        entities.extend(zh_names)
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
        return [self._row_to_record(row) for row in rows]
    
    def store_reflection(self, user_id: str, content: str, importance: float,
                         emotional_tag: Dict, source_ids: List[int], session_id: str = "") -> str:
        """存储一条reflection记忆"""
        return self.store(
            user_id=user_id,
            content=content,
            importance=importance,
            emotional_tag=emotional_tag,
            entities=[],
            tags=["reflection"],
            related_entities=[],
            session_id=session_id,
            memory_type="reflection",
            parent_id=None,
        )
    
    def get_stats(self, user_id: str = None) -> Dict:
        """获取记忆统计（治理用）"""
        conn = sqlite3.connect(self.db_path)
        
        sql_base = "SELECT COUNT(*) FROM memories WHERE 1=1"
        params_base = []
        if user_id:
            sql_base += " AND user_id = ?"
            params_base.append(user_id)
        
        total = conn.execute(sql_base, params_base).fetchone()[0]
        
        verified = conn.execute(sql_base + " AND status = 'verified'", params_base).fetchone()[0]
        archived = conn.execute(sql_base + " AND status = 'archived'", params_base).fetchone()[0]
        
        type_counts = {}
        for mtype in ['episodic', 'reflection', 'event']:
            count = conn.execute(sql_base + " AND memory_type = ?", params_base + [mtype]).fetchone()[0]
            type_counts[mtype] = count
        
        conn.close()
        
        return {
            'total': total,
            'verified': verified,
            'archived': archived,
            'by_type': type_counts,
        }
