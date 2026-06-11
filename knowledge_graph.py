"""
knowledge_graph.py
==================
Knowledge Graph — 结构化实体关系图

从episodic记忆中抽取实体和关系，构建有类型的图。
支持多跳推理：(用户, works_at, Anthropic) + (Anthropic, builds, Claude) → (用户, related_to, Claude)

存储: SQLite
"""

import sqlite3
import json
import time
import re
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field


@dataclass
class Entity:
    id: Optional[int] = None
    name: str = ""
    entity_type: str = "unknown"  # person, place, pet, concept, event, object
    properties: Dict = field(default_factory=dict)
    first_seen: float = 0.0
    last_seen: float = 0.0
    mention_count: int = 0


@dataclass
class Relation:
    id: Optional[int] = None
    source_id: int = 0
    target_id: int = 0
    relation_type: str = "related_to"  # likes, dislikes, owns, works_at, knows, etc.
    properties: Dict = field(default_factory=dict)
    confidence: float = 1.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    source_memory_id: Optional[int] = None  # 来源记忆ID


class KnowledgeGraph:
    """
    SQLite-backed 知识图谱
    
    用法:
        kg = KnowledgeGraph("kg.db")
        e1 = kg.add_entity("小花", "pet", {"species": "猫", "age": 3})
        e2 = kg.add_entity("user1", "person")
        kg.add_relation(e2, e1, "owns")
        
        # 查询
        pets = kg.get_related(user1_id, "owns")  # → [小花]
        facts = kg.get_entity_facts(pet_id)  # → ["小花是一只3岁的猫"]
    """
    
    def __init__(self, db_path: str = "knowledge_graph.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                entity_type TEXT DEFAULT 'unknown',
                properties TEXT DEFAULT '{}',
                first_seen REAL,
                last_seen REAL,
                mention_count INTEGER DEFAULT 1,
                UNIQUE(name, entity_type)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                relation_type TEXT NOT NULL,
                properties TEXT DEFAULT '{}',
                confidence REAL DEFAULT 1.0,
                first_seen REAL,
                last_seen REAL,
                source_memory_id INTEGER,
                FOREIGN KEY (source_id) REFERENCES entities(id),
                FOREIGN KEY (target_id) REFERENCES entities(id),
                UNIQUE(source_id, target_id, relation_type)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_source ON relations(source_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_target ON relations(target_id)")
        conn.commit()
        conn.close()
    
    def add_entity(self, name: str, entity_type: str = "unknown", 
                   properties: Dict = None) -> int:
        """添加或更新实体"""
        now = time.time()
        conn = sqlite3.connect(self.db_path)
        
        # 尝试更新已有实体
        existing = conn.execute(
            "SELECT id, mention_count FROM entities WHERE name = ? AND entity_type = ?",
            (name, entity_type)
        ).fetchone()
        
        if existing:
            entity_id = existing[0]
            conn.execute("""
                UPDATE entities SET last_seen = ?, mention_count = mention_count + 1,
                properties = ? WHERE id = ?
            """, (now, json.dumps(properties or {}, ensure_ascii=False), entity_id))
        else:
            cursor = conn.execute("""
                INSERT INTO entities (name, entity_type, properties, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?)
            """, (name, entity_type, json.dumps(properties or {}, ensure_ascii=False), now, now))
            entity_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        return entity_id
    
    def add_relation(self, source_id: int, target_id: int, relation_type: str,
                     properties: Dict = None, confidence: float = 1.0,
                     source_memory_id: int = None) -> int:
        """添加或更新关系"""
        now = time.time()
        conn = sqlite3.connect(self.db_path)
        
        existing = conn.execute(
            "SELECT id FROM relations WHERE source_id = ? AND target_id = ? AND relation_type = ?",
            (source_id, target_id, relation_type)
        ).fetchone()
        
        if existing:
            rel_id = existing[0]
            conn.execute("""
                UPDATE relations SET last_seen = ?, confidence = MAX(confidence, ?),
                properties = ? WHERE id = ?
            """, (now, confidence, json.dumps(properties or {}, ensure_ascii=False), rel_id))
        else:
            cursor = conn.execute("""
                INSERT INTO relations (source_id, target_id, relation_type, properties, 
                                       confidence, first_seen, last_seen, source_memory_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (source_id, target_id, relation_type, 
                  json.dumps(properties or {}, ensure_ascii=False),
                  confidence, now, now, source_memory_id))
            rel_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        return rel_id
    
    def get_entity(self, name: str) -> Optional[Entity]:
        """按名称查找实体"""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT * FROM entities WHERE name = ?", (name,)
        ).fetchone()
        conn.close()
        if row:
            return Entity(id=row[0], name=row[1], entity_type=row[2],
                         properties=json.loads(row[3]) if row[3] else {},
                         first_seen=row[4], last_seen=row[5], mention_count=row[6])
        return None
    
    def get_related(self, entity_id: int, relation_type: str = None,
                    direction: str = "outgoing") -> List[Tuple[Entity, Relation]]:
        """获取与某实体相关的实体"""
        conn = sqlite3.connect(self.db_path)
        
        results = []
        if direction in ("outgoing", "both"):
            sql = """
                SELECT e.*, r.* FROM relations r
                JOIN entities e ON e.id = r.target_id
                WHERE r.source_id = ?
            """
            params = [entity_id]
            if relation_type:
                sql += " AND r.relation_type = ?"
                params.append(relation_type)
            
            for row in conn.execute(sql, params).fetchall():
                entity = Entity(id=row[0], name=row[1], entity_type=row[2],
                               properties=json.loads(row[3]) if row[3] else {},
                               first_seen=row[4], last_seen=row[5], mention_count=row[6])
                rel = Relation(id=row[7], source_id=row[8], target_id=row[9],
                              relation_type=row[10],
                              properties=json.loads(row[11]) if row[11] else {},
                              confidence=row[12], first_seen=row[13], last_seen=row[14],
                              source_memory_id=row[15])
                results.append((entity, rel))
        
        if direction in ("incoming", "both"):
            sql = """
                SELECT e.*, r.* FROM relations r
                JOIN entities e ON e.id = r.source_id
                WHERE r.target_id = ?
            """
            params = [entity_id]
            if relation_type:
                sql += " AND r.relation_type = ?"
                params.append(relation_type)
            
            for row in conn.execute(sql, params).fetchall():
                entity = Entity(id=row[0], name=row[1], entity_type=row[2],
                               properties=json.loads(row[3]) if row[3] else {},
                               first_seen=row[4], last_seen=row[5], mention_count=row[6])
                rel = Relation(id=row[7], source_id=row[8], target_id=row[9],
                              relation_type=row[10],
                              properties=json.loads(row[11]) if row[11] else {},
                              confidence=row[12], first_seen=row[13], last_seen=row[14],
                              source_memory_id=row[15])
                results.append((entity, rel))
        
        conn.close()
        return results
    
    def get_entity_facts(self, entity_id: int) -> List[str]:
        """生成关于一个实体的自然语言事实列表"""
        facts = []
        entity = self._get_entity_by_id(entity_id)
        if not entity:
            return facts
        
        props = entity.properties
        if props:
            prop_str = ", ".join(f"{k}={v}" for k, v in props.items())
            facts.append(f"{entity.name}的属性: {prop_str}")
        
        related = self.get_related(entity_id, direction="outgoing")
        for target, rel in related:
            facts.append(f"{entity.name} {rel.relation_type} {target.name}")
        
        related_in = self.get_related(entity_id, direction="incoming")
        for source, rel in related_in:
            facts.append(f"{source.name} {rel.relation_type} {entity.name}")
        
        return facts
    
    def multi_hop(self, start_id: int, hop_types: List[str]) -> List[List[Entity]]:
        """多跳推理"""
        paths = [[self._get_entity_by_id(start_id)]]
        
        for rel_type in hop_types:
            new_paths = []
            for path in paths:
                current = path[-1]
                if current is None or current.id is None:
                    continue
                related = self.get_related(current.id, rel_type)
                for target, _ in related:
                    new_paths.append(path + [target])
            paths = new_paths
        
        return [p for p in paths if len(p) == len(hop_types) + 1]
    
    def _get_entity_by_id(self, entity_id: int) -> Optional[Entity]:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
        conn.close()
        if row:
            return Entity(id=row[0], name=row[1], entity_type=row[2],
                         properties=json.loads(row[3]) if row[3] else {},
                         first_seen=row[4], last_seen=row[5], mention_count=row[6])
        return None
    
    def search_entities(self, query: str, limit: int = 10) -> List[Entity]:
        """模糊搜索实体"""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT * FROM entities WHERE name LIKE ? ORDER BY mention_count DESC LIMIT ?",
            (f"%{query}%", limit)
        ).fetchall()
        conn.close()
        return [Entity(id=r[0], name=r[1], entity_type=r[2],
                       properties=json.loads(r[3]) if r[3] else {},
                       first_seen=r[4], last_seen=r[5], mention_count=r[6])
                for r in rows]
    
    def extract_from_text(self, text: str) -> List[Tuple[str, str, Dict]]:
        """
        从文本中提取实体（规则方法，后续可升级为LLM NER）
        
        Returns: [(entity_name, entity_type, properties), ...]
        """
        entities = []
        
        # 中文称呼
        for match in re.finditer(r'(?:我的|你的|他的|她的)?(?:猫|狗|宠物)[\s]*[叫名叫]?[\s]*([^\s，。,.]{1,6})', text):
            entities.append((match.group(1), "pet", {}))
        
        # 人名（简单规则：引号或书名号内的）
        for match in re.finditer(r'[「"\'](.*?)[」"\'"]', text):
            name = match.group(1)
            if 1 < len(name) < 10:
                entities.append((name, "person", {}))
        
        # 喜欢/讨厌 + 事物
        for match in re.finditer(r'(?:喜欢|讨厌|爱|恨|害怕|享受)(.{1,10}?)(?:[，。,.]|$)', text):
            thing = match.group(1).strip()
            if thing:
                entities.append((thing, "concept", {}))
        
        # 工作/学校
        for match in re.finditer(r'(?:在|去)(.{2,15}?)(?:工作|上学|读书|上班)', text):
            entities.append((match.group(1).strip(), "organization", {}))
        
        return entities
    
    def extract_relations_from_text(self, text: str, user_id: str) -> List[Tuple[str, str, str]]:
        """
        从文本中提取关系（规则方法）
        
        Returns: [(subject, relation, object), ...]
        """
        relations = []
        
        # 喜欢/讨厌
        for match in re.finditer(r'(?:我|你|他|她)(?:很|非常|特别|超)?(喜欢|讨厌|爱|恨|害怕)(.{1,10}?)(?:[，。,.]|$)', text):
            rel = match.group(1)
            obj = match.group(2).strip()
            if obj:
                relations.append((user_id, rel, obj))
        
        # 拥有
        for match in re.finditer(r'(?:我的|你的|他的|她的)(猫|狗|宠物|车|房|电脑|手机)[\s]*[叫名叫]?[\s]*([^\s，。,.]{1,6})', text):
            obj_type = match.group(1)
            obj_name = match.group(2) if match.group(2) else obj_type
            relations.append((user_id, "拥有", obj_name))
        
        # 朋友
        for match in re.finditer(r'(?:我的|你的)(?:好朋友?|同学|同事|室友)[\s]*[叫名叫]?[\s]*([^\s，。,.]{1,6})', text):
            relations.append((user_id, "朋友", match.group(1)))
        
        return relations
    
    def get_all_facts(self, user_id: str) -> List[str]:
        """获取关于某用户的所有已知事实"""
        entity = self.get_entity(user_id)
        if not entity:
            return []
        return self.get_entity_facts(entity.id)
    
    def count(self) -> Tuple[int, int]:
        """返回 (实体数, 关系数)"""
        conn = sqlite3.connect(self.db_path)
        ec = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        rc = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        conn.close()
        return ec, rc
