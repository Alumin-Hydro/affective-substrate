"""
knowledge_graph.py
==================
Knowledge Graph — 结构化实体关系图 (v2)

改进:
  - entity_id: 全局唯一ID (ENT_XXXX)
  - version: 版本时间戳
  - status: verified / draft / suspected_broken / archived
  - confidence_score: 置信度
  - 别名解析支持

存储: SQLite
"""

import sqlite3
import json
import time
import re
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field

from episodic_memory import _next_entity_id


@dataclass
class Entity:
    id: Optional[int] = None
    entity_id: str = ""           # 全局唯一 ENT_XXXX
    name: str = ""
    entity_type: str = "unknown"
    properties: Dict = field(default_factory=dict)
    first_seen: float = 0.0
    last_seen: float = 0.0
    mention_count: int = 0
    version: str = ""
    status: str = "verified"      # verified / draft / suspected_broken / archived
    confidence_score: float = 1.0
    tags: List[str] = field(default_factory=list)


@dataclass
class Relation:
    id: Optional[int] = None
    relation_id: str = ""         # 全局唯一 REL_XXXX
    source_id: int = 0
    target_id: int = 0
    relation_type: str = "related_to"
    properties: Dict = field(default_factory=dict)
    confidence: float = 1.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    source_memory_id: Optional[int] = None
    version: str = ""
    status: str = "verified"


class KnowledgeGraph:
    """
    SQLite-backed 知识图谱 (v2)
    
    改进:
    - entity_id全局唯一
    - 别名解析: 别名词典 → 规范实体名
    - status/confidence治理
    """
    
    def __init__(self, db_path: str = "knowledge_graph.db", alias_dict=None):
        self.db_path = db_path
        self.alias_dict = alias_dict
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id TEXT UNIQUE,
                name TEXT NOT NULL,
                entity_type TEXT DEFAULT 'unknown',
                properties TEXT DEFAULT '{}',
                first_seen REAL,
                last_seen REAL,
                mention_count INTEGER DEFAULT 1,
                version TEXT DEFAULT '',
                status TEXT DEFAULT 'verified',
                confidence_score REAL DEFAULT 1.0,
                tags TEXT DEFAULT '[]',
                UNIQUE(name, entity_type)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                relation_id TEXT UNIQUE,
                source_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                relation_type TEXT NOT NULL,
                properties TEXT DEFAULT '{}',
                confidence REAL DEFAULT 1.0,
                first_seen REAL,
                last_seen REAL,
                source_memory_id INTEGER,
                version TEXT DEFAULT '',
                status TEXT DEFAULT 'verified',
                FOREIGN KEY (source_id) REFERENCES entities(id),
                FOREIGN KEY (target_id) REFERENCES entities(id),
                UNIQUE(source_id, target_id, relation_type)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_id ON entities(entity_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_source ON relations(source_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_target ON relations(target_id)")
        
        # 迁移旧表
        cursor = conn.execute("PRAGMA table_info(entities)")
        existing_cols = {row[1] for row in cursor.fetchall()}
        
        entity_migrations = {
            'entity_id': 'TEXT UNIQUE',
            'version': "TEXT DEFAULT ''",
            'status': "TEXT DEFAULT 'verified'",
            'confidence_score': "REAL DEFAULT 1.0",
            'tags': "TEXT DEFAULT '[]'",
        }
        for col, col_type in entity_migrations.items():
            if col not in existing_cols:
                try:
                    conn.execute(f"ALTER TABLE entities ADD COLUMN {col} {col_type}")
                except sqlite3.OperationalError:
                    pass
        
        cursor = conn.execute("PRAGMA table_info(relations)")
        existing_cols = {row[1] for row in cursor.fetchall()}
        
        rel_migrations = {
            'relation_id': 'TEXT UNIQUE',
            'version': "TEXT DEFAULT ''",
            'status': "TEXT DEFAULT 'verified'",
        }
        for col, col_type in rel_migrations.items():
            if col not in existing_cols:
                try:
                    conn.execute(f"ALTER TABLE relations ADD COLUMN {col} {col_type}")
                except sqlite3.OperationalError:
                    pass
        
        conn.commit()
        conn.close()
    
    def add_entity(self, name: str, entity_type: str = "unknown", 
                   properties: Dict = None, confidence_score: float = 1.0,
                   tags: List[str] = None) -> str:
        """添加或更新实体，返回entity_id"""
        # 别名解析
        if self.alias_dict:
            resolved = self.alias_dict.resolve(name)
            if resolved:
                name = resolved
        
        now = time.time()
        version = time.strftime("%Y-%m-%d", time.localtime(now))
        conn = sqlite3.connect(self.db_path)
        
        existing = conn.execute(
            "SELECT id, entity_id, mention_count FROM entities WHERE name = ? AND entity_type = ?",
            (name, entity_type)
        ).fetchone()
        
        if existing:
            entity_db_id = existing[0]
            entity_id = existing[1]
            if not entity_id:
                entity_id = _next_entity_id()
            conn.execute("""
                UPDATE entities SET last_seen = ?, mention_count = mention_count + 1,
                properties = ?, version = ?, confidence_score = MAX(confidence_score, ?)
                WHERE id = ?
            """, (now, json.dumps(properties or {}, ensure_ascii=False), version,
                  confidence_score, entity_db_id))
        else:
            entity_id = _next_entity_id()
            cursor = conn.execute("""
                INSERT INTO entities (entity_id, name, entity_type, properties, first_seen, last_seen,
                                      version, status, confidence_score, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'verified', ?, ?)
            """, (entity_id, name, entity_type,
                  json.dumps(properties or {}, ensure_ascii=False), now, now,
                  version, confidence_score,
                  json.dumps(tags or [], ensure_ascii=False)))
        
        conn.commit()
        conn.close()
        return entity_id
    
    def add_relation(self, source_id: int, target_id: int, relation_type: str,
                     properties: Dict = None, confidence: float = 1.0,
                     source_memory_id: int = None) -> str:
        """添加或更新关系，返回relation_id"""
        now = time.time()
        version = time.strftime("%Y-%m-%d", time.localtime(now))
        conn = sqlite3.connect(self.db_path)
        
        existing = conn.execute(
            "SELECT id, relation_id FROM relations WHERE source_id = ? AND target_id = ? AND relation_type = ?",
            (source_id, target_id, relation_type)
        ).fetchone()
        
        if existing:
            rel_db_id = existing[0]
            relation_id = existing[1]
            if not relation_id:
                relation_id = f"REL_{_next_entity_id()[4:]}"  # 复用ID生成器
            conn.execute("""
                UPDATE relations SET last_seen = ?, confidence = MAX(confidence, ?),
                properties = ?, version = ? WHERE id = ?
            """, (now, confidence, json.dumps(properties or {}, ensure_ascii=False), version, rel_db_id))
        else:
            relation_id = f"REL_{_next_entity_id()[4:]}"
            conn.execute("""
                INSERT INTO relations (relation_id, source_id, target_id, relation_type, properties, 
                                       confidence, first_seen, last_seen, source_memory_id, version, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'verified')
            """, (relation_id, source_id, target_id, relation_type, 
                  json.dumps(properties or {}, ensure_ascii=False),
                  confidence, now, now, source_memory_id, version))
        
        conn.commit()
        conn.close()
        return relation_id
    
    def get_entity(self, name: str) -> Optional[Entity]:
        """按名称查找实体（含别名解析）"""
        if self.alias_dict:
            resolved = self.alias_dict.resolve(name)
            if resolved:
                name = resolved
        
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT * FROM entities WHERE name = ?", (name,)).fetchone()
        conn.close()
        if row:
            return self._row_to_entity(row)
        return None
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """按entity_id精确查找"""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT * FROM entities WHERE entity_id = ?", (entity_id,)).fetchone()
        conn.close()
        if row:
            return self._row_to_entity(row)
        return None
    
    def get_related(self, entity_db_id: int, relation_type: str = None,
                    direction: str = "outgoing") -> List[Tuple[Entity, Relation]]:
        """获取与某实体相关的实体"""
        conn = sqlite3.connect(self.db_path)
        results = []
        
        if direction in ("outgoing", "both"):
            sql = """
                SELECT e.*, r.* FROM relations r
                JOIN entities e ON e.id = r.target_id
                WHERE r.source_id = ? AND r.status = 'verified'
            """
            params = [entity_db_id]
            if relation_type:
                sql += " AND r.relation_type = ?"
                params.append(relation_type)
            for row in conn.execute(sql, params).fetchall():
                entity = self._row_to_entity(row[:12])
                rel = self._row_to_relation(row[12:])
                results.append((entity, rel))
        
        if direction in ("incoming", "both"):
            sql = """
                SELECT e.*, r.* FROM relations r
                JOIN entities e ON e.id = r.source_id
                WHERE r.target_id = ? AND r.status = 'verified'
            """
            params = [entity_db_id]
            if relation_type:
                sql += " AND r.relation_type = ?"
                params.append(relation_type)
            for row in conn.execute(sql, params).fetchall():
                entity = self._row_to_entity(row[:12])
                rel = self._row_to_relation(row[12:])
                results.append((entity, rel))
        
        conn.close()
        return results
    
    def get_entity_facts(self, entity_db_id: int) -> List[str]:
        """生成关于一个实体的自然语言事实列表"""
        facts = []
        entity = self._get_entity_by_db_id(entity_db_id)
        if not entity:
            return facts
        
        props = entity.properties
        if props:
            prop_str = ", ".join(f"{k}={v}" for k, v in props.items())
            facts.append(f"{entity.name}的属性: {prop_str}")
        
        related = self.get_related(entity_db_id, direction="outgoing")
        for target, rel in related:
            facts.append(f"{entity.name} {rel.relation_type} {target.name}")
        
        related_in = self.get_related(entity_db_id, direction="incoming")
        for source, rel in related_in:
            facts.append(f"{source.name} {rel.relation_type} {entity.name}")
        
        return facts
    
    def multi_hop(self, start_db_id: int, hop_types: List[str]) -> List[List[Entity]]:
        """多跳推理"""
        paths = [[self._get_entity_by_db_id(start_db_id)]]
        
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
    
    def _get_entity_by_db_id(self, db_id: int) -> Optional[Entity]:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT * FROM entities WHERE id = ?", (db_id,)).fetchone()
        conn.close()
        if row:
            return self._row_to_entity(row)
        return None
    
    def search_entities(self, query: str, limit: int = 10) -> List[Entity]:
        """模糊搜索实体（含别名扩展）"""
        conn = sqlite3.connect(self.db_path)
        
        # 别名扩展
        candidates = [query]
        if self.alias_dict:
            expanded = self.alias_dict.get_candidates(query)
            candidates.extend(expanded)
        
        results = []
        seen_ids = set()
        for candidate in candidates:
            rows = conn.execute(
                "SELECT * FROM entities WHERE name LIKE ? AND status = 'verified' ORDER BY mention_count DESC LIMIT ?",
                (f"%{candidate}%", limit)
            ).fetchall()
            for row in rows:
                if row[0] not in seen_ids:
                    results.append(self._row_to_entity(row))
                    seen_ids.add(row[0])
        
        conn.close()
        return results[:limit]
    
    def extract_from_text(self, text: str) -> List[Tuple[str, str, Dict]]:
        """从文本中提取实体"""
        entities = []
        
        for match in re.finditer(r'(?:我的|你的|他的|她的)?(?:猫|狗|宠物)[\s]*[叫名叫]?[\s]*([^\s，。,.]{1,6})', text):
            entities.append((match.group(1), "pet", {}))
        
        for match in re.finditer(r'[\u201c\u201d\u2018\u2019\"\x27](.*?)[\u201c\u201d\u2018\u2019\"\x27]', text):
            name = match.group(1)
            if 1 < len(name) < 10:
                entities.append((name, "person", {}))
        
        for match in re.finditer(r'(?:喜欢|讨厌|爱|恨|害怕|享受)(.{1,10}?)(?:[，。,.]|$)', text):
            thing = match.group(1).strip()
            if thing:
                entities.append((thing, "concept", {}))
        
        for match in re.finditer(r'(?:在|去)(.{2,15}?)(?:工作|上学|读书|上班)', text):
            entities.append((match.group(1).strip(), "organization", {}))
        
        return entities
    
    def extract_relations_from_text(self, text: str, user_id: str) -> List[Tuple[str, str, str]]:
        """从文本中提取关系"""
        relations = []
        
        for match in re.finditer(r'(?:我|你|他|她)(?:很|非常|特别|超)?(喜欢|讨厌|爱|恨|害怕)(.{1,10}?)(?:[，。,.]|$)', text):
            rel = match.group(1)
            obj = match.group(2).strip()
            if obj:
                relations.append((user_id, rel, obj))
        
        for match in re.finditer(r'(?:我的|你的|他的|她的)(猫|狗|宠物|车|房|电脑|手机)[\s]*[叫名叫]?[\s]*([^\s，。,.]{1,6})', text):
            obj_type = match.group(1)
            obj_name = match.group(2) if match.group(2) else obj_type
            relations.append((user_id, "拥有", obj_name))
        
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
        ec = conn.execute("SELECT COUNT(*) FROM entities WHERE status = 'verified'").fetchone()[0]
        rc = conn.execute("SELECT COUNT(*) FROM relations WHERE status = 'verified'").fetchone()[0]
        conn.close()
        return ec, rc
    
    def update_status(self, entity_id: str, status: str, confidence_score: float = None):
        """更新实体状态"""
        conn = sqlite3.connect(self.db_path)
        if confidence_score is not None:
            conn.execute("UPDATE entities SET status = ?, confidence_score = ? WHERE entity_id = ?",
                         (status, confidence_score, entity_id))
        else:
            conn.execute("UPDATE entities SET status = ? WHERE entity_id = ?", (status, entity_id))
        conn.commit()
        conn.close()
    
    def get_stats(self) -> Dict:
        """获取图谱统计"""
        conn = sqlite3.connect(self.db_path)
        total_entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        verified_entities = conn.execute("SELECT COUNT(*) FROM entities WHERE status = 'verified'").fetchone()[0]
        total_relations = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        verified_relations = conn.execute("SELECT COUNT(*) FROM relations WHERE status = 'verified'").fetchone()[0]
        conn.close()
        return {
            'entities': {'total': total_entities, 'verified': verified_entities},
            'relations': {'total': total_relations, 'verified': verified_relations},
        }
    
    def _row_to_entity(self, row) -> Entity:
        return Entity(
            id=row[0],
            entity_id=row[1] if len(row) > 1 and row[1] else "",
            name=row[2] if len(row) > 2 else "",
            entity_type=row[3] if len(row) > 3 else "unknown",
            properties=json.loads(row[4]) if len(row) > 4 and row[4] else {},
            first_seen=row[5] if len(row) > 5 else 0,
            last_seen=row[6] if len(row) > 6 else 0,
            mention_count=row[7] if len(row) > 7 else 0,
            version=row[8] if len(row) > 8 and row[8] else "",
            status=row[9] if len(row) > 9 and row[9] else "verified",
            confidence_score=row[10] if len(row) > 10 and row[10] else 1.0,
            tags=json.loads(row[11]) if len(row) > 11 and row[11] else [],
        )
    
    def _row_to_relation(self, row) -> Relation:
        return Relation(
            id=row[0],
            relation_id=row[1] if len(row) > 1 and row[1] else "",
            source_id=row[2] if len(row) > 2 else 0,
            target_id=row[3] if len(row) > 3 else 0,
            relation_type=row[4] if len(row) > 4 else "related_to",
            properties=json.loads(row[5]) if len(row) > 5 and row[5] else {},
            confidence=row[6] if len(row) > 6 else 1.0,
            first_seen=row[7] if len(row) > 7 else 0,
            last_seen=row[8] if len(row) > 8 else 0,
            source_memory_id=row[9] if len(row) > 9 else None,
            version=row[10] if len(row) > 10 and row[10] else "",
            status=row[11] if len(row) > 11 and row[11] else "verified",
        )
