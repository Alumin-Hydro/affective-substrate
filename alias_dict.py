"""
alias_dict.py
=============
别名词典 — 解决口语查询与规范实体之间的映射

用法:
    alias = AliasDict("data/alias_dict.json")
    canonical = alias.resolve("早报")  # → "arXiv Morning Brief"
    candidates = alias.get_candidates("吃内存")  # → ["内存溢出", "swap 风险"]
"""

import json
import os
import re
from typing import List, Optional, Dict


class AliasDict:
    """
    别名词典
    
    格式:
    {
        "早报": {"canonical": "arXiv Morning Brief", "type": "project"},
        "学姐": {"canonical": "Cherry", "type": "person"},
        "吃内存": {"canonical": "内存溢出", "alternates": ["swap 风险", "OOM"], "type": "concept"},
    }
    """
    
    def __init__(self, dict_path: str = "data/alias_dict.json"):
        self.dict_path = dict_path
        self.aliases: Dict[str, Dict] = {}
        self._load()
    
    def _load(self):
        if os.path.exists(self.dict_path):
            with open(self.dict_path, 'r', encoding='utf-8') as f:
                self.aliases = json.load(f)
    
    def _save(self):
        os.makedirs(os.path.dirname(self.dict_path), exist_ok=True)
        with open(self.dict_path, 'w', encoding='utf-8') as f:
            json.dump(self.aliases, f, ensure_ascii=False, indent=2)
    
    def add(self, alias: str, canonical: str, entity_type: str = "concept",
            alternates: List[str] = None):
        """添加别名映射"""
        self.aliases[alias.lower()] = {
            "canonical": canonical,
            "type": entity_type,
            "alternates": alternates or [],
        }
        self._save()
    
    def resolve(self, query: str) -> Optional[str]:
        """将别名解析为规范名称。如果没有匹配，返回None"""
        query_lower = query.lower().strip()
        
        # 精确匹配
        if query_lower in self.aliases:
            return self.aliases[query_lower]["canonical"]
        
        # 子串匹配（query包含别名）
        for alias, entry in self.aliases.items():
            if alias in query_lower:
                return entry["canonical"]
        
        return None
    
    def get_candidates(self, query: str) -> List[str]:
        """获取查询的所有候选（canonical + alternates）"""
        query_lower = query.lower().strip()
        candidates = []
        
        for alias, entry in self.aliases.items():
            if alias in query_lower or query_lower in alias:
                candidates.append(entry["canonical"])
                candidates.extend(entry.get("alternates", []))
        
        return list(set(candidates))
    
    def expand_query(self, query: str) -> str:
        """
        扩展查询：将别名替换为规范名+候选拼接
        用于检索前的查询扩展
        """
        expanded = query
        query_lower = query.lower()
        
        for alias, entry in self.aliases.items():
            if alias in query_lower:
                canonical = entry["canonical"]
                alternates = entry.get("alternates", [])
                # 在查询后追加规范名和候选拼接
                expansion = " ".join([canonical] + alternates)
                if canonical not in expanded:
                    expanded = f"{expanded} {expansion}"
        
        return expanded
    
    def all_aliases(self) -> Dict[str, Dict]:
        """返回所有别名"""
        return dict(self.aliases)
    
    def remove(self, alias: str) -> bool:
        """删除别名"""
        alias_lower = alias.lower().strip()
        if alias_lower in self.aliases:
            del self.aliases[alias_lower]
            self._save()
            return True
        return False
