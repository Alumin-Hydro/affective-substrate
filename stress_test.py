"""
stress_test.py
==============
检索质量压力测试

每次改动后运行，必须全部PASS。
参考OpenClaw的23用例设计。

用法:
    python stress_test.py
    python stress_test.py --verbose
"""

import os
import sys
import time
import json
import tempfile
import shutil

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from episodic_memory import EpisodicMemory
from knowledge_graph import KnowledgeGraph
from alias_dict import AliasDict
from vad_analyzer import VADAnalyzer


class StressTest:
    """检索质量压力测试"""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        self.errors = []
        
        # 临时目录
        self.tmp_dir = tempfile.mkdtemp(prefix="affective_stress_")
        self.db_path = os.path.join(self.tmp_dir, "test.db")
        self.kg_path = os.path.join(self.tmp_dir, "test_kg.db")
        self.alias_path = os.path.join(self.tmp_dir, "alias.json")
        
        # 初始化
        self.alias = AliasDict(self.alias_path)
        self.memory = EpisodicMemory(self.db_path, alias_dict=self.alias)
        self.kg = KnowledgeGraph(self.kg_path, alias_dict=self.alias)
        self.vad = VADAnalyzer()
        
        # 填充测试数据
        self._seed_data()
    
    def cleanup(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
    
    def _seed_data(self):
        """填充测试数据"""
        import time as _time
        
        # 情景记忆
        self.memory.store(user_id="test_user", content="用户提到他的猫小花去世了，非常伤心",
                          importance=0.9, tags=["pet", "loss"],
                          emotional_tag={"valence": 0.2, "arousal": 0.6, "dominance": 0.3, "stress": 0.8})
        
        _time.sleep(0.01)
        self.memory.store(user_id="test_user", content="用户说期中考试考得很好，很开心",
                          importance=0.7, tags=["exam", "positive"],
                          emotional_tag={"valence": 0.9, "arousal": 0.7, "dominance": 0.7, "stress": 0.2})
        
        _time.sleep(0.01)
        self.memory.store(user_id="test_user", content="用户提到喜欢物理，尤其是量子力学",
                          importance=0.6, tags=["physics", "interest"],
                          emotional_tag={"valence": 0.8, "arousal": 0.5, "dominance": 0.6, "stress": 0.2})
        
        _time.sleep(0.01)
        self.memory.store(user_id="test_user", content="用户说最近压力很大，作业太多了",
                          importance=0.5, tags=["stress", "school"],
                          emotional_tag={"valence": 0.3, "arousal": 0.6, "dominance": 0.3, "stress": 0.8})
        
        _time.sleep(0.01)
        self.memory.store(user_id="test_user", content="用户提到朋友小李生日快到了",
                          importance=0.4, tags=["friend", "birthday"],
                          emotional_tag={"valence": 0.7, "arousal": 0.4, "dominance": 0.5, "stress": 0.3})
        
        # 知识图谱
        user_ent = self.kg.get_entity("test_user") or self.kg._row_to_entity(
            (None, None, "test_user", "person", "{}", 0, 0, 1, "", "verified", 1.0, "[]"))
        # 先添加实体获取DB id
        self.kg.add_entity("test_user", "person")
        self.kg.add_entity("小花", "pet", {"species": "猫"})
        self.kg.add_entity("物理", "concept")
        self.kg.add_entity("小李", "person")
        
        # 用DB id添加关系
        user_row = self.kg.get_entity("test_user")
        cat_row = self.kg.get_entity("小花")
        physics_row = self.kg.get_entity("物理")
        friend_row = self.kg.get_entity("小李")
        
        if user_row and cat_row:
            self.kg.add_relation(user_row.id, cat_row.id, "拥有")
        if user_row and physics_row:
            self.kg.add_relation(user_row.id, physics_row.id, "喜欢")
        if user_row and friend_row:
            self.kg.add_relation(user_row.id, friend_row.id, "朋友")
        
        # 别名
        self.alias.add("考试", "测验", "event", ["测试", "期中", "期末"])
        self.alias.add("学姐", "Cherry", "person")
        self.alias.add("猫", "宠物", "concept", ["猫猫", "喵"])
    
    def _check(self, name: str, condition: bool, detail: str = ""):
        if condition:
            self.passed += 1
            if self.verbose:
                print(f"  ✅ {name}")
        else:
            self.failed += 1
            self.errors.append(f"{name}: {detail}")
            print(f"  ❌ {name}: {detail}")
    
    def run_all(self):
        print("=" * 60)
        print("Affective Substrate — 检索质量压力测试")
        print("=" * 60)
        
        self.test_basic_retrieval()
        self.test_alias_resolution()
        self.test_idf_scoring()
        self.test_emotional_congruence()
        self.test_status_filtering()
        self.test_entity_id_system()
        self.test_knowledge_graph()
        self.test_multi_hop()
        self.test_confidence_decay()
        self.test_archive_stale()
        self.test_frontmatter()
        self.test_vad_analyzer()
        self.test_reflection_storage()
        self.test_stats()
        
        print("=" * 60)
        total = self.passed + self.failed
        print(f"结果: {self.passed}/{total} PASS, {self.failed} FAIL")
        if self.errors:
            print("\n失败用例:")
            for e in self.errors:
                print(f"  - {e}")
        print("=" * 60)
        
        self.cleanup()
        return self.failed == 0
    
    # === 测试用例 ===
    
    def test_basic_retrieval(self):
        """基础召回: 关键词匹配"""
        print("\n[基础召回]")
        
        results = self.memory.retrieve(user_id="test_user", query="猫 去世", top_k=3)
        self._check("猫去世检索", len(results) > 0 and "猫" in results[0].content,
                     f"期望包含'猫'，实际: {results[0].content[:50] if results else '空'}")
        
        # top-3内应包含物理
        results = self.memory.retrieve(user_id="test_user", query="物理 量子", top_k=3)
        physics_found = any("物理" in r.content for r in results)
        self._check("物理检索", physics_found,
                     f"top-3应包含'物理'，实际: {[r.content[:30] for r in results]}")
        
        # top-5内应包含压力（importance=0.5较低，但relevance高）
        results = self.memory.retrieve(user_id="test_user", query="压力 作业", top_k=5)
        stress_found = any("压力" in r.content for r in results)
        self._check("压力检索", stress_found,
                     f"top-5应包含'压力'，实际: {[r.content[:30] for r in results]}")
    
    def test_alias_resolution(self):
        """别名解析: 考试 → 测验"""
        print("\n[别名解析]")
        
        # "测验"是"考试"的候选拼接，应该能匹配到"期中考试"
        results = self.memory.retrieve(user_id="test_user", query="测验 成绩", top_k=3)
        self._check("测验→考试别名", len(results) > 0,
                     f"别名扩展后应匹配到考试相关记忆")
        
        # 别名词典直接测试
        resolved = self.alias.resolve("考试")
        self._check("别名resolve", resolved == "测验",
                     f"期望'测验'，实际: {resolved}")
        
        expanded = self.alias.expand_query("考试考得好")
        self._check("查询扩展", "测验" in expanded,
                     f"扩展后应包含'测验'，实际: {expanded}")
    
    def test_idf_scoring(self):
        """IDF-aware: 高频词降权"""
        print("\n[IDF检索]")
        
        # 插入大量包含"用户"的记忆（高频词）
        for i in range(10):
            self.memory.store(user_id="test_user", content=f"用户说了第{i}句话，没什么特别的",
                              importance=0.3)
        
        # "用户"是高频词，IDF低；"物理"是低频词，IDF高
        # 搜"物理"时，包含"物理"的记忆应排在包含"用户"的记忆前面
        results = self.memory.retrieve(user_id="test_user", query="物理", top_k=5)
        if results:
            top_has_physics = "物理" in results[0].content
            top_is_filler = "没什么特别" in results[0].content
            self._check("低频词优先", top_has_physics or not top_is_filler,
                         f"'物理'应排在填充记忆前面，实际top1: {results[0].content[:50]}")
        else:
            self._check("低频词优先", False, "无结果")
    
    def test_emotional_congruence(self):
        """情感一致性: 悲伤状态偏向悲伤记忆"""
        print("\n[情感一致性]")
        
        import numpy as np
        
        # 悲伤状态
        sad_bias = np.array([0.2, 0.3, 0.3, 0.8])  # low valence, high stress
        results = self.memory.retrieve(user_id="test_user", query="", top_k=3,
                                       emotional_bias=sad_bias)
        if results:
            # 悲伤状态应偏向猫去世(0.2)或压力(0.3)
            top_valence = results[0].emotional_tag.get('valence', 0.5)
            self._check("悲伤→低valence", top_valence < 0.5,
                         f"悲伤状态应检索低valence记忆，实际valence={top_valence}")
        else:
            self._check("悲伤检索", False, "无结果")
        
        # 开心状态
        happy_bias = np.array([0.7, 0.9, 0.7, 0.2])
        results = self.memory.retrieve(user_id="test_user", query="", top_k=3,
                                       emotional_bias=happy_bias)
        if results:
            # top-3内应有高valence记忆
            has_high = any(r.emotional_tag.get('valence', 0.5) > 0.5 for r in results)
            self._check("开心→高valence", has_high,
                         f"top-3应有高valence记忆，实际: {[r.emotional_tag.get('valence',0.5) for r in results]}")
        else:
            self._check("开心检索", False, "无结果")
    
    def test_status_filtering(self):
        """状态过滤: archived不被检索"""
        print("\n[状态过滤]")
        
        # 存一条然后归档
        eid = self.memory.store(user_id="test_user", content="这条要被归档",
                                importance=0.1)
        self.memory.update_status(eid, "archived")
        
        results = self.memory.retrieve(user_id="test_user", query="归档", top_k=10,
                                       status_filter="verified")
        archived_found = any("归档" in r.content for r in results)
        self._check("archived不被检索", not archived_found,
                     "archived状态的记忆不应出现在verified过滤结果中")
    
    def test_entity_id_system(self):
        """Entity ID系统: 唯一性、可查询"""
        print("\n[Entity ID]")
        
        eid1 = self.memory.store(user_id="test_user", content="测试entity_id唯一性")
        eid2 = self.memory.store(user_id="test_user", content="另一条测试")
        
        self._check("ID格式", eid1.startswith("ENT_"),
                     f"期望ENT_开头，实际: {eid1}")
        self._check("ID唯一", eid1 != eid2,
                     f"两条记忆ID应不同: {eid1} vs {eid2}")
        
        # 按entity_id查询
        rec = self.memory.get_by_entity_id(eid1)
        self._check("按ID查询", rec is not None and "entity_id" in rec.content,
                     f"按ID查询应返回记录，实际: {rec}")
    
    def test_knowledge_graph(self):
        """知识图谱: 实体+关系+事实"""
        print("\n[知识图谱]")
        
        # 查实体
        entity = self.kg.get_entity("小花")
        self._check("实体查找", entity is not None and entity.entity_type == "pet",
                     f"期望pet类型，实际: {entity}")
        
        # 查事实
        facts = self.kg.get_all_facts("test_user")
        self._check("事实查询", len(facts) > 0,
                     f"应有事实，实际: {facts}")
        
        # 模糊搜索
        results = self.kg.search_entities("花")
        self._check("模糊搜索", len(results) > 0,
                     f"搜索'花'应找到'小花'，实际: {len(results)}条")
    
    def test_multi_hop(self):
        """多跳推理"""
        print("\n[多跳推理]")
        
        user = self.kg.get_entity("test_user")
        if user:
            # test_user → 拥有 → 小花
            paths = self.kg.multi_hop(user.id, ["拥有"])
            self._check("单跳:拥有", len(paths) > 0,
                         f"应找到test_user→拥有→小花，实际: {len(paths)}条路径")
        else:
            self._check("多跳前提", False, "test_user实体不存在")
    
    def test_confidence_decay(self):
        """置信度衰减"""
        print("\n[置信度衰减]")
        
        eid = self.memory.store(user_id="test_user", content="测试置信度",
                                confidence_score=1.0)
        
        # 衰减
        self.memory.decay("test_user", decay_factor=0.9)
        
        # 验证importance衰减了
        rec = self.memory.get_by_entity_id(eid)
        self._check("importance衰减", rec and rec.importance < 1.0,
                     f"importance应衰减，实际: {rec.importance if rec else 'N/A'}")
    
    def test_archive_stale(self):
        """过期归档"""
        print("\n[过期归档]")
        
        # 存一条低importance的记忆（模拟过期）
        old_time = time.time() - 40 * 86400  # 40天前
        eid = self.memory.store(user_id="test_user", content="很久以前的事",
                                importance=0.1)
        # 手动改时间
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        conn.execute("UPDATE memories SET timestamp = ? WHERE entity_id = ?",
                     (old_time, eid))
        conn.commit()
        conn.close()
        
        # 归档
        count = self.memory.archive_stale("test_user", max_age_days=30, min_importance=0.3)
        self._check("过期归档", count > 0,
                     f"应归档过期低重要性记忆，实际归档: {count}")
        
        # 验证状态
        rec = self.memory.get_by_entity_id(eid)
        self._check("归档状态", rec and rec.status == "archived",
                     f"期望archived，实际: {rec.status if rec else 'N/A'}")
    
    def test_frontmatter(self):
        """YAML frontmatter导出"""
        print("\n[Frontmatter]")
        
        from episodic_memory import MemoryRecord
        rec = MemoryRecord(
            entity_id="ENT_0001",
            content="测试frontmatter",
            version="2026-06-10",
            status="verified",
            confidence_score=0.9,
            tags=["test"],
            related_entities=["ENT_0000"],
        )
        fm = rec.to_frontmatter()
        self._check("frontmatter格式", "entity_id:" in fm and "status:" in fm and "tags:" in fm,
                     f"frontmatter应包含必要字段，实际: {fm[:100]}")
    
    def test_vad_analyzer(self):
        """VAD分析器"""
        print("\n[VAD分析器]")
        
        # 正面情感
        vad = self.vad.analyze("今天心情很好，非常开心")
        self._check("正面valence", vad[1] > 0.5,
                     f"正面文本valence应>0.5，实际: {vad[1]:.2f}")
        
        # 负面情感
        sad_vad = self.vad.analyze("很难过，压力很大")
        self._check("负面valence", sad_vad[1] < 0.5,
                     f"负面文本valence应<0.5，实际: {sad_vad[1]:.2f}")
        
        # 否定翻转
        neg_vad = self.vad.analyze("不开心")
        self._check("否定翻转", neg_vad[1] < 0.5,
                     f"'不开心'valence应<0.5，实际: {neg_vad[1]:.2f}")
        
        # 情绪标签
        label = self.vad.get_emotion_label(vad)
        self._check("情绪标签", len(label) > 0,
                     f"应返回标签，实际: {label}")
    
    def test_reflection_storage(self):
        """Reflection存储"""
        print("\n[Reflection存储]")
        
        ref_id = self.memory.store_reflection(
            user_id="test_user",
            content="用户近期情感趋势分析...",
            importance=0.7,
            emotional_tag={"valence": 0.5, "arousal": 0.4, "dominance": 0.5, "stress": 0.4},
            source_ids=[1, 2, 3],
        )
        
        self._check("reflection ID", ref_id.startswith("ENT_"),
                     f"reflection应有entity_id，实际: {ref_id}")
        
        # 检索reflection
        results = self.memory.retrieve(user_id="test_user", query="情感趋势", top_k=3,
                                       memory_types=["reflection"])
        self._check("reflection检索", len(results) > 0,
                     f"应检索到reflection，实际: {len(results)}条")
    
    def test_stats(self):
        """统计信息"""
        print("\n[统计]")
        
        mem_stats = self.memory.get_stats("test_user")
        self._check("记忆统计", mem_stats['total'] > 0,
                     f"应有记忆，实际: {mem_stats}")
        
        kg_stats = self.kg.get_stats()
        self._check("图谱统计", kg_stats['entities']['total'] > 0,
                     f"应有实体，实际: {kg_stats}")


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    test = StressTest(verbose=verbose)
    success = test.run_all()
    sys.exit(0 if success else 1)
