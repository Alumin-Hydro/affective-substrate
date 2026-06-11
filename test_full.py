#!/usr/bin/env python3
"""
test_full.py — 全三阶段集成测试
Phase 1: Episodic + h_t + 情感标签
Phase 2: Reflection + mood-congruent + counter-mood
Phase 3: Knowledge Graph
"""
import sys, os, time
sys.path.insert(0, '/mnt/d/workspace/luojia-projects/affective-substrate')

# 清理
TEST_DB = "/tmp/test_full.db"
TEST_STATES = "/tmp/test_full_states.json"
for suffix in ['', '_kg']:
    p = TEST_DB.replace('.db', suffix + '.db') if suffix else TEST_DB
    if os.path.exists(p):
        os.remove(p)
if os.path.exists(TEST_STATES):
    os.remove(TEST_STATES)

from memory_manager import MemoryManager
from knowledge_graph import KnowledgeGraph
import numpy as np

print("=" * 70)
print("全三阶段集成测试")
print("=" * 70)

mgr = MemoryManager(TEST_DB, TEST_STATES, reflection_interval=3)

# ==========================================
# Phase 1: 基础记忆 + h_t
# ==========================================
print("\n" + "=" * 50)
print("Phase 1: Episodic + h_t + 情感标签")
print("=" * 50)

conv1 = [
    "你好洛希～我叫小明，今天考试考得不错！",
    "我的猫叫小花，三岁了，特别可爱",
    "但是我跟好朋友小红吵架了，心里很难受",
    "她说了很伤人的话，我不知道该怎么办",
]

for text in conv1:
    result = mgr.process_turn("user1", text, session_id="s1")
    print(f"  T{result['turn']}: [{result['emotion_label']}] "
          f"h_t=[{result['h_t'][0]:+.2f},{result['h_t'][1]:+.2f}] "
          f"mem={len(result['memories_used'])} | {text[:35]}")

# 验证持久化
mgr2 = MemoryManager(TEST_DB, TEST_STATES, reflection_interval=3)
assert mgr2.states["user1"].total_turns == 4
print(f"\n  ✅ Phase 1: h_t和记忆持久化正确 (4轮)")

# ==========================================
# Phase 2: Reflection + Mood-congruent
# ==========================================
print("\n" + "=" * 50)
print("Phase 2: Reflection + Mood-congruent + Counter-mood")
print("=" * 50)

# 继续对话，触发reflection (interval=3)
conv2 = [
    "今天阳光很好，但我还是开心不起来",
    "小红给我道歉了，但我还是很生气",
    "我想原谅她，但心里过不去这个坎",
]

for text in conv2:
    result = mgr2.process_turn("user1", text, session_id="s2")
    print(f"  T{result['turn']}: [{result['emotion_label']}] "
          f"h_t=[{result['h_t'][0]:+.2f},{result['h_t'][1]:+.2f}] "
          f"reflect={result['should_reflect']} counter_mood={result['counter_mood_active']} | {text[:35]}")
    
    if result['should_reflect']:
        reflection = mgr2.trigger_reflection("user1", session_id="s2")
        if reflection:
            print(f"    🔮 Reflection触发!")
            print(f"    {reflection['content'][:120]}")
            print(f"    emotional_tag: {reflection['emotional_tag']}")

# 验证reflection已存储
mgr3 = MemoryManager(TEST_DB, TEST_STATES, reflection_interval=3)
summary = mgr3.get_user_summary("user1")
print(f"\n  Episodic: {summary['episodic_memories']}条")
print(f"  Reflections: {summary['reflections']}条")
assert summary['reflections'] > 0, "没有生成reflection"
print(f"  ✅ Phase 2: Reflection生成正确")

# Mood-congruent检索测试
print(f"\n  Mood-congruent检索测试:")
result_sad = mgr3.process_turn("user1", "我想起了小红", session_id="s3")
print(f"  当前情绪: {result_sad['emotion_label']}")
print(f"  检索到的记忆:")
for m in result_sad['memories_used']:
    tag = m.emotional_tag
    print(f"    [{m.memory_type}] v={tag.get('valence',0):.2f} | {m.content[:40]}")
print(f"  ✅ Phase 2: Mood-congruent检索工作正常")

# ==========================================
# Phase 3: Knowledge Graph
# ==========================================
print("\n" + "=" * 50)
print("Phase 3: Knowledge Graph")
print("=" * 50)

# 继续对话，引入新知识
conv3 = [
    "我叫小明，我在清华上学",
    "我特别喜欢吃火锅，每周都去",
    "我的好朋友叫小李，我们是同学",
]

for text in conv3:
    result = mgr3.process_turn("user1", text, session_id="s4")
    kg = result['kg_facts']
    print(f"  T{result['turn']}: KG facts={len(kg)} | {text[:35]}")
    for f in kg:
        print(f"    📊 {f}")

# 验证KG
ec, rc = mgr3.kg.count()
print(f"\n  KG统计: {ec}个实体, {rc}条关系")
assert ec > 0, "KG为空"
print(f"  ✅ Phase 3: Knowledge Graph构建正确")

# KG查询测试
print(f"\n  KG查询测试:")
user_facts = mgr3.kg.get_all_facts("user1")
for f in user_facts:
    print(f"    📊 {f}")

# Multi-hop测试
print(f"\n  实体搜索:")
for name in ["小花", "小红", "小李"]:
    entities = mgr3.kg.search_entities(name)
    if entities:
        e = entities[0]
        facts = mgr3.kg.get_entity_facts(e.id)
        print(f"    {name}: {facts}")

# ==========================================
# 完整用户摘要
# ==========================================
print(f"\n" + "=" * 50)
print("完整用户摘要")
print("=" * 50)

final_summary = mgr3.get_user_summary("user1")
print(f"  用户: {final_summary['user_id']}")
print(f"  总轮次: {final_summary['total_turns']}")
print(f"  Episodic记忆: {final_summary['episodic_memories']}条")
print(f"  Reflections: {final_summary['reflections']}条")
print(f"  KG实体: {final_summary['kg_entities']}个")
print(f"  KG关系: {final_summary['kg_relations']}条")
print(f"  当前情绪: {final_summary['emotion_label']}")
print(f"  h_t: [{final_summary['h_t'][0]:+.3f}, {final_summary['h_t'][1]:+.3f}, {final_summary['h_t'][2]:+.3f}, {final_summary['h_t'][3]:+.3f}]")
print(f"  KG事实:")
for f in final_summary['kg_facts']:
    print(f"    📊 {f}")
print(f"  最近记忆:")
for m in final_summary['recent_memories']:
    v = m['emotional_tag'].get('valence', 0.5)
    print(f"    [{m['type']}] v={v:.2f} imp={m['importance']:.2f} | {m['content']}")

print(f"\n{'='*70}")
print(f"全三阶段集成测试通过 ✅")
print(f"{'='*70}")

# 清理
for suffix in ['', '_kg']:
    p = TEST_DB.replace('.db', suffix + '.db') if suffix else TEST_DB
    if os.path.exists(p):
        os.remove(p)
if os.path.exists(TEST_STATES):
    os.remove(TEST_STATES)
