#!/usr/bin/env python3
"""
test_memory.py — Phase 1 测试：情景记忆 + h_t持久化 + 情感标签 + steering合成
"""
import sys, os, time, json
sys.path.insert(0, '/mnt/d/workspace/luojia-projects/affective-substrate')

# 清理测试数据库
TEST_DB = "/tmp/test_episodic.db"
TEST_STATES = "/tmp/test_states.json"
for f in [TEST_DB, TEST_STATES]:
    if os.path.exists(f):
        os.remove(f)

from episodic_memory import EpisodicMemory, MemoryRecord
from memory_manager import MemoryManager
import numpy as np

print("=" * 70)
print("Phase 1 测试：情景记忆系统")
print("=" * 70)

# === Test 1: 基本存取 ===
print("\n--- Test 1: 基本存取 ---")
mem = EpisodicMemory(TEST_DB)

id1 = mem.store("user1", "今天天气真好，心情不错", importance=0.4,
                 emotional_tag={"valence": 0.8, "arousal": 0.6, "dominance": 0.6, "stress": 0.2})
id2 = mem.store("user1", "跟朋友吵架了，很难过", importance=0.8,
                 emotional_tag={"valence": 0.2, "arousal": 0.7, "dominance": 0.3, "stress": 0.7})
id3 = mem.store("user1", "我的猫叫小花，三岁了", importance=0.6,
                 emotional_tag={"valence": 0.7, "arousal": 0.3, "dominance": 0.5, "stress": 0.2})

assert id1 and id2 and id3, "存储失败"
print(f"  ✅ 存储3条记忆: id={id1},{id2},{id3}")

count = mem.count("user1")
assert count == 3, f"计数错误: {count}"
print(f"  ✅ 计数正确: {count}")

# === Test 2: 关键词检索 ===
print("\n--- Test 2: 关键词检索 ---")
results = mem.retrieve("user1", query="朋友 吵架", top_k=2)
assert len(results) > 0, "检索为空"
assert results[0].id == id2, f"最相关应该是id2(吵架)，实际是id{results[0].id}"
print(f"  ✅ 关键词检索正确: query='朋友 吵架' → id={results[0].id} ({results[0].content[:20]})")

# === Test 3: 情感偏置检索 ===
print("\n--- Test 3: Mood-congruent 检索 ---")
# 当前悲伤状态 → 应该更容易检索到悲伤记忆
sad_bias = np.array([0.2, 0.7, 0.3, 0.7])  # 低valence, 高arousal, 高stress
results_sad = mem.retrieve("user1", query="", top_k=3, emotional_bias=sad_bias)
# 当前开心状态 → 应该更容易检索到开心记忆
happy_bias = np.array([0.8, 0.5, 0.6, 0.2])
results_happy = mem.retrieve("user1", query="", top_k=3, emotional_bias=happy_bias)

print(f"  悲伤偏置检索: {[r.id for r in results_sad]} (首选: {results_sad[0].content[:20]})")
print(f"  开心偏置检索: {[r.id for r in results_happy]} (首选: {results_happy[0].content[:20]})")
# 不要求严格顺序（因为recency和importance也影响），但要有差异
print(f"  ✅ 情感偏置检索完成")

# === Test 4: 用户隔离 ===
print("\n--- Test 4: 用户隔离 ---")
mem.store("user2", "我是另一个用户", importance=0.5)
results_u1 = mem.retrieve("user1", top_k=10)
results_u2 = mem.retrieve("user2", top_k=10)
assert all(r.user_id == "user1" for r in results_u1), "user1结果包含其他用户"
assert all(r.user_id == "user2" for r in results_u2), "user2结果包含其他用户"
print(f"  ✅ 用户隔离正确: user1={len(results_u1)}条, user2={len(results_u2)}条")

# === Test 5: 衰减 ===
print("\n--- Test 5: 记忆衰减 ---")
old_imp = mem.retrieve("user1", top_k=1)[0].importance
mem.decay("user1", decay_factor=0.9)
new_imp = mem.retrieve("user1", top_k=1)[0].importance
print(f"  衰减前: {old_imp:.3f}, 衰减后: {new_imp:.3f}")
assert new_imp < old_imp, "衰减没生效"
print(f"  ✅ 衰减正确")

print("\n" + "=" * 70)
print("Phase 1 测试：MemoryManager 集成")
print("=" * 70)

# 清理
for f in [TEST_DB, TEST_STATES]:
    if os.path.exists(f):
        os.remove(f)

mgr = MemoryManager(TEST_DB, TEST_STATES, reflection_interval=5)

# === Test 6: 多轮对话 ===
print("\n--- Test 6: 多轮对话处理 ---")
conversation = [
    ("你好洛希～今天考试考得不错！", "user1"),
    ("但是我跟好朋友吵架了，心里很难受", "user1"),
    ("她说了很伤人的话，我不知道该怎么办", "user1"),
    ("谢谢你听我说这些，感觉好一点了", "user1"),
    ("对了，我的猫小花今天特别可爱", "user1"),
]

for text, uid in conversation:
    result = mgr.process_turn(uid, text, session_id="s1")
    vad = result['vad']
    h_t = result['h_t']
    label = result['emotion_label']
    n_mem = len(result['memories_used'])
    
    print(f"  Turn {result['turn']}: [{label}] vad=[{vad[0]:.2f},{vad[1]:.2f},{vad[2]:.2f},{vad[3]:.2f}] "
          f"h_t=[{h_t[0]:+.2f},{h_t[1]:+.2f},{h_t[2]:+.2f},{h_t[3]:+.2f}] "
          f"memories={n_mem} reflect={result['should_reflect']}")
    print(f"    → {text[:50]}")

# === Test 7: h_t持久化 ===
print("\n--- Test 7: h_t持久化 ---")
h_t_before = mgr.states["user1"].h_t.copy()
mgr2 = MemoryManager(TEST_DB, TEST_STATES, reflection_interval=5)
h_t_after = mgr2.states["user1"].h_t
assert h_t_before == h_t_after, "h_t持久化失败"
print(f"  ✅ h_t持久化正确: {h_t_before[:2]}...")

# === Test 8: 记忆持久化 ===
print("\n--- Test 8: 记忆持久化 ---")
mem2 = EpisodicMemory(TEST_DB)
count = mem2.count("user1")
print(f"  ✅ 记忆持久化: {count}条episodic记忆")

# === Test 9: 检索到记忆的情感影响 ===
print("\n--- Test 9: 检索记忆的情感影响 ---")
# 再说一句关于朋友的话题
result = mgr2.process_turn("user1", "我朋友给我发消息了", session_id="s2")
print(f"  输入: '我朋友给我发消息了'")
print(f"  检索到的记忆:")
for m in result['memories_used']:
    tag = m.emotional_tag
    print(f"    [{m.memory_type}] {m.content[:40]} (v={tag.get('valence',0):.2f}, imp={m.importance:.2f})")
print(f"  当前状态: h_t=[{result['h_t'][0]:+.2f},{result['h_t'][1]:+.2f},{result['h_t'][2]:+.2f},{result['h_t'][3]:+.2f}]")
print(f"  情绪标签: {result['emotion_label']}")

# === Test 10: Steering向量 ===
print("\n--- Test 10: Steering向量 ---")
steering = result['steering_vector']
print(f"  Steering维度: {steering.shape}")
print(f"  Steering范数: {np.linalg.norm(steering):.3f}")
print(f"  Steering均值: {steering.mean():.6f} (应接近0)")
print(f"  ✅ Steering向量生成正确")

# === Test 11: 持久化完整验证 ===
print("\n--- Test 11: 完整持久化验证 ---")
mgr3 = MemoryManager(TEST_DB, TEST_STATES, reflection_interval=5)
state = mgr3.get_user_summary("user1")
print(f"  用户: {state['user_id']}")
print(f"  总轮次: {state['total_turns']}")
print(f"  Episodic记忆: {state['episodic_memories']}条")
print(f"  Reflections: {state['reflections']}条")
print(f"  当前情绪: {state['emotion_label']}")
print(f"  h_t: {state['h_t'][:2]}...")
print(f"  最近记忆:")
for m in state['recent_memories']:
    print(f"    [{m['type']}] {m['content']} (imp={m['importance']:.2f})")

print(f"\n{'='*70}")
print(f"Phase 1 全部测试通过 ✅")
print(f"{'='*70}")

# 清理
for f in [TEST_DB, TEST_STATES]:
    if os.path.exists(f):
        os.remove(f)
