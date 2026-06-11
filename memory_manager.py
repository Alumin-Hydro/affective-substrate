"""
memory_manager.py
=================
记忆管理器 — 整合动力学状态、情景记忆、Reflection

职责：
  1. h_t 状态向量持久化（per-user）
  2. 每轮对话后自动写入episodic记忆
  3. 检索时合成三层steering: h_t + 记忆情感标签 + 即时VAD
  4. 触发reflection循环
  5. Mood-congruent检索 + counter-mood机制
"""

import json
import os
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from core import AffectiveSubstrate
from vad_analyzer import VADAnalyzer
from episodic_memory import EpisodicMemory, MemoryRecord
from knowledge_graph import KnowledgeGraph
from alias_dict import AliasDict


@dataclass
class UserState:
    """单个用户的完整情感状态"""
    user_id: str
    h_t: List[float]       # 动力学状态向量（激素水平）
    h_y: List[float]       # 慢系统状态
    h_r: List[float]       # 储备池状态（可选，序列化大）
    session_turns: int     # 当前会话轮次
    total_turns: int       # 历史总轮次
    last_session: float    # 上次会话时间
    reflection_threshold: int  # 距下次reflection还差几轮
    mood_trend: List[float]    # 最近N轮的情绪趋势（用于counter-mood）


class MemoryManager:
    """
    记忆管理器
    
    用法:
        mgr = MemoryManager("data/memory.db", "data/user_states.json")
        
        # 每轮对话
        result = mgr.process_turn(user_id, user_input, session_id)
        # result: {
        #   'steering_vector': np.ndarray,  # 注入LLM的向量
        #   'memories_used': List[MemoryRecord],  # 本轮检索到的记忆
        #   'vad': List[float],  # 当前输入的VAD
        #   'emotion_label': str,  # 当前情绪标签
        #   'should_reflect': bool,  # 是否需要触发reflection
        # }
        
        # 生成回复后（可选：LLM自己评估重要性）
        mgr.post_response(user_id, response_text, importance=0.5)
    """
    
    def __init__(
        self,
        db_path: str = "episodic_memory.db",
        states_path: str = "user_states.json",
        alias_dict_path: str = None,
        reflection_interval: int = 10,
        mood_window: int = 5,
        counter_mood_threshold: float = 0.7,
    ):
        # 别名词典
        if alias_dict_path:
            self.alias_dict = AliasDict(alias_dict_path)
        else:
            self.alias_dict = None
        
        self.memory = EpisodicMemory(db_path, alias_dict=self.alias_dict)
        self.states_path = states_path
        self.vad = VADAnalyzer()
        self.reflection_interval = reflection_interval
        self.mood_window = mood_window
        self.counter_mood_threshold = counter_mood_threshold
        
        # Knowledge Graph
        kg_path = db_path.replace('.db', '_kg.db')
        self.kg = KnowledgeGraph(kg_path, alias_dict=self.alias_dict)
        
        # 加载用户状态
        self.states: Dict[str, UserState] = {}
        self._load_states()
    
    def _load_states(self):
        if os.path.exists(self.states_path):
            with open(self.states_path, 'r') as f:
                data = json.load(f)
            for uid, s in data.items():
                self.states[uid] = UserState(**s)
    
    def _save_states(self):
        data = {}
        for uid, s in self.states.items():
            data[uid] = {
                'user_id': s.user_id,
                'h_t': s.h_t, 'h_y': s.h_y, 'h_r': s.h_r[:20],  # 只存前20维
                'session_turns': s.session_turns,
                'total_turns': s.total_turns,
                'last_session': s.last_session,
                'reflection_threshold': s.reflection_threshold,
                'mood_trend': s.mood_trend[-self.mood_window:],
            }
        with open(self.states_path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _get_or_create_state(self, user_id: str) -> UserState:
        if user_id not in self.states:
            substrate = AffectiveSubstrate(n_hormones=4, seed=hash(user_id) % 10000)
            self.states[user_id] = UserState(
                user_id=user_id,
                h_t=substrate.x.tolist(),
                h_y=substrate.y.tolist(),
                h_r=substrate.r[:20].tolist(),
                session_turns=0,
                total_turns=0,
                last_session=time.time(),
                reflection_threshold=self.reflection_interval,
                mood_trend=[],
            )
        return self.states[user_id]
    
    def process_turn(
        self,
        user_id: str,
        user_input: str,
        session_id: str = "",
        retrieve_memories: bool = True,
        top_k_memories: int = 3,
    ) -> Dict:
        """
        处理一轮对话的完整流程
        
        1. VAD分析当前输入
        2. 更新h_t动力学状态
        3. 检索相关记忆（mood-congruent）
        4. 合成steering向量
        5. 判断是否触发reflection
        """
        state = self._get_or_create_state(user_id)
        
        # 1. VAD分析
        vad = self.vad.analyze(user_input)
        
        # 2. 更新h_t（动力学演化）
        # 恢复substrate状态
        substrate = AffectiveSubstrate(n_hormones=4, seed=hash(user_id) % 10000)
        substrate.x = np.array(state.h_t)
        substrate.y = np.array(state.h_y)
        
        # VAD → 反馈映射
        feedback = np.array([
            vad[0] * 2 - 1,    # arousal → 肾上腺素
            vad[1] * 2 - 1,    # valence → 多巴胺
            vad[3] * 2 - 1,    # stress → 皮质醇
            (vad[1] - vad[3]) * 2 - 1,  # valence-stress → 血清素
        ])
        
        dyn_state = substrate.step(feedback)
        state.h_t = substrate.x.tolist()
        state.h_y = substrate.y.tolist()
        state.h_r = substrate.r[:20].tolist()
        state.session_turns += 1
        state.total_turns += 1
        state.last_session = time.time()
        
        # 更新mood trend
        state.mood_trend.append(vad[1])  # 追踪valence趋势
        if len(state.mood_trend) > self.mood_window:
            state.mood_trend = state.mood_trend[-self.mood_window:]
        
        # 3. 检索记忆
        memories_used = []
        kg_facts = []
        if retrieve_memories:
            # Mood-congruent: 当前情感状态偏置检索
            emotional_bias = np.array(state.h_t[:4])
            
            # Counter-mood: 检测是否需要注入相反情感的记忆
            counter_mood_active = self._check_counter_mood(state)
            if counter_mood_active:
                emotional_bias = -emotional_bias  # 反转情感偏置
            
            memories_used = self.memory.retrieve(
                user_id=user_id,
                query=user_input,
                top_k=top_k_memories,
                emotional_bias=emotional_bias,
                recency_weight=0.3,
                importance_weight=0.3,
                relevance_weight=0.3,
                emotional_weight=0.1,
            )
            
            # Knowledge Graph: 从输入中提取关键词查询图谱
            import re
            keywords = re.findall(r'[\w\u4e00-\u9fff]+', user_input)
            for kw in keywords:
                entities = self.kg.search_entities(kw, limit=3)
                for ent in entities:
                    facts = self.kg.get_entity_facts(ent.id)
                    kg_facts.extend(facts[:2])  # 每个实体最多2条事实
            kg_facts = list(set(kg_facts))[:5]  # 去重，最多5条
        
        # 4. 合成steering向量
        steering = self._compute_steering(state, vad, memories_used)
        
        # 5. 判断reflection
        state.reflection_threshold -= 1
        should_reflect = state.reflection_threshold <= 0
        
        # 6. 自动存储episodic记忆
        importance = self._estimate_importance(user_input, vad)
        mem_entity_id = self.memory.store(
            user_id=user_id,
            content=user_input,
            importance=importance,
            emotional_tag={
                'valence': vad[1], 'arousal': vad[0],
                'dominance': vad[2], 'stress': vad[3],
            },
            tags=self._extract_auto_tags(user_input, vad),
            session_id=session_id,
        )
        
        # 7. 知识图谱提取
        kg_entities = self.kg.extract_from_text(user_input)
        kg_relations = self.kg.extract_relations_from_text(user_input, user_id)
        
        # 确保用户实体存在
        self.kg.add_entity(user_id, "person")
        
        for name, etype, props in kg_entities:
            self.kg.add_entity(name, etype, props)
        
        for subj, rel, obj in kg_relations:
            subj_id = self.kg.add_entity(subj, "person" if subj == user_id else "concept")
            obj_id = self.kg.add_entity(obj, "concept")
            self.kg.add_relation(subj_id, obj_id, rel)
        
        # 保存状态
        self._save_states()
        
        emotion_label = self.vad.get_emotion_label(vad)
        
        return {
            'steering_vector': steering,
            'memories_used': memories_used,
            'kg_facts': kg_facts,
            'vad': vad,
            'emotion_label': emotion_label,
            'h_t': state.h_t,
            'should_reflect': should_reflect,
            'counter_mood_active': counter_mood_active if retrieve_memories else False,
            'turn': state.total_turns,
        }
    
    def _compute_steering(self, state: UserState, vad: List[float], 
                          memories: List[MemoryRecord]) -> np.ndarray:
        """
        合成steering向量 = h_t + 记忆情感标签加权和 + 即时VAD
        
        三层对应：人格底色 + 情境激活 + 即时反应
        """
        n_dim = 4096  # 匹配模型隐藏维度
        
        # 层1: h_t → 注入向量（人格底色）
        # 用和activation_steering_server相同的方式
        rng = np.random.RandomState(42)
        directions = rng.randn(4, n_dim)
        # 正交化
        for i in range(4):
            for j in range(i):
                directions[i] -= np.dot(directions[i], directions[j]) / (np.dot(directions[j], directions[j]) + 1e-8) * directions[j]
            directions[i] /= (np.linalg.norm(directions[i]) + 1e-8)
        
        h_t = np.array(state.h_t)
        h_y = np.array(state.h_y)
        baseline_x = np.ones(4)
        baseline_y = np.ones(4) * 0.5
        
        A, s = 0.1, 2.0
        a_s = A * np.tanh(s * (h_t - baseline_x))
        a_p = A * np.tanh(s * (h_y - baseline_y))
        
        steering_personality = a_s @ directions - a_p @ directions
        
        # 层2: 记忆情感标签加权和（情境激活）
        steering_memory = np.zeros(n_dim)
        if memories:
            for mem in memories:
                tag = mem.emotional_tag
                tag_vec = np.array([
                    tag.get('valence', 0.5) - 0.5,
                    tag.get('arousal', 0.3) - 0.3,
                    tag.get('dominance', 0.5) - 0.5,
                    tag.get('stress', 0.3) - 0.3,
                ])
                steering_memory += tag_vec @ directions * mem.importance
            steering_memory /= len(memories)
        
        # 层3: 即时VAD反应
        vad_vec = np.array([vad[0] - 0.5, vad[1] - 0.5, vad[2] - 0.5, vad[3] - 0.3])
        steering_immediate = vad_vec @ directions * 0.3  # 权重较低，避免即时反应过强
        
        # 合成
        steering = steering_personality + steering_memory * 0.5 + steering_immediate
        
        # 限制范数
        norm = np.linalg.norm(steering)
        limit = np.sqrt(n_dim) * 0.15
        if norm > limit:
            steering *= limit / norm
        
        return steering
    
    def _check_counter_mood(self, state: UserState) -> bool:
        """检测是否需要counter-mood机制"""
        if len(state.mood_trend) < 3:
            return False
        
        # 检查最近N轮是否持续单向
        trend = state.mood_trend[-self.mood_window:]
        if len(trend) < 3:
            return False
        
        # 全部低于0.3（持续负面）或全部高于0.7（持续正面）
        all_low = all(v < 0.3 for v in trend)
        all_high = all(v > 0.7 for v in trend)
        
        return all_low or all_high
    
    def _estimate_importance(self, text: str, vad: List[float]) -> float:
        """估计对话重要性"""
        importance = 0.3  # 基线
        
        # 情感强度高的更重要
        emotional_intensity = abs(vad[0] - 0.5) + abs(vad[1] - 0.5)
        importance += emotional_intensity * 0.3
        
        # 长文本通常更重要
        if len(text) > 100:
            importance += 0.1
        if len(text) > 300:
            importance += 0.1
        
        # 含特定关键词的更重要
        important_keywords = ['喜欢', '讨厌', '害怕', '梦想', '家人', '朋友', '爱', '恨',
                              '毕业', '工作', '结婚', '生病', '去世', '生日', '重要',
                              'love', 'hate', 'fear', 'dream', 'family', 'important']
        for kw in important_keywords:
            if kw in text.lower():
                importance += 0.1
                break
        
        return min(importance, 1.0)
    
    def _extract_auto_tags(self, text: str, vad: List[float]) -> List[str]:
        """自动提取标签"""
        tags = []
        
        # 情感标签
        if vad[1] < 0.3:
            tags.append("negative")
        elif vad[1] > 0.7:
            tags.append("positive")
        
        if vad[3] > 0.7:
            tags.append("stress")
        if vad[0] > 0.7:
            tags.append("excited")
        
        # 主题标签
        topic_map = {
            'pet': ['猫', '狗', '宠物', '小花', '小黑'],
            'exam': ['考试', '测验', '期中', '期末', '成绩'],
            'school': ['学校', '上课', '作业', '老师', '同学'],
            'work': ['工作', '上班', '公司', '项目'],
            'family': ['爸', '妈', '家人', '哥', '姐', '弟', '妹'],
            'friend': ['朋友', '同学', '小李', '小王'],
            'health': ['生病', '医院', '不舒服', '头疼'],
            'interest': ['喜欢', '爱好', '兴趣', '物理', '数学', '编程'],
        }
        text_lower = text.lower()
        for tag, keywords in topic_map.items():
            if any(kw in text_lower for kw in keywords):
                tags.append(tag)
        
        return tags
    
    def trigger_reflection(self, user_id: str, session_id: str = "") -> Optional[Dict]:
        """
        触发Reflection循环
        
        检索最近的episodic记忆，生成高阶抽象洞察
        
        Returns: None if no memories to reflect on, else reflection result
        """
        state = self._get_or_create_state(user_id)
        
        # 获取最近的episodic记忆
        recent = self.memory.get_recent(user_id, n=20, memory_type="episodic")
        if len(recent) < 3:
            state.reflection_threshold = self.reflection_interval
            self._save_states()
            return None
        
        # 分析这些记忆的情感趋势
        valences = [m.emotional_tag.get('valence', 0.5) for m in recent]
        arousals = [m.emotional_tag.get('arousal', 0.3) for m in recent]
        stresses = [m.emotional_tag.get('stress', 0.3) for m in recent]
        
        avg_valence = np.mean(valences)
        avg_arousal = np.mean(arousals)
        avg_stress = np.mean(stresses)
        
        # 趋势检测
        if len(valences) >= 3:
            trend = np.polyfit(range(len(valences)), valences, 1)[0]
            if trend > 0.05:
                trend_desc = "情绪逐渐好转"
            elif trend < -0.05:
                trend_desc = "情绪逐渐低落"
            else:
                trend_desc = "情绪相对稳定"
        else:
            trend_desc = "数据不足"
        
        # 提取高频实体
        all_entities = []
        for m in recent:
            all_entities.extend(m.entities)
        entity_counts = {}
        for e in all_entities:
            entity_counts[e] = entity_counts.get(e, 0) + 1
        top_entities = sorted(entity_counts.items(), key=lambda x: -x[1])[:5]
        
        # 生成reflection文本
        reflection_content = (
            f"用户近期{len(recent)}轮对话分析：\n"
            f"- 情感趋势：{trend_desc}（效价均值{avg_valence:.2f}）\n"
            f"- 唤醒度：{'高' if avg_arousal > 0.6 else '中' if avg_arousal > 0.3 else '低'}"
            f"（均值{avg_arousal:.2f}）\n"
            f"- 压力水平：{'高' if avg_stress > 0.6 else '中' if avg_stress > 0.3 else '低'}"
            f"（均值{avg_stress:.2f}）\n"
        )
        if top_entities:
            reflection_content += f"- 高频话题：{', '.join(e for e, _ in top_entities)}\n"
        
        # 存储reflection
        ref_importance = 0.6 + abs(avg_valence - 0.5) * 0.4  # 情感偏离越大越重要
        ref_emotional_tag = {
            'valence': float(avg_valence),
            'arousal': float(avg_arousal),
            'dominance': 0.5,
            'stress': float(avg_stress),
        }
        
        ref_id = self.memory.store_reflection(
            user_id=user_id,
            content=reflection_content,
            importance=ref_importance,
            emotional_tag=ref_emotional_tag,
            source_ids=[m.id for m in recent if m.id],
            session_id=session_id,
        )
        
        # 重置reflection计数器
        state.reflection_threshold = self.reflection_interval
        self._save_states()
        
        return {
            'reflection_id': ref_id,
            'content': reflection_content,
            'emotional_tag': ref_emotional_tag,
            'importance': ref_importance,
            'memories_analyzed': len(recent),
        }
    
    def get_user_summary(self, user_id: str) -> Dict:
        """获取用户记忆摘要"""
        state = self._get_or_create_state(user_id)
        
        episodic_count = self.memory.count(user_id, "episodic")
        reflection_count = self.memory.count(user_id, "reflection")
        kg_entities, kg_relations = self.kg.count()
        
        recent = self.memory.get_recent(user_id, n=5)
        
        # KG facts
        kg_facts = self.kg.get_all_facts(user_id)
        
        return {
            'user_id': user_id,
            'h_t': state.h_t,
            'total_turns': state.total_turns,
            'episodic_memories': episodic_count,
            'reflections': reflection_count,
            'kg_entities': kg_entities,
            'kg_relations': kg_relations,
            'kg_facts': kg_facts,
            'mood_trend': state.mood_trend,
            'emotion_label': self.vad.get_emotion_label([
                (state.h_t[0] + 1) / 2,
                (state.h_t[1] + 1) / 2,
                0.5,
                (state.h_t[2] + 1) / 2,
            ]),
            'recent_memories': [{
                'content': m.content[:80],
                'importance': m.importance,
                'type': m.memory_type,
                'emotional_tag': m.emotional_tag,
            } for m in recent],
        }
