"""
vad_analyzer.py
===============
升级版VAD情绪分析器

数据源:
  - Chinese EmoBank CVAW: 5512个中文词 → Valence/Arousal 评分
  - Emoji → 情绪映射
  - 规则: 否定词、程度副词、标点信号

输出: 4维情绪向量 [arousal, valence, dominance, stress]
  映射到动力学系统的4种"激素":
    [0] 肾上腺素样 (高唤醒/兴奋)
    [1] 多巴胺样 (正向效价/愉悦)
    [2] 皮质醇样 (压力/紧张)
    [3] 血清素样 (平静/满足)
"""

import csv
import os
import re
from typing import Dict, List, Tuple, Optional
from functools import lru_cache


class LexiconLoader:
    """加载情绪词典 (NRC + CVAW)"""
    
    _lexicon: Dict[str, dict] = {}  # word -> {v, a, d, s}
    
    @classmethod
    def get(cls) -> Dict[str, dict]:
        if cls._lexicon:
            return cls._lexicon
        
        import json
        
        # 优先: 预处理好的NRC合并词典
        candidates = [
            os.path.join(os.path.dirname(__file__), "data", "nrc_va_lexicon.json"),
            "/mnt/d/workspace/luojia-projects/affective-substrate/data/nrc_va_lexicon.json",
        ]
        
        for path in candidates:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    cls._lexicon = json.load(f)
                print(f"[VAD] Loaded {len(cls._lexicon)} entries from NRC lexicon")
                return cls._lexicon
        
        # 回退: CVAW CSV
        cvaw_candidates = [
            os.path.join(os.path.dirname(__file__), "data", "CVAW_all_SD.csv"),
            "/mnt/d/workspace/luojia-projects/emotion-translation/data/Chinese-EmoBank/ChineseEmoBank/CVAW_SD/CVAW_all_SD.csv",
        ]
        
        for path in cvaw_candidates:
            if os.path.exists(path):
                loaded = 0
                with open(path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f, delimiter='\t')
                    for row in reader:
                        word = row.get('Word', '').strip()
                        try:
                            v = float(row.get('Valence_Mean', 0))
                            a = float(row.get('Arousal_Mean', 0))
                            if word:
                                cls._lexicon[word] = {
                                    'v': (v - 1) / 8, 'a': (a - 1) / 8,
                                    'd': 0.5, 's': 0.3
                                }
                                loaded += 1
                        except (ValueError, TypeError):
                            continue
                print(f"[VAD] Loaded {loaded} words from CVAW fallback")
                return cls._lexicon
        
        print(f"[VAD] Warning: No lexicon found, using built-in fallback only")
        return cls._lexicon


# === Emoji → 情绪映射 ===
EMOJI_EMOTION = {
    # 正面高唤醒
    '😀': (0.8, 0.8), '😃': (0.8, 0.8), '😄': (0.9, 0.9), '😁': (0.85, 0.85),
    '😆': (0.9, 0.9), '🤣': (0.95, 0.95), '😂': (0.95, 0.9), '😍': (0.7, 0.9),
    '🥰': (0.6, 0.9), '😘': (0.5, 0.8), '😊': (0.5, 0.7), '☺️': (0.4, 0.6),
    '🎉': (0.9, 0.8), '🥳': (0.9, 0.85), '💪': (0.7, 0.6), '👍': (0.5, 0.5),
    '❤️': (0.5, 0.8), '💕': (0.5, 0.8), '💖': (0.6, 0.85), '✨': (0.5, 0.6),
    '🔥': (0.8, 0.7), '💯': (0.7, 0.7), '🥺': (0.3, 0.6), '😏': (0.4, 0.5),
    
    # 正面低唤醒 (平静)
    '😌': (0.3, 0.3), '😴': (0.1, 0.1), '☕': (0.2, 0.2), '🌙': (0.1, 0.15),
    
    # 负面高唤醒
    '😡': (0.9, 0.2), '🤬': (0.95, 0.15), '😤': (0.8, 0.25), '💢': (0.85, 0.2),
    '😱': (0.9, 0.15), '😰': (0.8, 0.2), '😨': (0.8, 0.15), '🤯': (0.85, 0.3),
    
    # 负面低唤醒
    '😢': (0.4, 0.2), '😭': (0.7, 0.15), '😔': (0.3, 0.2), '😞': (0.3, 0.2),
    '🥺': (0.4, 0.3), '💔': (0.4, 0.15), '😶': (0.2, 0.3), '😑': (0.1, 0.3),
    
    # 中性/暧昧
    '🤔': (0.3, 0.5), '😅': (0.5, 0.4), '🫠': (0.3, 0.3), '💀': (0.5, 0.2),
}

# === 否定词 ===
NEGATION_WORDS = {
    '不', '没', '没有', '不是', '别', '莫', '未', '未曾', '无', '非',
    '不要', '不用', '不必', '不曾', '绝不', '毫不', '从不', '从未',
    '并非', '否', '否认', '难以', '无法', '不会', '不能',
}

# === 程度副词 → 强度乘数 ===
INTENSIFIERS = {
    '非常': 1.5, '特别': 1.5, '极其': 2.0, '超级': 1.8, '太': 1.6,
    '超': 1.5, '巨': 1.7, '贼': 1.5, '老': 1.4, '特': 1.5,
    '相当': 1.3, '十分': 1.4, '格外': 1.3, '尤其': 1.3, '异常': 1.5,
    '稍微': 0.6, '略': 0.7, '有点': 0.7, '些许': 0.6, '一点': 0.6,
    'so': 1.5, 'very': 1.5, 'really': 1.5, 'extremely': 2.0,
    'super': 1.8, 'quite': 1.2, 'slightly': 0.6, 'a bit': 0.7,
}

# === 标点信号 ===
PUNCTUATION_AROUSAL = {
    '！': 0.15, '!': 0.15, '？': 0.05, '?': 0.05,
    '！！': 0.25, '!!!': 0.25, '？？': 0.1, '???': 0.1,
    '~': 0.05, '～': 0.05, '。。。': -0.05, '...': -0.05,
}

# === 内置回退词典 (CVAW加载失败时使用) ===
FALLBACK_LEXICON = {
    # 中文
    '开心': (0.8, 0.6), '高兴': (0.8, 0.6), '快乐': (0.8, 0.5), '幸福': (0.7, 0.4),
    '兴奋': (0.8, 0.9), '激动': (0.7, 0.85), '期待': (0.6, 0.7), '惊喜': (0.8, 0.8),
    '难过': (0.2, 0.5), '伤心': (0.15, 0.6), '痛苦': (0.1, 0.7), '失望': (0.2, 0.4),
    '生气': (0.2, 0.8), '愤怒': (0.15, 0.9), '烦躁': (0.25, 0.7), '讨厌': (0.2, 0.6),
    '害怕': (0.2, 0.8), '恐惧': (0.15, 0.9), '紧张': (0.3, 0.8), '焦虑': (0.25, 0.75),
    '平静': (0.5, 0.2), '放松': (0.6, 0.15), '安心': (0.6, 0.2), '满足': (0.7, 0.2),
    '无聊': (0.3, 0.15), '疲惫': (0.2, 0.2), '困': (0.2, 0.1), '累': (0.2, 0.15),
    '喜欢': (0.7, 0.5), '爱': (0.8, 0.6), '感动': (0.7, 0.6), '温暖': (0.7, 0.3),
    '孤独': (0.2, 0.3), '寂寞': (0.2, 0.3), '迷茫': (0.3, 0.4), '困惑': (0.3, 0.5),
    '感谢': (0.7, 0.4), '抱歉': (0.3, 0.4), '对不起': (0.3, 0.4), '不好意思': (0.35, 0.35),
    '厉害': (0.7, 0.6), '棒': (0.8, 0.6), '好': (0.7, 0.4), '不错': (0.6, 0.4),
    '差': (0.3, 0.4), '烂': (0.2, 0.5), '垃圾': (0.15, 0.6), '恶心': (0.15, 0.7),
    '哈哈': (0.9, 0.8), '呵呵': (0.4, 0.3), '嘿嘿': (0.7, 0.6), '呜呜': (0.2, 0.5),
    '嗯': (0.5, 0.2), '哦': (0.4, 0.15), '好吧': (0.4, 0.25), '行': (0.5, 0.3),
    # English
    'happy': (0.8, 0.6), 'sad': (0.2, 0.5), 'angry': (0.2, 0.8), 'afraid': (0.2, 0.8),
    'excited': (0.8, 0.9), 'calm': (0.5, 0.2), 'love': (0.8, 0.6), 'hate': (0.2, 0.7),
    'good': (0.7, 0.4), 'bad': (0.3, 0.5), 'great': (0.8, 0.5), 'terrible': (0.15, 0.7),
    'amazing': (0.9, 0.7), 'awful': (0.15, 0.6), 'wonderful': (0.85, 0.5),
    'thanks': (0.7, 0.3), 'sorry': (0.3, 0.4), 'please': (0.5, 0.4),
    'yes': (0.6, 0.4), 'no': (0.3, 0.4), 'ok': (0.5, 0.3), 'okay': (0.5, 0.3),
}


class VADAnalyzer:
    """
    4维情绪分析器
    
    输出: [arousal, valence, dominance, stress]
      - arousal: 唤醒度 (0=平静, 1=激动)
      - valence: 效价 (0=负面, 1=正面)
      - dominance: 控制感 (0=无力, 1=掌控)
      - stress: 压力 (0=放松, 1=紧张)
    """
    
    def __init__(self):
        self.lexicon = LexiconLoader.get()
        if not self.lexicon:
            self.lexicon = FALLBACK_LEXICON.copy()
        
        # 补充内置词典
        for w, va in FALLBACK_LEXICON.items():
            if w not in self.lexicon:
                self.lexicon[w] = va
    
    def analyze(self, text: str) -> List[float]:
        """
        分析文本情绪，返回4维向量 [arousal, valence, dominance, stress]
        
        值域: [0, 1]，0.5为中性
        """
        if not text.strip():
            return [0.5, 0.5, 0.5, 0.5]
        
        text_lower = text.lower().strip()
        
        # 1. 词典匹配
        arousal_scores = []
        valence_scores = []
        
        # 否定词检测窗口
        negation_window = 0
        intensifier_mult = 1.0
        
        # 分词（简单按字/词匹配）
        i = 0
        while i < len(text_lower):
            matched = False
            
            # 尝试最长匹配（4字→1字）
            for length in range(min(4, len(text_lower) - i), 0, -1):
                word = text_lower[i:i+length]
                
                # 检查程度副词
                if word in INTENSIFIERS:
                    intensifier_mult = INTENSIFIERS[word]
                    i += length
                    matched = True
                    break
                
                # 检查否定词
                if word in NEGATION_WORDS:
                    negation_window = 3  # 向后影响3个词
                    i += length
                    matched = True
                    break
                
                # 检查情绪词典
                if word in self.lexicon:
                    entry = self.lexicon[word]
                    v = entry['v'] if isinstance(entry, dict) else entry[0]
                    a = entry['a'] if isinstance(entry, dict) else entry[1]
                    
                    # 否定翻转效价
                    if negation_window > 0:
                        v = 1.0 - v
                        a = a * 0.8
                    
                    # 程度副词放大
                    v = 0.5 + (v - 0.5) * intensifier_mult
                    a = 0.5 + (a - 0.5) * intensifier_mult
                    v = max(0, min(1, v))
                    a = max(0, min(1, a))
                    
                    valence_scores.append(v)
                    arousal_scores.append(a)
                    
                    intensifier_mult = 1.0
                    i += length
                    matched = True
                    break
            
            if not matched:
                if negation_window > 0:
                    negation_window -= 1
                i += 1
        
        # 2. Emoji匹配
        for emoji, (a, v) in EMOJI_EMOTION.items():
            if emoji in text:
                arousal_scores.append(a)
                valence_scores.append(v)
        
        # 3. 标点信号
        arousal_boost = 0
        for punct, boost in PUNCTUATION_AROUSAL.items():
            count = text.count(punct)
            if count > 0:
                arousal_boost += boost * min(count, 3)
        
        # 4. 聚合
        if valence_scores:
            valence = sum(valence_scores) / len(valence_scores)
            arousal = sum(arousal_scores) / len(arousal_scores)
        else:
            valence = 0.5
            arousal = 0.3  # 默认偏平静
        
        arousal = max(0, min(1, arousal + arousal_boost))
        
        # 5. 推导 dominance 和 stress
        # dominance: 正面效价+适度唤醒 → 高控制感
        dominance = 0.3 + 0.4 * valence + 0.2 * (1 - abs(arousal - 0.5) * 2)
        # stress: 高唤醒+负面效价 → 高压力
        stress = 0.2 + 0.4 * arousal * (1 - valence) + 0.2 * (1 - valence)
        
        dominance = max(0, min(1, dominance))
        stress = max(0, min(1, stress))
        
        return [arousal, valence, dominance, stress]
    
    def analyze_batch(self, texts: List[str]) -> List[List[float]]:
        """批量分析"""
        return [self.analyze(t) for t in texts]
    
    def get_emotion_label(self, vad: List[float]) -> str:
        """将VAD向量转为可读情绪标签"""
        a, v, d, s = vad
        
        if v > 0.7 and a > 0.6:
            return "兴奋/开心"
        elif v > 0.7 and a <= 0.6:
            return "平静/满足"
        elif v < 0.3 and a > 0.6:
            return "愤怒/焦虑"
        elif v < 0.3 and a <= 0.6:
            return "难过/低落"
        elif s > 0.7:
            return "紧张/压力"
        elif d > 0.7:
            return "自信/掌控"
        else:
            return "中性"


# === 测试 ===
if __name__ == "__main__":
    analyzer = VADAnalyzer()
    
    test_cases = [
        "今天天气真好！开心！😄",
        "我好难过，什么都不想做...",
        "你太厉害了！！！超级棒！！！",
        "这个东西不怎么样",
        "嗯，好吧",
        "I'm so excited about this!!!",
        "我不开心",
        "不是特别难过",
        "有点无聊",
        "啊啊啊啊啊太好笑了吧哈哈哈哈",
        "谢谢你，我很感动",
        "滚，别烦我",
        "晚安，做个好梦~",
    ]
    
    for text in test_cases:
        vad = analyzer.analyze(text)
        label = analyzer.get_emotion_label(vad)
        print(f"  {text:30s} → [{vad[0]:.2f} {vad[1]:.2f} {vad[2]:.2f} {vad[3]:.2f}] {label}")
