"""
affective_substrate.hook
========================
nnsight 残差流注入层

在 LLM 的指定层残差流上注入情感调制向量：
  h' = h + α_s(x) · v_arousal - α_p(y) · v_rest + P · r

其中：
  α_s(x) = A · tanh(s · (x - x̄))  — 快系统调制（交感）
  α_p(y) = A · tanh(s · (y - ȳ))  — 慢系统调制（副交感）
  P · r   — 储备池记忆投影
"""

import torch
import numpy as np
from typing import Optional, List, Tuple


class ResidualSteering:
    """
    残差流转向器
    
    用法：
        steering = ResidualSteering(
            d_model=4096,       # Qwen3.5-9B: 4096
            n_emotions=4,
            inject_layer=15,    # 中后层
        )
        # 预设情感方向
        steering.set_directions(random_init=True)
        
        # 每步计算注入向量
        injection = steering.compute_injection(x_state, y_state, r_state)
    """
    
    def __init__(
        self,
        d_model: int = 4096,
        n_emotions: int = 4,
        inject_layer: int = 15,
        max_injection_norm: float = 0.2,  # 注入向量最大范数（相对h）
        device: str = "cuda",
    ):
        self.d_model = d_model
        self.n_emotions = n_emotions
        self.inject_layer = inject_layer
        self.max_injection_norm = max_injection_norm
        self.device = device
        
        # 情感方向 v_i ∈ R^{d_model}
        # v[0] = arousal（激动/兴奋）
        # v[1] = calm（平静/放松）
        # v[2] = positive（积极/愉悦）
        # v[3] = negative（消极/低落）
        self.directions = None  # [n_emotions, d_model]
        
        # 储备池投影 P: R^{n_reservoir} → R^{d_model}
        self.P = None
        
        # 调制参数
        self.A = 0.1        # 最大调制幅度
        self.s = 2.0         # tanh 陡度
        self.x_baseline = np.ones(n_emotions)   # x 基线
        self.y_baseline = np.ones(n_emotions) * 0.5  # y 基线
        
        # 监控
        self.last_injection_norm = 0.0
        self.last_alpha = None
        
    def set_directions(
        self, 
        directions: Optional[np.ndarray] = None,
        random_init: bool = True,
    ):
        """设置情感方向向量"""
        if directions is not None:
            self.directions = torch.tensor(
                directions, dtype=torch.float32, device=self.device
            )
        elif random_init:
            # 随机初始化，Gram-Schmidt 正交化
            raw = torch.randn(self.n_emotions, self.d_model, device=self.device)
            # Gram-Schmidt
            for i in range(self.n_emotions):
                v = raw[i].clone()
                for j in range(i):
                    v = v - (torch.dot(raw[i], self.directions[j]) / 
                            torch.dot(self.directions[j], self.directions[j])) * self.directions[j]
                raw[i] = v / v.norm()
            self.directions = raw
        else:
            # 默认：正交基的前k个
            self.directions = torch.eye(
                self.d_model, self.n_emotions, device=self.device
            ).T
            
    def set_reservoir_projection(self, n_reservoir: int, init: str = "random"):
        """设置储备池投影矩阵 P"""
        if init == "random":
            P = torch.randn(n_reservoir, self.d_model, device=self.device) * 0.01
        elif init == "ortho":
            P = torch.randn(n_reservoir, self.d_model, device=self.device)
            for i in range(n_reservoir):
                P[i] = P[i] / P[i].norm()
        self.P = P
        
    def compute_alpha(self, x: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算调制系数
        
        Returns:
            alpha_s: 快系统调制 [n_emotions]
            alpha_p: 慢系统调制 [n_emotions]
        """
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        x_base = torch.tensor(self.x_baseline, dtype=torch.float32, device=self.device)
        y_base = torch.tensor(self.y_baseline, dtype=torch.float32, device=self.device)
        
        alpha_s = self.A * torch.tanh(self.s * (x_t - x_base))
        alpha_p = self.A * torch.tanh(self.s * (y_t - y_base))
        
        self.last_alpha = (alpha_s.detach().cpu().numpy(), alpha_p.detach().cpu().numpy())
        return alpha_s, alpha_p
    
    def compute_injection(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        r: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        计算注入向量
        
        Returns:
            injection: [d_model] 注入到残差流的向量
        """
        alpha_s, alpha_p = self.compute_alpha(x, y)
        
        # v_arousal = directions[0] (激动), 其他取加权和
        # 简化：用所有方向加权
        injection = alpha_s @ self.directions - alpha_p @ self.directions
        
        # 加储备池记忆
        if r is not None and self.P is not None:
            r_t = torch.tensor(r, dtype=torch.float32, device=self.device)
            memory_injection = r_t @ self.P  # [d_model]
            injection = injection + memory_injection
        
        # 范数约束
        norm = injection.norm().item()
        h_norm_est = self.d_model ** 0.5  # 估计 ||h|| ≈ sqrt(d)
        relative_norm = norm / h_norm_est
        
        if relative_norm > self.max_injection_norm:
            injection = injection * (self.max_injection_norm / relative_norm)
            norm = injection.norm().item()
            
        self.last_injection_norm = norm
        return injection
    
    def inject_hook(self, activation: torch.Tensor, injection: torch.Tensor) -> torch.Tensor:
        """
        Hook 函数：修改残差流激活
        
        activation: [batch, seq_len, d_model] 或 [batch, d_model]
        injection: [d_model]
        """
        if activation.dim() == 3:
            # [batch, seq_len, d_model]
            return activation + injection.unsqueeze(0).unsqueeze(0)
        elif activation.dim() == 2:
            # [batch, d_model]
            return activation + injection.unsqueeze(0)
        else:
            return activation + injection


class EmotionProbe:
    """
    简易情感探针：用 LLM 内部激活预测 VAD 分数
    
    在训练阶段用线性探针读取特定层的激活，
    推理时直接从激活中解码情感状态。
    
    这里用简化版本：从输出 token 的 logprobs 估算。
    """
    
    def __init__(self, n_emotions: int = 4):
        self.n_emotions = n_emotions
        # 情感关键词（用于 logit-based 估算）
        self.emotion_words = {
            0: ["excited", "energetic", "passionate", "thrilled", "intense"],
            1: ["calm", "peaceful", "relaxed", "serene", "gentle"],
            2: ["happy", "joyful", "pleased", "delighted", "glad"],
            3: ["sad", "unhappy", "depressed", "miserable", "gloomy"],
        }
        
    def estimate_from_text(self, text: str) -> np.ndarray:
        """
        基于关键词的简易 VAD 估算（不需要额外模型）
        作为 fallback，后续可替换为真正的探针
        """
        text_lower = text.lower()
        scores = np.zeros(self.n_emotions)
        
        for emotion_idx, words in self.emotion_words.items():
            for word in words:
                if word in text_lower:
                    scores[emotion_idx] += 1.0
                    
        # 归一化
        total = scores.sum()
        if total > 0:
            scores = scores / total
        else:
            scores = np.ones(self.n_emotions) / self.n_emotions
            
        return scores
