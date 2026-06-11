"""
power_law_delay.py
===================
幂律分布式时延核 — 替代固定时延 τ

数学：
  标准 Mackey-Glass: dx/dt = β·x(t-τ)/(1+x(t-τ)^n) - γ·x
  幂律版本: dx/dt = β·∫₀^∞ K(s)·x(t-s)ds / (1+(...)）- γ·x
  
  K(s) = α · s^(-α-1),  α ∈ (0,1)
  
  离散近似：用 log-spaced 采样点 + 权重
"""

import numpy as np
from collections import deque
from typing import Optional


class PowerLawDelayBuffer:
    """
    幂律分布式时延缓冲区
    
    用 log-spaced 采样点近似 ∫₀^∞ K(s)·x(t-s)ds
    
    采样点: s_k = exp(k·Δ), k=0..N-1
    权重:   w_k ∝ s_k^(-α-1) · Δ · s_k  (log-space Jacobian)
    """
    
    def __init__(
        self,
        alpha: float = 0.5,        # 幂律指数 (0,1)
        n_samples: int = 50,        # 采样点数
        s_min: float = 1.0,         # 最小时延（步）
        s_max: float = 100.0,       # 最大时延（步）
    ):
        self.alpha = alpha
        self.n_samples = n_samples
        
        # log-spaced 采样点
        self.sample_points = np.exp(np.linspace(np.log(s_min), np.log(s_max), n_samples))
        
        # 幂律权重 K(s) = α · s^(-α-1)
        # 在 log-space 积分需要乘 s（Jacobian）
        weights = alpha * self.sample_points**(-alpha - 1) * self.sample_points
        self.weights = weights / weights.sum()  # 归一化
        
        # 历史缓冲（保存最近 s_max 步的状态）
        self.buffer = deque(maxlen=int(s_max) + 1)
    
    def update(self, x: float):
        """追加新状态"""
        self.buffer.append(x)
    
    def weighted_sum(self) -> float:
        """
        计算加权时延和: Σ_k w_k · x(t - s_k)
        
        如果历史不够长，用最早可用的状态填充
        """
        buf = list(self.buffer)
        if len(buf) < 2:
            return buf[0] if buf else 0.0
        
        result = 0.0
        for k, s in enumerate(self.sample_points):
            idx = len(buf) - 1 - int(s)
            if idx < 0:
                idx = 0  # 用最早的状态
            result += self.weights[k] * buf[idx]
        
        return result
    
    def effective_delay(self) -> float:
        """等效平均时延"""
        return float(np.sum(self.weights * self.sample_points))


class ExponentialDelayBuffer:
    """
    指数衰减时延核（简单替代方案）
    
    K(s) = (1/τ) · exp(-s/τ)
    
    优点：只需要一个 running average，不需要存完整历史
    """
    
    def __init__(self, tau: float = 20.0, decay: float = 0.95):
        self.tau = tau
        self.decay = decay
        self.state = 0.0
        self.initialized = False
    
    def update(self, x: float):
        if not self.initialized:
            self.state = x
            self.initialized = True
        else:
            self.state = self.decay * self.state + (1 - self.decay) * x
    
    def get(self) -> float:
        return self.state


# ============ 对比测试 ============
def compare_delays():
    """对比固定时延 vs 幂律时延 vs 指数时延"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    n_steps = 500
    beta, n, gamma = 0.2, 10.0, 0.1
    
    # 固定时延 τ=20
    x_fixed = np.zeros(n_steps)
    buf_fixed = deque([1.0]*20, maxlen=20)
    for t in range(n_steps):
        xd = buf_fixed[0]
        dx = beta * xd / (1 + xd**n) - gamma * x_fixed[t]
        if t < n_steps - 1:
            x_fixed[t+1] = x_fixed[t] + dx
        buf_fixed.append(x_fixed[t])
    
    # 幂律时延 α=0.5
    x_pl = np.zeros(n_steps)
    pl_buf = PowerLawDelayBuffer(alpha=0.5, n_samples=50, s_min=1, s_max=80)
    for t in range(n_steps):
        xd = pl_buf.weighted_sum() if len(pl_buf.buffer) > 0 else 1.0
        dx = beta * xd / (1 + xd**n) - gamma * x_pl[t]
        if t < n_steps - 1:
            x_pl[t+1] = x_pl[t] + dx
        pl_buf.update(x_pl[t])
    
    # 指数衰减 τ=20
    x_exp = np.zeros(n_steps)
    exp_buf = ExponentialDelayBuffer(tau=20, decay=0.95)
    for t in range(n_steps):
        xd = exp_buf.get() if exp_buf.initialized else 1.0
        dx = beta * xd / (1 + xd**n) - gamma * x_exp[t]
        if t < n_steps - 1:
            x_exp[t+1] = x_exp[t] + dx
        exp_buf.update(x_exp[t])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(x_fixed, color='#e74c3c'); axes[0].set_title(f'Fixed tau=20'); axes[0].grid(True, alpha=0.3)
    axes[1].plot(x_pl, color='#3498db'); axes[1].set_title(f'Power-law alpha=0.5'); axes[1].grid(True, alpha=0.3)
    axes[2].plot(x_exp, color='#2ecc71'); axes[2].set_title(f'Exponential tau=20'); axes[2].grid(True, alpha=0.3)
    plt.suptitle('Delay Kernel Comparison (Mackey-Glass)', fontsize=13)
    plt.tight_layout()
    plt.savefig('E:/Workspace/chaotic_substrate/delay_comparison.png', dpi=150)
    plt.close()
    print("Saved delay_comparison.png")

if __name__ == "__main__":
    compare_delays()
