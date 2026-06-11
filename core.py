"""
affective_substrate.core
========================
双系统混沌动力学 + 储备池记忆
模拟交感/副交感神经系统对LLM激活的调制

架构：
  快系统 x(t) — Mackey-Glass混沌，秒级，"激素水平"
  慢系统 y(t) — 积分泄漏，分钟级，"副交感调节"
  储备池 r(t) — 分布式时延，秒-分钟，"工作记忆"

数学：
  dx_i/dt = β_i · x_i(t-τ_i) / (1 + x_i(t-τ_i)^n_i) - γ_i(y_i) · x_i + ξ_i(t)
  dy_i/dt = ε_i · (|x_i| - y_i),  ε_i << 1/τ_i
  γ_i(y_i) = γ_0 + κ · y_i
  dr/dt = -r/τ_r + tanh(W_r·r + W_x·x + W_in·u(t))
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class HormoneConfig:
    """单个"激素"的Mackey-Glass参数"""
    beta: float = 0.2        # 增益
    tau: int = 20            # 时延（步数）
    n: float = 10.0          # 非线性阶数
    gamma_0: float = 0.1     # 基础阻尼
    kappa: float = 0.3       # 慢系统对阻尼的调制强度
    x0: float = 1.0          # 初始状态


@dataclass
class SlowSystemConfig:
    """慢系统（副交感）参数"""
    epsilon: float = 0.01    # 时间尺度分离比（<< 1/tau）
    y0: float = 0.5          # 初始状态


@dataclass
class ReservoirConfig:
    """储备池参数"""
    n_units: int = 300       # 储备池维度
    spectral_radius: float = 0.9  # 谱半径（<1保证echo state）
    input_scaling: float = 0.5
    tau_r: float = 10.0      # 储备池时间常数
    sparsity: float = 0.1    # 连接稀疏度


class AffectiveSubstrate:
    """
    情感基底：双系统 + 储备池
    
    用法：
        sub = AffectiveSubstrate(n_hormones=4)
        for step in range(1000):
            feedback = np.random.randn(4) * 0.1  # 模拟LLM输出的情感反馈
            state = sub.step(feedback)
            # state包含 x, y, r，可以注入LLM
    """
    
    def __init__(
        self,
        n_hormones: int = 4,
        hormone_configs: Optional[List[HormoneConfig]] = None,
        slow_config: Optional[SlowSystemConfig] = None,
        reservoir_config: Optional[ReservoirConfig] = None,
        seed: Optional[int] = None,
    ):
        self.n_hormones = n_hormones
        rng = np.random.RandomState(seed)
        
        # === 快系统 x(t) ===
        if hormone_configs is None:
            # 默认4种"激素"，不同的时间尺度
            hormone_configs = [
                HormoneConfig(tau=10, beta=0.2, n=10, x0=1.0),   # 快：肾上腺素样
                HormoneConfig(tau=20, beta=0.15, n=10, x0=0.8),  # 中：多巴胺样
                HormoneConfig(tau=30, beta=0.1, n=12, x0=0.6),   # 慢：皮质醇样
                HormoneConfig(tau=50, beta=0.08, n=8, x0=0.5),   # 极慢：血清素样
            ][:n_hormones]
        
        self.betas = np.array([c.beta for c in hormone_configs])
        self.taus = np.array([c.tau for c in hormone_configs], dtype=int)
        self.ns = np.array([c.n for c in hormone_configs])
        self.gamma_0s = np.array([c.gamma_0 for c in hormone_configs])
        self.kappas = np.array([c.kappa for c in hormone_configs])
        
        # 状态
        self.x = np.array([c.x0 for c in hormone_configs])
        
        # 时延缓冲区
        self.delay_buffers = [
            deque([c.x0] * c.tau, maxlen=c.tau) for c in hormone_configs
        ]
        
        # === 慢系统 y(t) ===
        if slow_config is None:
            slow_config = SlowSystemConfig()
        self.epsilon = slow_config.epsilon
        self.y = np.array([slow_config.y0] * n_hormones)
        
        # === 储备池 r(t) ===
        if reservoir_config is None:
            reservoir_config = ReservoirConfig()
        rc = reservoir_config
        
        # 储备池内部权重（随机，不训练）
        W_r = rng.randn(rc.n_units, rc.n_units)
        mask = rng.random((rc.n_units, rc.n_units)) < rc.sparsity
        W_r *= mask
        # 谱归一化
        eigvals = np.linalg.eigvals(W_r)
        W_r *= rc.spectral_radius / np.max(np.abs(eigvals))
        self.W_r = W_r
        
        # 输入权重：激素状态 → 储备池
        self.W_x = rng.randn(rc.n_units, n_hormones) * rc.input_scaling
        
        # 储备池状态
        self.r = np.zeros(rc.n_units)
        self.tau_r = rc.tau_r
        
        # 联合噪声源
        self.rng = rng
    
    def _mackey_glass_step(self, x: np.ndarray, x_delayed: np.ndarray, 
                           gamma_eff: np.ndarray, feedback: np.ndarray) -> np.ndarray:
        """快系统单步（Euler法）"""
        # dx/dt = β * x(t-τ) / (1 + x(t-τ)^n) - γ(y) * x + ξ
        numerator = self.betas * x_delayed
        denominator = 1.0 + np.power(np.maximum(x_delayed, 0), self.ns)
        dx = numerator / denominator - gamma_eff * x + feedback
        return dx
    
    def _slow_system_step(self, x: np.ndarray) -> np.ndarray:
        """慢系统单步"""
        return self.epsilon * (np.abs(x) - self.y)
    
    def _reservoir_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """储备池单步"""
        # dr/dt = -r/τ_r + tanh(W_r·r + W_x·x + W_in·u)
        # u 是外部输入（比如LLM输出嵌入的投影），这里先用零
        dr = -self.r / self.tau_r + np.tanh(
            self.W_r @ self.r + self.W_x @ x
        )
        return dr
    
    def step(self, feedback: np.ndarray, dt: float = 1.0) -> dict:
        """
        推进一步
        
        Args:
            feedback: 来自LLM的情感反馈 [n_hormones]
                      （模拟VAD分类器输出）
            dt: 时间步长
        
        Returns:
            dict: {
                'x': 快系统状态,
                'y': 慢系统状态,
                'r': 储备池状态,
                'gamma_eff': 有效阻尼,
                'injection': 注入LLM的调制向量
            }
        """
        # 获取时延值
        x_delayed = np.array([buf[0] for buf in self.delay_buffers])
        
        # 有效阻尼 = γ_0 + κ * y
        gamma_eff = self.gamma_0s + self.kappas * self.y
        
        # 更新快系统
        dx = self._mackey_glass_step(self.x, x_delayed, gamma_eff, feedback)
        self.x += dx * dt
        
        # 更新慢系统
        dy = self._slow_system_step(self.x)
        self.y += dy * dt
        
        # 更新储备池
        dr = self._reservoir_step(self.x, np.zeros(1))  # 暂时无外部输入
        self.r += dr * dt
        
        # 更新时延缓冲区
        for i, buf in enumerate(self.delay_buffers):
            buf.append(self.x[i])
        
        return {
            'x': self.x.copy(),
            'y': self.y.copy(),
            'r': self.r.copy(),
            'gamma_eff': gamma_eff.copy(),
            'dx': dx.copy(),
        }


class AffectiveVisualizer:
    """可视化工具"""
    
    @staticmethod
    def run_simulation(substrate: AffectiveSubstrate, n_steps: int = 2000,
                       feedback_fn=None) -> dict:
        """
        运行仿真并记录所有状态
        
        Args:
            substrate: 情感基底实例
            n_steps: 仿真步数
            feedback_fn: 反馈函数 (step, state) -> feedback
                        默认：给一个小的随机扰动 + 正弦驱动
        """
        history = {
            'x': np.zeros((n_steps, substrate.n_hormones)),
            'y': np.zeros((n_steps, substrate.n_hormones)),
            'r': np.zeros((n_steps, substrate.r.shape[0])),
            'gamma_eff': np.zeros((n_steps, substrate.n_hormones)),
            'dx': np.zeros((n_steps, substrate.n_hormones)),
        }
        
        for t in range(n_steps):
            if feedback_fn is not None:
                fb = feedback_fn(t, history)
            else:
                # 默认：小随机 + 正弦驱动（模拟间歇性刺激）
                fb = substrate.rng.randn(substrate.n_hormones) * 0.05
                # 每隔一段时间给一个刺激脉冲
                if t % 200 < 10:
                    fb += 0.3 * np.sin(2 * np.pi * t / 500)
            
            state = substrate.step(fb)
            for key in ['x', 'y', 'r', 'gamma_eff', 'dx']:
                history[key][t] = state[key]
        
        return history
