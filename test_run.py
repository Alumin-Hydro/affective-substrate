#!/usr/bin/env python3
"""
裸跑测试：双系统 + 储备池，不接LLM
验证波形、相图、记忆效应
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import AffectiveSubstrate, AffectiveVisualizer, HormoneConfig, ReservoirConfig
from visualize import generate_all


def main():
    print("=" * 60)
    print("Affective Substrate — 裸跑测试")
    print("=" * 60)
    
    # 创建基底，4种激素
    substrate = AffectiveSubstrate(
        n_hormones=4,
        hormone_configs=[
            HormoneConfig(tau=10, beta=0.2, n=10, x0=1.0, gamma_0=0.1, kappa=0.3),
            HormoneConfig(tau=20, beta=0.15, n=10, x0=0.8, gamma_0=0.1, kappa=0.3),
            HormoneConfig(tau=30, beta=0.1, n=12, x0=0.6, gamma_0=0.1, kappa=0.3),
            HormoneConfig(tau=50, beta=0.08, n=8, x0=0.5, gamma_0=0.1, kappa=0.3),
        ],
        reservoir_config=ReservoirConfig(
            n_units=300,
            spectral_radius=0.9,
            input_scaling=0.5,
            tau_r=10.0,
            sparsity=0.1,
        ),
        seed=42,
    )
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    history = generate_all(substrate, output_dir=output_dir, n_steps=3000)
    
    # 打印统计信息
    print("\n--- 统计 ---")
    for i in range(4):
        x = history['x'][:, i]
        y = history['y'][:, i]
        print(f"hormone_{i}: x mean={x.mean():.3f} std={x.std():.3f} "
              f"y mean={y.mean():.3f} std={y.std():.3f}")
    
    r = history['r']
    print(f"\n储备池: mean={r.mean():.4f} std={r.std():.4f}")
    print(f"储备池激活率: {(np.abs(r) > 0.1).mean():.1%}")
    
    # 检查记忆效应：计算自相关
    for i in range(4):
        x = history['x'][:, i]
        autocorr = np.correlate(x - x.mean(), x - x.mean(), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr /= autocorr[0]
        # 找第一个过零点（近似记忆长度）
        zero_crossings = np.where(autocorr < 0)[0]
        if len(zero_crossings) > 0:
            memory_length = zero_crossings[0]
            print(f"hormone_{i} 记忆长度（自相关过零）: {memory_length} 步")


if __name__ == "__main__":
    main()
