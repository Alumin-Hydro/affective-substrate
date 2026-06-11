#!/usr/bin/env python3
"""演示：对话驱动的情绪动力学可视化"""
import sys
sys.path.insert(0, '/mnt/d/workspace/luojia-projects/affective-substrate')

from core import AffectiveSubstrate
from vad_analyzer import VADAnalyzer
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

analyzer = VADAnalyzer()
substrate = AffectiveSubstrate(n_hormones=4, seed=42)

user_inputs = [
    "你好呀洛希～今天天气好好！😄",
    "嗯嗯，昨天考试考得还不错，嘿嘿",
    "但是...我跟我朋友吵架了，心里有点不舒服",
    "就是一些小事，但我就是控制不住自己的情绪，好烦",
    "谢谢你听我说这些...感觉好一点了",
    "晚安洛希，做个好梦~",
]

n_pause = 30
history = {'x': [], 'y': []}
stimulus_times = []
t = 0

for text in user_inputs:
    vad = analyzer.analyze(text)
    fb = [vad[0]*2-1, vad[1]*2-1, vad[3]*2-1, (vad[1]-vad[3])*2-1]
    state = substrate.step(fb)
    history['x'].append(state['x'])
    history['y'].append(state['y'])
    stimulus_times.append(t)
    t += 1
    
    for _ in range(n_pause):
        state = substrate.step(np.zeros(4) * 0.05)
        history['x'].append(state['x'])
        history['y'].append(state['y'])
        t += 1

history['x'] = np.array(history['x'])
history['y'] = np.array(history['y'])

fig, axes = plt.subplots(4, 1, figsize=(16, 12), facecolor='#0d1b2a')
colors = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff']
labels = ['肾上腺素 (兴奋)', '多巴胺 (愉悦)', '皮质醇 (压力)', '血清素 (满足)']
short_texts = ["天气好！😄", "考试不错", "吵架了...", "好烦", "好一点了", "晚安~"]

for i in range(4):
    ax = axes[i]
    ax.set_facecolor('#0d1b2a')
    ax.plot(history['x'][:, i], color=colors[i], alpha=0.9, linewidth=1.2, label='快系统 x(t)')
    ax.plot(history['y'][:, i], color=colors[i], alpha=0.4, linewidth=0.8, linestyle='--', label='慢系统 y(t)')
    for st in stimulus_times:
        ax.axvline(x=st, color='white', alpha=0.2, linestyle=':', linewidth=0.5)
    ax.set_ylabel(labels[i], color='white', fontsize=11)
    ax.tick_params(colors='#888888')
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_color('#333333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', fontsize=8, facecolor='#1a1a2e', edgecolor='#333333', labelcolor='white')

ax = axes[0]
for st, txt in zip(stimulus_times, short_texts):
    ax.annotate(txt, (st, history['x'][st, 0]), textcoords="offset points", xytext=(5, 10),
                fontsize=9, color='white', alpha=0.8, fontfamily='sans-serif',
                arrowprops=dict(arrowstyle='->', color='white', alpha=0.3))

axes[0].set_title('洛希情绪外挂 — 对话驱动的混沌激素动力学', color='white', fontsize=14, pad=15)
axes[-1].set_xlabel('时间步', color='white', fontsize=10)

plt.tight_layout()
outpath = '/mnt/d/workspace/luojia-projects/affective-substrate/demo_dynamics.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='#0d1b2a')
plt.close()
print(f"Saved: {outpath}")
