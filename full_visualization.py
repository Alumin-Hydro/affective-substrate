"""
full_visualization.py
======================
综合可视化：连服务器采集数据，生成完整图表

输出:
  1. 情感状态时间序列 (x, y)
  2. 相图 (x vs y)
  3. 储备池 PCA
  4. 注入向量热力图
  5. 记忆存储时间线
  6. 延迟核对比
"""

import json, os, sys
import numpy as np
from urllib.request import Request, urlopen

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from matplotlib.gridspec import GridSpec

SERVER = "http://100.113.129.68:11435"
OUTPUT_DIR = "/mnt/d/workspace/luojia-projects/affective-substrate"

def api(path, data=None):
    url = f"{SERVER}{path}"
    if data:
        req = Request(url, data=json.dumps(data).encode(),
                     headers={"Content-Type":"application/json"}, method="POST")
    else:
        req = Request(url, method="GET")
    with urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())

def collect_data():
    """采集多轮对话数据"""
    # 重置
    api("/reset")
    
    # 定义对话序列：模拟不同情感状态的交互
    scenarios = [
        # (反馈VAD, 用户输入)
        ([1.5, 0.2, 1.2, 0.1], "今天阳光真好，我心情很棒！"),
        ([1.3, 0.3, 1.0, 0.1], "和朋友出去玩了一整天"),
        ([0.5, 0.5, 0.5, 0.5], "你觉得人生意义是什么？"),
        ([0.3, 0.3, 0.2, 1.2], "最近压力好大，有点焦虑"),
        ([0.2, 0.2, 0.1, 1.5], "感觉什么都提不起兴趣"),
        ([0.2, 1.5, 0.5, 0.2], "我想安静地待一会儿"),
        ([0.3, 1.2, 0.6, 0.3], "泡了杯茶，看了会儿书"),
        ([1.0, 0.5, 0.8, 0.3], "突然有了个新想法！"),
        ([1.2, 0.3, 1.0, 0.2], "这个项目太有趣了"),
        ([0.5, 0.5, 0.5, 0.5], "晚安"),
    ]
    
    all_states = []
    all_responses = []
    all_vad = []
    all_injection_norms = []
    
    for vad_fb, prompt in scenarios:
        # 推送反馈
        for _ in range(10):
            api("/feedback", {"vad": vad_fb})
        
        # 生成
        result = api("/generate", {
            "prompt": prompt,
            "system": "你是一个有情感的人工智能助手。",
            "max_tokens": 150,
        })
        
        state = result["state"]
        all_states.append(state)
        all_responses.append(result["response"][:100])
        all_vad.append(result["vad"])
        all_injection_norms.append(result["injection_norm"])
    
    return all_states, all_responses, all_vad, all_injection_norms

def plot_all(states, responses, vad_list, inj_norms):
    """生成综合图表"""
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Affective Substrate — Full Visualization", fontsize=16, fontweight='bold')
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    x_data = np.array([s['x'] for s in states])
    y_data = np.array([s['y'] for s in states])
    turns = np.arange(len(states))
    labels = ["Arousal", "Calm", "Positive", "Negative"]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    
    # 1. 快系统 x(t)
    ax1 = fig.add_subplot(gs[0, 0])
    for i in range(4):
        ax1.plot(turns, x_data[:, i], label=labels[i], color=colors[i], linewidth=2)
    ax1.set_title("Fast System x(t) — Sympathetic", fontsize=11)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("Turn")
    
    # 2. 慢系统 y(t)
    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(4):
        ax2.plot(turns, y_data[:, i], label=labels[i], color=colors[i], linewidth=2)
    ax2.set_title("Slow System y(t) — Parasympathetic", fontsize=11)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Turn")
    
    # 3. 注入向量范数
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(turns, inj_norms, color='#f39c12', alpha=0.7)
    ax3.set_title("Injection Norm ||α·v||", fontsize=11)
    ax3.set_xlabel("Turn")
    ax3.grid(True, alpha=0.3)
    
    # 4. 相图 (Arousal vs Positive)
    ax4 = fig.add_subplot(gs[1, 0])
    scatter = ax4.scatter(x_data[:, 0], x_data[:, 2], c=turns, cmap='viridis', s=80, alpha=0.8, edgecolors='white')
    for i in range(len(turns)):
        ax4.annotate(str(i), (x_data[i, 0], x_data[i, 2]), fontsize=7, ha='center', va='bottom')
    ax4.set_xlabel("Arousal (x₀)")
    ax4.set_ylabel("Positive (x₂)")
    ax4.set_title("Phase Portrait — Arousal vs Positive", fontsize=11)
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label="Turn")
    
    # 5. 相图 (Arousal vs Negative)
    ax5 = fig.add_subplot(gs[1, 1])
    scatter2 = ax5.scatter(x_data[:, 0], x_data[:, 3], c=turns, cmap='plasma', s=80, alpha=0.8, edgecolors='white')
    for i in range(len(turns)):
        ax5.annotate(str(i), (x_data[i, 0], x_data[i, 3]), fontsize=7, ha='center', va='bottom')
    ax5.set_xlabel("Arousal (x₀)")
    ax5.set_ylabel("Negative (x₃)")
    ax5.set_title("Phase Portrait — Arousal vs Negative", fontsize=11)
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax5, label="Turn")
    
    # 6. 快慢系统时间尺度分离
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(turns, x_data[:, 0], label='x[0] (fast)', color='#e74c3c', linewidth=2)
    ax6.plot(turns, y_data[:, 0], label='y[0] (slow)', color='#e74c3c', linewidth=2, linestyle='--', alpha=0.6)
    ax6.plot(turns, x_data[:, 3], label='x[3] (slow hormone)', color='#9b59b6', linewidth=2)
    ax6.plot(turns, y_data[:, 3], label='y[3] (slow)', color='#9b59b6', linewidth=2, linestyle='--', alpha=0.6)
    ax6.set_title("Time Scale Separation", fontsize=11)
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlabel("Turn")
    
    # 7. 对话内容 + 情感状态 (表格)
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    table_data = []
    for i, (resp, s) in enumerate(zip(responses, states)):
        x_str = ", ".join([f"{v:.1f}" for v in s['x']])
        table_data.append([str(i), resp[:60]+"...", x_str])
    
    table = ax7.table(
        cellText=table_data,
        colLabels=["#", "Response (truncated)", "x = [arousal, calm, positive, negative]"],
        loc='center', cellLoc='left',
        colWidths=[0.05, 0.55, 0.4]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    ax7.set_title("Conversation Log with Emotional State", fontsize=11, pad=20)
    
    out = os.path.join(OUTPUT_DIR, "full_visualization.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")
    return out

def main():
    print("Collecting data from server...")
    states, responses, vad_list, inj_norms = collect_data()
    print(f"Collected {len(states)} turns")
    
    print("Generating visualization...")
    out = plot_all(states, responses, vad_list, inj_norms)
    
    # 也把数据存下来
    data_out = os.path.join(OUTPUT_DIR, "experiment_data.json")
    with open(data_out, 'w', encoding='utf-8') as f:
        json.dump({
            'states': states,
            'responses': responses,
            'vad': vad_list,
            'injection_norms': inj_norms,
        }, f, ensure_ascii=False, indent=2)
    print(f"Data saved: {data_out}")

if __name__ == "__main__":
    main()
