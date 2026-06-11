#!/usr/bin/env python3
"""
affective_client.py
====================
本地客户端：连接 5090 推理服务器，管理完整 AffectiveSubstrate

运行方式：
  python affective_client.py [--server 100.113.129.68:11435]

功能：
  - 交互式对话
  - 实时情感状态可视化（终端）
  - 状态历史记录 + 图表导出
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from urllib.request import Request, urlopen
from urllib.error import URLError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import AffectiveSubstrate, HormoneConfig, ReservoirConfig


class AffectiveClient:
    """连接 5090 服务器的客户端"""
    
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")
        self.substrate = AffectiveSubstrate(
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
        self.history = []  # (turn, x, y, vad, response)
        
    def _api(self, path: str, data: dict = None) -> dict:
        """调用服务器 API"""
        url = f"{self.server_url}{path}"
        if data is not None:
            req = Request(
                url,
                data=json.dumps(data).encode(),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
        else:
            req = Request(url, method="GET")
        
        with urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    
    def chat(self, user_input: str, system_prompt: str = "你是一个有情感的人工智能助手。") -> dict:
        """
        发送消息并获取回复
        
        完整流程：
        1. 推送本地 substrate 状态到服务器
        2. 服务器生成回复
        3. 分析回复，更新本地 substrate
        """
        # 获取当前 substrate 状态
        state = self.substrate.step(np.zeros(4))  # 空反馈，纯推进
        
        # 发送到服务器（附带情感上下文）
        response = self._api("/chat", {
            "messages": [{"role": "user", "content": user_input}],
            "system": system_prompt,
        })
        
        reply = response.get("reply", "")
        vad = response.get("vad", [0.5, 0.5, 0.5, 0.5])
        
        # 用服务器返回的 VAD 更新本地 substrate
        state = self.substrate.step(np.array(vad))
        
        # 记录历史
        self.history.append({
            'turn': len(self.history),
            'user': user_input,
            'response': reply,
            'x': state['x'].tolist(),
            'y': state['y'].tolist(),
            'vad': vad,
            'gamma_eff': state['gamma_eff'].tolist(),
        })
        
        return {
            'reply': reply,
            'vad': vad,
            'state': state,
        }
    
    def print_state(self, state: dict):
        """终端打印当前情感状态"""
        x = state['x']
        y = state['y']
        gamma = state['gamma_eff']
        labels = ["激动", "平静", "积极", "消极"]
        
        print("\n┌─ 情感状态 ─────────────────────────┐")
        for i in range(len(x)):
            bar_x = "█" * int(abs(x[i]) * 5) + "░" * (10 - int(abs(x[i]) * 5))
            bar_y = "█" * int(abs(y[i]) * 5) + "░" * (10 - int(abs(y[i]) * 5))
            direction = "+" if x[i] > 0 else "-"
            print(f"│ {labels[i]:2s}  x={direction}{bar_x} {x[i]:+.2f}  y={bar_y} {y[i]:.2f}  γ={gamma[i]:.2f}")
        print(f"│ 储备池: mean={state['r'].mean():.3f} std={state['r'].std():.3f}")
        print(f"└──────────────────────────────────┘")
    
    def plot_history(self, save_path: str = None):
        """导出历史图表"""
        if not self.history:
            print("No history to plot.")
            return
            
        try:
            import matplotlib.pyplot as plt
            matplotlib_available = True
        except ImportError:
            print("matplotlib not available, skipping plot.")
            return
        
        turns = [h['turn'] for h in self.history]
        x_data = np.array([h['x'] for h in self.history])
        y_data = np.array([h['y'] for h in self.history])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle("Affective Substrate — 对话历史", fontsize=14)
        
        labels = ["激动", "平静", "积极", "消极"]
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
        
        # 快系统
        ax = axes[0, 0]
        for i in range(4):
            ax.plot(turns, x_data[:, i], label=labels[i], color=colors[i], alpha=0.8)
        ax.set_title("快系统 x(t) — 交感")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 慢系统
        ax = axes[0, 1]
        for i in range(4):
            ax.plot(turns, y_data[:, i], label=labels[i], color=colors[i], alpha=0.8)
        ax.set_title("慢系统 y(t) — 副交感")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 相图 (x vs y for first hormone)
        ax = axes[1, 0]
        ax.scatter(x_data[:, 0], y_data[:, 0], c=turns, cmap='viridis', s=10, alpha=0.6)
        ax.set_xlabel("x (激动)")
        ax.set_ylabel("y (累积)")
        ax.set_title("相图 — 激动维度")
        ax.grid(True, alpha=0.3)
        
        # VAD 历史
        ax = axes[1, 1]
        vad_data = np.array([h['vad'] for h in self.history])
        for i in range(4):
            ax.plot(turns, vad_data[:, i], label=labels[i], color=colors[i], alpha=0.6, linestyle='--')
        ax.set_title("VAD 反馈")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path is None:
            save_path = os.path.join(os.path.dirname(__file__), "conversation_history.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    def run_interactive(self):
        """交互式对话循环"""
        print("=" * 50)
        print("  Affective LLM — 混沌动力学情感调制")
        print("  输入 'quit' 退出, 'plot' 导出图表, 'state' 查看状态")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            
            if not user_input:
                continue
            if user_input.lower() == 'quit':
                break
            if user_input.lower() == 'plot':
                self.plot_history()
                continue
            if user_input.lower() == 'state':
                state = self.substrate.step(np.zeros(4))
                self.print_state(state)
                continue
            
            result = self.chat(user_input)
            print(f"\nAI: {result['reply']}")
            self.print_state(result['state'])


def main():
    parser = argparse.ArgumentParser(description="Affective LLM Client")
    parser.add_argument("--server", default="100.113.129.68:11435",
                       help="Server address (default: 100.113.129.68:11435)")
    args = parser.parse_args()
    
    server_url = f"http://{args.server}"
    
    # 测试连接
    try:
        req = Request(f"{server_url}/health")
        with urlopen(req, timeout=5) as resp:
            info = json.loads(resp.read())
            print(f"Connected to server: {info}")
    except Exception as e:
        print(f"Cannot connect to server at {server_url}: {e}")
        print("Make sure the server is running on the 5090 machine.")
        sys.exit(1)
    
    client = AffectiveClient(server_url)
    client.run_interactive()


if __name__ == "__main__":
    main()
