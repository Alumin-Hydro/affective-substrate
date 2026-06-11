"""
affective_llm_server.py
========================
5090 端推理服务器

两种模式：
  1. Ollama 模式（快速，无 hook）— 通过 prompt 注入情感状态
  2. Full 模式（transformers + nnsight）— 残差流注入

API:
  POST /generate    — 生成文本
  GET  /state       — 获取当前情感状态
  POST /feedback    — 提供反馈更新状态
  GET  /health      — 健康检查
"""

import json
import sys
import os
import time
import threading
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import URLError

# === 配置 ===
OLLAMA_URL = "http://127.0.0.1:11434"
MODEL_NAME = "qwen3.5:9b"
API_PORT = 11435  # 我们的API端口（避开Ollama的11434）
DEVICE = "cuda"

# === 情感状态（进程内维护） ===
class EmotionalState:
    """情感状态管理器"""
    
    def __init__(self, n_hormones=4):
        self.n = n_hormones
        self.x = np.ones(n_hormones)      # 快系统
        self.y = np.ones(n_hormones) * 0.5  # 慢系统
        self.turn_count = 0
        self.history = []  # 最近N轮的(x, y)快照
        
    def feedback(self, vad_scores):
        """接收 VAD 反馈，推进状态"""
        fb = np.array(vad_scores[:self.n])
        # 简化版动力学（完整版在 core.py）
        tau = np.array([10, 20, 30, 50])
        beta = np.array([0.2, 0.15, 0.1, 0.08])
        gamma_0 = 0.1
        kappa = 0.3
        epsilon = 0.01
        
        for i in range(self.n):
            # 简化 Mackey-Glass（用移动平均代替精确时延）
            x_delayed = self.x[i]  # 简化
            gamma_eff = gamma_0 + kappa * self.y[i]
            dx = beta[i] * x_delayed / (1 + x_delayed**10) - gamma_eff * self.x[i] + fb[i]
            self.x[i] += dx * 0.1
            self.x[i] = np.clip(self.x[i], -5, 5)
            
            dy = epsilon * (np.abs(self.x[i]) - self.y[i])
            self.y[i] += dy * 0.1
            
        self.turn_count += 1
        self.history.append({
            'x': self.x.copy().tolist(),
            'y': self.y.copy().tolist(),
            'turn': self.turn_count,
        })
        # 只保留最近100轮
        if len(self.history) > 100:
            self.history = self.history[-100:]
    
    def get_prompt_context(self):
        """生成情感状态的 prompt 上下文"""
        labels = ["激动", "平静", "积极", "消极"]
        parts = []
        for i in range(self.n):
            level = self.x[i]
            if level > 1.5:
                parts.append(f"{labels[i]}度高")
            elif level < 0.5:
                parts.append(f"{labels[i]}度低")
        if not parts:
            return ""
        return f"[情感状态：{', '.join(parts)}]"
    
    def to_dict(self):
        return {
            'x': self.x.tolist(),
            'y': self.y.tolist(),
            'turn': self.turn_count,
            'prompt_context': self.get_prompt_context(),
        }


# === Ollama API 调用 ===
def ollama_generate(prompt, system_prompt="", temperature=0.7, max_tokens=2048):
    """调用 Ollama API 生成"""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }
    req = Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            return result.get("response", "")
    except Exception as e:
        return f"[Ollama error: {e}]"


def ollama_embed(text):
    """获取 Ollama 嵌入（用于反馈分析）"""
    payload = {
        "model": MODEL_NAME,
        "prompt": text,
    }
    req = Request(
        f"{OLLAMA_URL}/api/embeddings",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            return np.array(result.get("embedding", []))
    except:
        return None


# === VAD 简易分析 ===
EMOTION_LEXICON = {
    # 正向 arousal
    'excited': [1.5, 0, 0, 0], 'thrilled': [1.5, 0, 0, 0],
    'passionate': [1.3, 0, 0, 0], 'intense': [1.2, 0, 0, 0],
    # 正向 calm
    'calm': [0, 1.5, 0, 0], 'peaceful': [0, 1.5, 0, 0],
    'relaxed': [0, 1.3, 0, 0], 'serene': [0, 1.5, 0, 0],
    # 正向 positive
    'happy': [0, 0, 1.5, 0], 'joyful': [0, 0, 1.5, 0],
    'glad': [0, 0, 1.2, 0], 'pleased': [0, 0, 1.2, 0],
    'good': [0, 0, 1.0, 0], 'great': [0, 0, 1.3, 0],
    # 正向 negative
    'sad': [0, 0, 0, 1.5], 'unhappy': [0, 0, 0, 1.3],
    'depressed': [0, 0, 0, 1.5], 'miserable': [0, 0, 0, 1.5],
    'bad': [0, 0, 0, 1.0], 'terrible': [0, 0, 0, 1.5],
    # 中文
    '开心': [0, 0, 1.5, 0], '高兴': [0, 0, 1.3, 0],
    '难过': [0, 0, 0, 1.3], '伤心': [0, 0, 0, 1.5],
    '生气': [1.5, 0, 0, 0.5], '愤怒': [1.5, 0, 0, 0.8],
    '平静': [0, 1.5, 0, 0], '放松': [0, 1.3, 0, 0],
    '焦虑': [1.0, 0, 0, 1.0], '紧张': [1.2, 0, 0, 0.5],
}

def analyze_text_vad(text):
    """简易文本 VAD 分析"""
    scores = np.zeros(4)
    words_found = 0
    text_lower = text.lower()
    
    for word, vec in EMOTION_LEXICON.items():
        if word in text_lower:
            scores += np.array(vec)
            words_found += 1
    
    if words_found > 0:
        scores = scores / words_found
    else:
        # 无情感词时，用长度和标点估算 arousal
        exclaim = text.count('!') + text.count('！')
        quest = text.count('?') + text.count('？')
        scores[0] = min(exclaim * 0.3, 1.5)  # arousal
        scores[1] = max(1.0 - exclaim * 0.2, 0.3)  # calm
        scores[2] = 0.5  # neutral positive
        scores[3] = quest * 0.2  # slight negative from questions
    
    return scores.tolist()


# === 全局状态 ===
state = EmotionalState(n_hormones=4)


# === HTTP API ===
class APIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # 静默日志
    
    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok", "model": MODEL_NAME, "turn": state.turn_count})
        elif self.path == "/state":
            self._send_json(state.to_dict())
        else:
            self._send_json({"error": "not found"}, 404)
    
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length)) if content_length > 0 else {}
        
        if self.path == "/generate":
            prompt = body.get("prompt", "")
            system = body.get("system", "")
            use_injection = body.get("use_injection", True)
            
            # 注入情感上下文到 system prompt
            if use_injection:
                ctx = state.get_prompt_context()
                if ctx:
                    system = f"{system}\n\n{ctx}" if system else ctx
            
            response = ollama_generate(prompt, system)
            
            # 分析输出，更新情感状态
            vad = analyze_text_vad(response)
            state.feedback(vad)
            
            self._send_json({
                "response": response,
                "vad": vad,
                "state": state.to_dict(),
            })
            
        elif self.path == "/feedback":
            vad = body.get("vad", [0.5, 0.5, 0.5, 0.5])
            state.feedback(vad)
            self._send_json({"state": state.to_dict()})
            
        elif self.path == "/chat":
            # 完整对话接口（带历史）
            messages = body.get("messages", [])
            system = body.get("system", "你是一个有情感的人工智能助手。")
            
            # 注入情感状态
            ctx = state.get_prompt_context()
            if ctx:
                system = f"{system}\n\n{ctx}"
            
            # 调用 Ollama chat API
            payload = {
                "model": MODEL_NAME,
                "messages": [{"role": "system", "content": system}] + messages,
                "stream": False,
                "options": {"temperature": 0.7},
            }
            req = Request(
                f"{OLLAMA_URL}/api/chat",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            try:
                with urlopen(req, timeout=120) as resp:
                    result = json.loads(resp.read())
                    reply = result.get("message", {}).get("content", "")
            except Exception as e:
                reply = f"[Error: {e}]"
            
            # 分析并更新
            vad = analyze_text_vad(reply)
            state.feedback(vad)
            
            self._send_json({
                "reply": reply,
                "vad": vad,
                "state": state.to_dict(),
            })
        else:
            self._send_json({"error": "not found"}, 404)


def main():
    print(f"=== Affective LLM Server ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Ollama: {OLLAMA_URL}")
    print(f"API port: {API_PORT}")
    print(f"Device: {DEVICE}")
    
    # 测试 Ollama 连接
    try:
        req = Request(f"{OLLAMA_URL}/api/tags")
        with urlopen(req, timeout=5) as resp:
            models = json.loads(resp.read())
            print(f"Ollama models: {[m['name'] for m in models.get('models', [])]}")
    except Exception as e:
        print(f"WARNING: Ollama not reachable: {e}")
        print("Server will start but generation will fail.")
    
    server = HTTPServer(("0.0.0.0", API_PORT), APIHandler)
    print(f"\nListening on port {API_PORT}...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
