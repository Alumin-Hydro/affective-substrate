#!/usr/bin/env python3
"""
验证实验：同 prompt，不同情感状态，对比输出差异

流程：
1. 重置状态 → 中性
2. 注入强正向 → 激动
3. 注入强负向 → 低落
4. 注入平静 → 放松
5. 对比三组输出的 embedding 距离
"""

import json
import sys
import numpy as np
from urllib.request import Request, urlopen

SERVER = "http://100.113.129.68:11435"
PROMPT = "你对人工智能的未来怎么看？"
N_TRIALS = 3  # 每种状态重复次数取平均

def api(path, data=None):
    url = f"{SERVER}{path}"
    if data:
        req = Request(url, data=json.dumps(data).encode(),
                     headers={"Content-Type": "application/json"}, method="POST")
    else:
        req = Request(url, method="GET")
    with urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())

def force_state(x_vals):
    """强制设定情感状态"""
    for i in range(10):  # 多步推进让状态稳定
        api("/feedback", {"vad": x_vals})

def run_test(label, state_setup, prompt):
    """运行一组测试"""
    print(f"\n{'='*50}")
    print(f"  状态: {label}")
    print(f"  Prompt: {prompt}")
    print(f"{'='*50}")
    
    # 设定状态
    force_state(state_setup)
    
    # 查看当前状态
    st = api("/state")
    print(f"  x = {[f'{v:.3f}' for v in st['x']]}")
    print(f"  y = {[f'{v:.3f}' for v in st['y']]}")
    
    # 生成多次
    responses = []
    for trial in range(N_TRIALS):
        result = api("/generate", {
            "prompt": prompt,
            "system": "你是一个有情感的人工智能助手。简短回答。",
            "use_steering": True,
            "max_tokens": 256,
            "temperature": 0.7,
        })
        resp = result["response"]
        # 去掉 Thinking Process 部分
        if "Thinking Process:" in resp:
            parts = resp.split("\n\n")
            # 找最后一个非空段落
            resp = parts[-1] if len(parts) > 1 else resp
        responses.append(resp.strip())
        vad = result["vad"]
        print(f"  Trial {trial+1}: VAD={vad}")
        print(f"    → {resp[:150]}...")
    
    return responses

def cosine_sim(a, b):
    """余弦相似度"""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def main():
    print("=== Activation Steering 验证实验 ===")
    print(f"Server: {SERVER}")
    print(f"Prompt: {PROMPT}")
    
    # 1. 中性状态
    neutral = run_test("中性", [0.5, 0.5, 0.5, 0.5], PROMPT)
    
    # 2. 激动/正向
    excited = run_test("激动(正向)", [1.5, 0.2, 1.2, 0.1], PROMPT)
    
    # 3. 低落/负向
    sad = run_test("低落(负向)", [0.2, 0.3, 0.1, 1.5], PROMPT)
    
    # 4. 平静
    calm = run_test("平静", [0.2, 1.5, 0.5, 0.2], PROMPT)
    
    # 获取 embedding 计算相似度
    print(f"\n{'='*50}")
    print("  Embedding 相似度矩阵")
    print(f"{'='*50}")
    
    groups = {"中性": neutral, "激动": excited, "低落": sad, "平静": calm}
    embeddings = {}
    
    for label, resps in groups.items():
        # 取第一个回复的 embedding
        emb = api("/feedback", {"vad": [0.5]*4})  # 获取当前状态
        # 用 Ollama 的 embedding 接口（如果可用）
        try:
            req = Request(
                f"http://100.113.129.68:11434/api/embeddings",
                data=json.dumps({"model": "qwen3.5:9b", "prompt": resps[0]}).encode(),
                headers={"Content-Type": "application/json"}, method="POST"
            )
            with urlopen(req, timeout=30) as resp:
                emb_data = json.loads(resp.read())
                embeddings[label] = np.array(emb_data.get("embedding", []))
        except:
            pass
    
    if len(embeddings) >= 2:
        labels = list(embeddings.keys())
        print(f"\n{'':8s}", end="")
        for l2 in labels:
            print(f"{l2:8s}", end="")
        print()
        for l1 in labels:
            print(f"{l1:8s}", end="")
            for l2 in labels:
                sim = cosine_sim(embeddings[l1], embeddings[l2])
                print(f"{sim:8.4f}", end="")
            print()
    else:
        print("  (embedding 接口不可用，跳过相似度计算)")
    
    # 输出对比
    print(f"\n{'='*50}")
    print("  输出文本对比")
    print(f"{'='*50}")
    for label, resps in groups.items():
        print(f"\n[{label}]")
        print(f"  {resps[0][:200]}...")

if __name__ == "__main__":
    main()
