"""
activation_steering_server.py
残差流注入服务器 — 修复版
"""

import json, sys, os, time, threading
import numpy as np
import torch
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from hopfield_memory import HopfieldMemory

MODEL_PATH = r"E:\Workspace\Qwen3.5-9B"
API_PORT = 11435
DTYPE = torch.bfloat16

# ============ 情感动力学 ============
class AffectiveDynamics:
    def __init__(self, n=4, n_res=300):
        from collections import deque
        self.n = n
        self.betas = np.array([0.2, 0.15, 0.1, 0.08])
        self.taus = np.array([10, 20, 30, 50])
        self.ns_arr = np.array([10.0, 10.0, 12.0, 8.0])
        self.gamma_0, self.kappa, self.epsilon = 0.1, 0.3, 0.01
        self.x = np.array([1.0, 0.8, 0.6, 0.5])
        self.y = np.array([0.5]*4)
        self.delay_buf = [deque([self.x[i]]*int(self.taus[i]), maxlen=int(self.taus[i])) for i in range(n)]
        rng = np.random.RandomState(42)
        W_r = rng.randn(n_res, n_res)
        mask = rng.random((n_res, n_res)) < 0.1
        W_r *= mask
        eig = np.linalg.eigvals(W_r)
        W_r *= 0.9 / (np.max(np.abs(eig)) + 1e-8)
        self.W_r, self.W_x = W_r, rng.randn(n_res, n)*0.5
        self.r = np.zeros(n_res)
        self.tau_r = 10.0
        self.turn_count = 0
        d = 4096
        raw = rng.randn(n, d)
        for i in range(n):
            v = raw[i].copy()
            for j in range(i):
                v -= np.dot(raw[i], raw[j])/(np.dot(raw[j],raw[j])+1e-8)*raw[j]
            raw[i] = v/(np.linalg.norm(v)+1e-8)
        self.directions = raw
        self.P = rng.randn(n_res, d)*0.01
        self.A, self.s = 0.1, 2.0
        self.x_base = np.ones(n)
        self.y_base = np.ones(n)*0.5

    def step(self, fb, dt=1.0):
        fb = np.array(fb[:self.n])
        dx = np.zeros(self.n)
        for i in range(self.n):
            xd = self.delay_buf[i][0]
            g = self.gamma_0 + self.kappa*self.y[i]
            dx[i] = self.betas[i]*xd/(1+xd**self.ns_arr[i]) - g*self.x[i] + fb[i]
            self.x[i] = np.clip(self.x[i]+dx[i]*dt, -5, 5)
            self.y[i] += self.epsilon*(np.abs(self.x[i])-self.y[i])*dt
            self.delay_buf[i].append(self.x[i])
        dr = -self.r/self.tau_r + np.tanh(self.W_r@self.r + self.W_x@self.x)
        self.r += dr*dt
        self.turn_count += 1
        return self.x.copy(), self.y.copy(), self.r.copy(), dx.copy()

    def compute_injection(self):
        a_s = self.A*np.tanh(self.s*(self.x-self.x_base))
        a_p = self.A*np.tanh(self.s*(self.y-self.y_base))
        inj = a_s@self.directions - a_p@self.directions + self.r@self.P
        norm = np.linalg.norm(inj)
        limit = 4096**0.5 * 0.15
        if norm > limit:
            inj *= limit/norm
        return inj

    def to_dict(self):
        return {'x':self.x.tolist(),'y':self.y.tolist(),
                'r_mean':float(self.r.mean()),'r_std':float(self.r.std()),
                'turn':self.turn_count}

# ============ VAD ============
LEX = {
    'excited':[1.5,0,0,0],'calm':[0,1.5,0,0],'happy':[0,0,1.5,0],
    'sad':[0,0,0,1.5],'开心':[0,0,1.5,0],'难过':[0,0,0,1.3],
    '生气':[1.5,0,0,0.5],'焦虑':[1.0,0,0,1.0],'平静':[0,1.5,0,0],
}
def analyze_vad(text):
    s = np.zeros(4)
    c = 0
    for w,v in LEX.items():
        if w in text.lower(): s += np.array(v); c += 1
    return (s/c).tolist() if c else [0.5]*4

# ============ 停止条件 ============
class ThinkStopper(StoppingCriteria):
    def __init__(self, input_len, think_end_id):
        self.input_len = input_len
        self.think_end_id = think_end_id
    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] <= self.input_len:
            return False
        last_token = input_ids[0, -1].item()
        return last_token == self.think_end_id

# ============ 模型管理 ============
class ModelManager:
    def __init__(self):
        self.model = self.tokenizer = None
        self.hooks = []
        self.dynamics = AffectiveDynamics()
        self.memory = HopfieldMemory(n_reservoir=300, n_hormones=4)
        self._injection = None
        self._active = False
        self._think_end_id = None

    def load(self):
        print(f"Loading {MODEL_PATH}...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=DTYPE, device_map="auto", trust_remote_code=True)
        self.model.eval()
        self._think_end_id = self.tokenizer.convert_tokens_to_ids("</think>")
        print(f"Loaded in {time.time()-t0:.1f}s on {self.model.device}")
        self._register_hooks()

    def _register_hooks(self):
        for idx in [15, 18, 21, 24]:
            layer = self.model.model.layers[idx]
            self.hooks.append(layer.register_forward_hook(self._hook_fn))
            print(f"  Hook on layer {idx}")

    def _hook_fn(self, module, input, output):
        if not self._active or self._injection is None:
            return output
        inj = torch.tensor(self._injection, dtype=output[0].dtype, device=output[0].device)
        hidden = output[0] + inj.unsqueeze(0).unsqueeze(0)
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    @torch.no_grad()
    def generate(self, prompt, system="", max_tokens=512, temperature=0.7):
        # 1. 计算注入（含 Hopfield 回忆）
        injection = self.dynamics.compute_injection()
        recall_r, _, _, _ = self.memory.recall(self.dynamics.r)
        injection += recall_r @ self.dynamics.P
        # 再次约束
        norm = np.linalg.norm(injection)
        limit = 4096**0.5 * 0.15
        if norm > limit:
            injection *= limit/norm
        self._injection = injection

        # 2. 构造输入
        msgs = []
        if system:
            msgs.append({"role":"system","content":system})
        msgs.append({"role":"user","content":prompt})
        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        # 3. 生成（带停止条件）
        self._active = True
        t0 = time.time()
        stopping = StoppingCriteriaList([ThinkStopper(input_len, self._think_end_id)])
        outputs = self.model.generate(
            **inputs, max_new_tokens=max_tokens, temperature=temperature,
            do_sample=True, top_p=0.9, stopping_criteria=stopping,
            eos_token_id=self.tokenizer.eos_token_id)
        gen_time = time.time()-t0
        self._active = False

        # 4. 解码（只取新生成的token）
        new_tokens = outputs[0][input_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # 5. 更新动力学
        vad = analyze_vad(response)
        x, y, r, dx = self.dynamics.step(vad)

        # 6. Hopfield 存储
        if self.memory.should_store(dx, self.dynamics.turn_count):
            self.memory.store(r, x, y, self.dynamics.turn_count, response[:200])

        return {'response':response, 'vad':vad, 'state':self.dynamics.to_dict(),
                'gen_time':gen_time, 'injection_norm':float(np.linalg.norm(injection))}

    def reset(self):
        self.dynamics = AffectiveDynamics()
        self.memory = HopfieldMemory(n_reservoir=300, n_hormones=4)
        return self.dynamics.to_dict()

# ============ API ============
mgr = ModelManager()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _j(self, d, s=200):
        self.send_response(s)
        self.send_header("Content-Type","application/json")
        self.send_header("Access-Control-Allow-Origin","*")
        self.end_headers()
        self.wfile.write(json.dumps(d, ensure_ascii=False).encode())
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin","*")
        self.send_header("Access-Control-Allow-Methods","GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers","Content-Type")
        self.end_headers()
    def do_GET(self):
        if self.path=="/health":
            self._j({"status":"ok","model":mgr.model is not None,"device":str(mgr.model.device) if mgr.model else None})
        elif self.path=="/state":
            self._j(mgr.dynamics.to_dict())
        elif self.path=="/reset":
            self._j({"status":"reset","state":mgr.reset()})
        elif self.path=="/memory":
            self._j({"stats":mgr.memory.get_stats(),"recent":mgr.memory.get_recent(5)})
        else:
            self._j({"error":"not found"},404)
    def do_POST(self):
        ln = int(self.headers.get("Content-Length",0))
        b = json.loads(self.rfile.read(ln)) if ln else {}
        if self.path=="/generate":
            r = mgr.generate(b.get("prompt",""), b.get("system","你是一个有情感的人工智能助手。"),
                            b.get("max_tokens",512), b.get("temperature",0.7))
            self._j(r)
        elif self.path=="/chat":
            msgs = b.get("messages",[])
            user = ""
            for m in reversed(msgs):
                if m.get("role")=="user": user=m["content"]; break
            r = mgr.generate(user, b.get("system","你是一个有情感的人工智能助手。"))
            self._j({"reply":r['response'],"vad":r['vad'],"state":r['state'],"gen_time":r['gen_time']})
        elif self.path=="/feedback":
            mgr.dynamics.step(b.get("vad",[0.5]*4))
            self._j({"state":mgr.dynamics.to_dict()})
        else:
            self._j({"error":"not found"},404)

def main():
    print("="*50)
    print("  Activation Steering Server v2")
    print("="*50)
    mgr.load()
    # 测试
    r = mgr.generate("你好","Be brief.",100)
    print(f"Test: {r['response'][:80]}")
    print(f"  time={r['gen_time']:.1f}s norm={r['injection_norm']:.2f}")

    class Serv(ThreadingMixIn, HTTPServer): daemon_threads=True
    srv = Serv(("0.0.0.0", API_PORT), Handler)
    print(f"\nAPI on port {API_PORT}")
    try: srv.serve_forever()
    except KeyboardInterrupt:
        srv.server_close()

if __name__=="__main__": main()
