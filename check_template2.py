from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(r"E:\Workspace\Qwen3.5-9B", trust_remote_code=True)
msgs = [{"role": "system", "content": "你是一个有情感的人工智能助手。"}, {"role": "user", "content": "/no_think 你好"}]
r = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print(repr(r))
