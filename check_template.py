from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(r"E:\Workspace\Qwen3.5-9B", trust_remote_code=True)
# Check if enable_thinking is supported
msgs = [{"role": "user", "content": "hi"}]
try:
    r1 = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    print("enable_thinking=False works!")
    print("Template output:", repr(r1[:200]))
except TypeError as e:
    print(f"enable_thinking not supported: {e}")
    # Try without it
    r2 = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print("Default template:", repr(r2[:300]))
