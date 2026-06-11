from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(r"E:\Workspace\Qwen3.5-9B", trust_remote_code=True)
# Check special tokens
print("eos_token:", t.eos_token, t.eos_token_id)
print("pad_token:", t.pad_token, t.pad_token_id)
# Check chat template for thinking tags
msgs = [{"role": "user", "content": "hi"}]
r = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
print("Template (no think):", repr(r))
# Check with thinking
r2 = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
print("Template (think):", repr(r2))
# Check if </think> is in vocab
print("<think> id:", t.convert_tokens_to_ids("<think>"))
print("</think> id:", t.convert_tokens_to_ids("</think>"))
