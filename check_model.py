from transformers import AutoConfig
c = AutoConfig.from_pretrained(r"E:\Workspace\Qwen3.5-9B")
print(f"model_type={c.model_type}")
print(f"arch={c.architectures}")
# Multimodal: text config is nested
if hasattr(c, 'text_config'):
    tc = c.text_config
    print(f"text_hidden_size={tc.hidden_size}")
    print(f"text_num_layers={tc.num_hidden_layers}")
    print(f"text_num_heads={tc.num_attention_heads}")
    print(f"text_head_dim={tc.head_dim}")
else:
    print(f"hidden_size={c.hidden_size}")
    print(f"num_layers={c.num_hidden_layers}")
