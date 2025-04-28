import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.student_state import StudentState
from src.utils.state_vector import state_to_tensor
from src.utils.state_encoder import StateEncoder

# 1) Load GPT-2 and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model     = AutoModelForCausalLM.from_pretrained("gpt2")
# use eos token as pad
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 2) Encode your state
student    = StudentState()
raw_vec    = torch.tensor(student.to_vector(), dtype=torch.float32).unsqueeze(0).to(device)  # [1, state_dim]
encoder    = StateEncoder(
    state_dim=raw_vec.size(-1),
    prefix_len=10,
    hidden_size=model.config.hidden_size
).to(device)
prefix_embs = encoder(raw_vec)  # [1, prefix_len, hidden_size]

# 3) Prepare a dummy start token
bos_id    = tokenizer.bos_token_id or tokenizer.eos_token_id
text_ids  = torch.tensor([[bos_id]], device=device)   # [1,1]
text_embs = model.transformer.wte(text_ids)           # [1,1,hidden_size]

# 4) Concatenate prefix + text embeddings
inputs_embeds = torch.cat([prefix_embs, text_embs], dim=1)  # [1, prefix_len+1, hidden_size]
attention_mask = torch.ones(
  inputs_embeds.size()[:2],    # shape [1, P+1]
  dtype=torch.long,
  device=device
)

# 5) Generate a prompt continuation
out = model.generate(
    inputs_embeds=inputs_embeds, attention_mask=attention_mask,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id,
)

# 6) Decode only the newly generated tokens
generated = tokenizer.decode(
    out[0, prefix_embs.size(1):],
    skip_special_tokens=True
)

# inputs_embeds: [1, P+1, H]
attention_mask = torch.ones(inputs_embeds.size()[:2], dtype=torch.long, device=device)

out = model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=attention_mask,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id,
)


print("STATE VECTOR:", raw_vec)
print("GENERATED PROMPT:", generated)
