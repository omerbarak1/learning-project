import torch
from src.utils.state_vector import state_to_tensor
from src.utils.state_encoder import StateEncoder
from src.student_state import StudentState
        

# 1) Build a sample state tensor
student = StudentState()
state_dict = student.state
state_vec = state_to_tensor(state_dict).unsqueeze(0)   # [1, state_dim]

# 2) Instantiate the encoder
state_dim   = state_vec.size(-1)
prefix_len  = 10      # e.g. 10 prefix tokens
hidden_size = 768     # GPT-2 hidden size
encoder     = StateEncoder(state_dim, prefix_len, hidden_size)

# 3) Forward pass
prefix_embeds = encoder(state_vec)                    # [1, 10, 768]

print("state_dim:", state_dim)
print("prefix_embeds.shape:", prefix_embeds.shape)
