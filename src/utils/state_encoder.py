import torch.nn as nn

class StateEncoder(nn.Module):
    """
    Maps a numeric state vector [B, state_dim]
    to prefix embeddings [B, prefix_len, hidden_size].
    """
    def __init__(self, state_dim: int, prefix_len: int, hidden_size: int):
        super().__init__()
        self.prefix_len = prefix_len
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, prefix_len * hidden_size),
        )

    def forward(self, state_vec):
        # state_vec: FloatTensor of shape [B, state_dim]
        flat = self.mlp(state_vec)  # [B, prefix_len * hidden_size]
        return flat.view(-1, self.prefix_len, self.hidden_size)

