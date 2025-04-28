import torch
from src.utils.flatten import flatten_dict

def state_to_tensor(state_dict: dict) -> torch.Tensor:
    flat = flatten_dict(state_dict)
    keys = sorted(flat.keys())

    values = []
    for k in keys:
        v = flat[k]
        if isinstance(v, (int, float)):
            values.append(float(v))
        elif isinstance(v, (list, tuple)):
            for x in v:
                if isinstance(x, (int, float)):
                    values.append(float(x))
        else:
            continue

    return torch.tensor(values, dtype=torch.float32)
