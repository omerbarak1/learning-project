from src.student_state import StudentState
import torch

student = StudentState()
vec = torch.tensor(student.to_vector(), dtype=torch.float32)

print("State vector shape:", vec.shape)
print("First few values:", vec[:10])
