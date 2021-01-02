from model.components import Embed
import torch
from torch import nn

model = Embed(1000, 200, 200)

u = torch.randint(1, 1000, (4, 10))
v = torch.randint(1, 1000, (4, 7))
u_mask = torch.tensor(
    [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]
)
v_mask = torch.tensor(
    [[1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 0, 0]]
)
u = u * u_mask
u_length = torch.tensor([10, 6, 8, 4])
v = v * v_mask
v_length = torch.tensor([7, 7, 6, 4])

print(model(u, u_length).detach().numpy())