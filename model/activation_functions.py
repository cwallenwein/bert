import torch
from torch.nn import functional as F


class GatedLinearUnit(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(self, inputs):
        x, gate = inputs.chunk(2, dim=-1)
        return x * F.sigmoid(gate)
