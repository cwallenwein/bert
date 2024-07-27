import torch
from torch import nn
from torch.nn import functional as F
from model.config import BertConfig


class GatedLinearUnit(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.feed_forward_intermediate_size // 2)
        self.output = nn.Linear(config.d_model, config.feed_forward_intermediate_size // 2)

    def forward(self, x):
        a = self.gate(x)
        b = self.output(x)
        return a * F.sigmoid(b)


class GatedLinearUnit1(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.d_model)

    def forward(self, x):
        return x * F.sigmoid(self.gate(x))


class GatedLinearUnit2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        x, gate = inputs.chunk(2, dim=-1)
        return x * F.sigmoid(gate)
