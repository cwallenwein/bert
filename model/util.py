from torch import Tensor, nn


def init_xavier(
    linear: nn.Linear = None,
    weights: Tensor = None,
    bias: Tensor = None,
    embedding: nn.Embedding = None,
    sequential: nn.Sequential = None,
):
    if linear is not None:
        nn.init.xavier_uniform_(linear.weight)
        if linear.bias is not None:
            nn.init.constant_(linear.bias, 0)
    if embedding is not None:
        nn.init.xavier_uniform_(embedding.weight)
    if weights is not None:
        nn.init.xavier_uniform_(weights)
    if bias is not None:
        nn.init.constant_(bias, 0)
    if sequential is not None:
        for module in sequential.children():
            if isinstance(module, nn.Linear):
                init_xavier(linear=module)
