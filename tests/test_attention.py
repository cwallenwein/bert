import random
import warnings

import pytest
import torch

from bert.model.attention import MultiHeadSelfAttention


def is_gpu_available() -> bool:
    return torch.cuda.is_available()


def is_flash_attention_available() -> bool:
    """Check if flash attention is available."""
    try:
        import flash_attn  # noqa: F401  #type: ignore

        return True
    except ImportError:
        warnings.warn(
            "Can't test flash attention as it is not available. \
Please install flash_attn",
        )
        return False


def is_pytorch_attention_available() -> bool:
    """Check if PyTorch's scaled dot product attention is available."""
    has_sdpa = hasattr(torch.nn.functional, "scaled_dot_product_attention")
    if not has_sdpa:
        warnings.warn(
            "Can't test pytorch implementation as \
torch.nn.functional.scaled_dot_product_attention is not available. \
Please upgrade to torch >= 2.0",
        )
    return has_sdpa


ATTENTION_IMPLEMENTATIONS = ["from-scratch"]
if is_pytorch_attention_available():
    ATTENTION_IMPLEMENTATIONS.append("pytorch")
# if is_gpu_available() and is_flash_attention_available():
#     ATTENTION_IMPLEMENTATIONS.append("flash-attn")


@pytest.mark.parametrize(
    "attention_implementation",
    ATTENTION_IMPLEMENTATIONS,
)
@pytest.mark.parametrize("with_random_attention_mask", [True, False])
def test_attention_implementations(
    attention_implementation: str, with_random_attention_mask: bool
):
    batch_size = 2
    seq_len = 8
    d_model = 4

    device = "cuda" if attention_implementation == "flash-attn" else "cpu"

    inputs = torch.randn(
        (batch_size, seq_len, d_model), dtype=torch.float16, device=device
    )
    if with_random_attention_mask:
        attention_mask = torch.ones(
            batch_size, seq_len, device=device, dtype=torch.bool
        )
        for sample_index in range(batch_size):
            attention_mask[sample_index, : random.randrange(seq_len)] = False
    else:
        attention_mask = None

    mha = MultiHeadSelfAttention(
        d_model=d_model,
        n_heads=4,
        bias=True,
        p_dropout=0.0,
        attention_implementation=attention_implementation,
    ).to(device=device, dtype=torch.float16)

    output = mha(inputs, attention_mask)

    assert output.shape == inputs.shape, (
        f"Output shape mismatch for {attention_implementation} "
        f"{'with' if with_random_attention_mask else 'without'} mask"
    )


@pytest.mark.parametrize("n_heads", [1, 4])
@pytest.mark.parametrize("p_dropout", [0.0, 0.2])
@pytest.mark.parametrize("with_random_attention_mask", [True, False])
def test_compare_pytorch_and_from_scratch_implementations(
    n_heads: int,
    p_dropout: float,
    with_random_attention_mask: bool,
):
    if not is_pytorch_attention_available():
        return

    print("p_dropout", p_dropout)

    batch_size = 2
    seq_len = 8
    d_model = 4

    inputs = torch.randn(
        batch_size, n_heads, seq_len, d_model, device="cpu", dtype=torch.float32
    )

    if with_random_attention_mask:
        attention_mask = torch.ones(batch_size, seq_len, device="cpu", dtype=torch.bool)
        for sample_index in range(batch_size):
            attention_mask[sample_index, : random.randrange(seq_len)] = False
    else:
        attention_mask = None

    attention_input = {
        "query": inputs,
        "key": inputs,
        "value": inputs,
        "attention_mask": attention_mask,
        "p_dropout": p_dropout,
    }

    torch.manual_seed(42)
    pytorch_output = MultiHeadSelfAttention.scaled_dot_product_attention_pytorch(
        **attention_input
    )
    torch.manual_seed(42)
    from_scratch_output = (
        MultiHeadSelfAttention.scaled_dot_product_attention_from_scratch(
            **attention_input
        )
    )

    assert torch.allclose(
        pytorch_output, from_scratch_output, atol=1e-5
    ), "pytorch and from-scratch implementations produce different results"


@pytest.mark.parametrize("n_heads", [1, 4])
@pytest.mark.parametrize("with_random_attention_mask", [True, False])
def test_compare_flash_attn_and_from_scratch_implementations(
    n_heads: int,
    with_random_attention_mask: bool,
    p_dropout: float = 0.0,
):
    if not is_flash_attention_available():
        return

    batch_size = 2
    seq_len = 8
    d_model = 4

    inputs = torch.randn(
        batch_size, n_heads, seq_len, d_model, device="cuda", dtype=torch.float16
    )

    if with_random_attention_mask:
        attention_mask = torch.ones(
            batch_size, seq_len, device="cuda", dtype=torch.bool
        )
        for sample_index in range(batch_size):
            attention_mask[sample_index, : random.randrange(seq_len)] = False
    else:
        attention_mask = None

    torch.manual_seed(42)
    flash_attention_output = (
        MultiHeadSelfAttention.scaled_dot_product_attention_flash_attention(
            query=inputs,
            key=inputs,
            value=inputs,
            attention_mask=attention_mask,
            p_dropout=p_dropout,
        )
    )
    torch.manual_seed(42)
    from_scratch_output = (
        MultiHeadSelfAttention.scaled_dot_product_attention_from_scratch(
            query=inputs,
            key=inputs,
            value=inputs,
            attention_mask=attention_mask,
            p_dropout=p_dropout,
        )
    )

    torch.manual_seed(42)
    reference = MultiHeadSelfAttention.scaled_dot_product_attention_from_scratch(
        query=inputs.float(),
        key=inputs.float(),
        value=inputs.float(),
        attention_mask=attention_mask,
        p_dropout=p_dropout,
    )

    # From test_flash_attn.py in the flash_attn package
    # https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py#L702C6-L703C35
    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of the Pytorch implementation.

    print(torch.isclose(flash_attention_output, from_scratch_output))

    assert (flash_attention_output - reference).abs().max().item() <= 2 * (
        from_scratch_output - reference
    ).abs().max().item() + 1e-5
