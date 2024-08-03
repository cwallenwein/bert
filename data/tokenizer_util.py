from transformers import PreTrainedTokenizerFast


def tokenize(batch, tokenizer: PreTrainedTokenizerFast, context_length: int, return_tensors=None):
    result = tokenizer(
        batch,
        return_special_tokens_mask=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        max_length=context_length,
        padding="max_length",
        return_overflowing_tokens=True,
        truncation=True,
        return_tensors=return_tensors,
    )
    result.pop("overflow_to_sample_mapping")
    return result
