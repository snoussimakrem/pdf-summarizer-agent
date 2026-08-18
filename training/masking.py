"""Assistant-response label masking for QLoRA SFT on Qwen2.5-7B-Instruct.

Verified 2026-08-18 (see project memory): Qwen2.5's chat template has no
`{% generation %}` block, so transformers' native
`return_assistant_tokens_mask=True` silently returns an all-zero mask. The
workaround verified to work for this exact model/template: tokenize the
prompt-only prefix separately (with `add_generation_prompt=True`) and use its
length to mask everything before the assistant's response. Confirmed this is
a true prefix of the full tokenization for this template — do not reuse this
approach for a different base model without re-verifying that.
"""
IGNORE_INDEX = -100


def build_labels(tokenizer, user_content: str, assistant_content: str):
    """Returns (input_ids, labels) for one training example: `labels` has
    IGNORE_INDEX everywhere except the assistant's response tokens."""
    prefix_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=True,
        add_generation_prompt=True,
    )["input_ids"]
    full_ids = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        tokenize=True,
        add_generation_prompt=False,
    )["input_ids"]

    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError(
            "prompt-only tokenization is not a true prefix of the full "
            "tokenization for this tokenizer/template — the manual masking "
            "workaround is unsafe here and needs re-verification."
        )

    labels = [IGNORE_INDEX] * len(prefix_ids) + list(full_ids[len(prefix_ids) :])
    return full_ids, labels
