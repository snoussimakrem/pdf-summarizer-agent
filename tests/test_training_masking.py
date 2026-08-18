import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
from masking import IGNORE_INDEX, build_labels  # noqa: E402


class FakeTokenizer:
    """Simulates a broken chat template where the prompt-only prefix
    diverges from the full tokenization — the case build_labels must reject."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        if len(messages) == 1:
            return {"input_ids": [1, 2, 3]}
        return {"input_ids": [1, 2, 99, 4, 5]}  # not a prefix match at index 2


def test_rejects_non_prefix_template() -> None:
    with pytest.raises(ValueError, match="not a true prefix"):
        build_labels(FakeTokenizer(), "prompt", "response")


class ConsistentFakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        if len(messages) == 1:
            return {"input_ids": [1, 2, 3]}
        return {"input_ids": [1, 2, 3, 4, 5]}


def test_masks_everything_before_assistant_response() -> None:
    input_ids, labels = build_labels(ConsistentFakeTokenizer(), "prompt", "response")
    assert input_ids == [1, 2, 3, 4, 5]
    assert labels == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 4, 5]
