from datasets import load_dataset  # type: ignore[import-untyped]
from transformers import AutoTokenizer  # type: ignore[import-untyped]

from src.constants import SEQUENCE_LENGTH


def create_dataset():
    """Create and preprocess the MBPP dataset."""

    dataset = load_dataset("mbpp", split="train")
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["code"],
            truncation=True,
            padding="max_length",
            max_length=SEQUENCE_LENGTH,
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="numpy", columns=["input_ids"])
    return tokenized_dataset, tokenizer
