"""Dataset loader for the retool-dapo-math-scout-toolmandate parquet files.

The parquet schema produced by verl's retool prep is:
  data_source: str
  prompt: list[{role, content}]   # already hermes-formatted, single user msg
  ability: str
  reward_model: {ground_truth: str, style: str}
  extra_info: {...}
  agent_name: str

We project down to the two fields the AReaL workflow needs:
  messages: list[{role, content}]
  ground_truth: str
"""

from __future__ import annotations

from typing import Any, Optional

from datasets import load_dataset


def _coerce_messages(prompt_field: Any) -> list[dict[str, Any]]:
    # parquet -> datasets may yield numpy array of dicts; normalize to list.
    if hasattr(prompt_field, "tolist"):
        prompt_field = prompt_field.tolist()
    return [
        {"role": m["role"], "content": m["content"]} for m in prompt_field
    ]


def get_retool_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: Optional[int] = None,
):
    dataset = load_dataset("parquet", data_files=path, split="train")

    def process(sample):
        return {
            "messages": _coerce_messages(sample["prompt"]),
            "ground_truth": sample["reward_model"]["ground_truth"],
        }

    keep = {"messages", "ground_truth"}
    drop = [c for c in dataset.column_names if c not in keep]
    dataset = dataset.map(process).remove_columns(drop)

    if max_length is not None:

        def filter_length(sample):
            ids = tokenizer.apply_chat_template(
                sample["messages"],
                add_generation_prompt=True,
                tokenize=True,
            )
            return len(ids) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
