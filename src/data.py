"""
Load a Liars' Bench subset and turn it into (text, label) pairs for probing.

Liars' Bench (Kretschmar, Laurito, Maiya & Marks, 2025 - arXiv:2511.16035)
packages 72,863 examples of model-generated lies and honest responses across
seven settings. We default to the "Instructed-Deception" subset: a model is
asked true/false factual questions and either told to answer honestly or
told to lie, giving clean, balanced, easy-to-verify labels.
"""

from __future__ import annotations

from dataclasses import dataclass

from datasets import Dataset, get_dataset_config_names, load_dataset

_TEXT_FIELD_CANDIDATES = ["response", "completion", "text", "assistant_response", "output", "messages"]
_LABEL_FIELD_CANDIDATES = ["is_lie", "label", "lie", "deceptive", "is_deceptive"]


@dataclass
class ProbingExample:
    text: str
    label: int  # 1 = lie / deceptive, 0 = honest


def _resolve_field(columns: list[str], candidates: list[str], role: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(
        f"Could not find a {role} column automatically.\n"
        f"Available columns: {columns}\n"
        f"Add the correct name to the appropriate candidates list in src/data.py."
    )


def _message_content_to_str(content) -> str:
    """`content` is usually a plain string, but some chat schemas nest it as
    a list of {"type": "text", "text": ...} blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", str(block)))
            else:
                parts.append(str(block))
        return " ".join(parts)
    return "" if content is None else str(content)


def _render_messages(messages: list[dict] | None) -> str | None:
    """Render a chat-format message list into one text block, e.g.:
    'System: ...\\nUser: ...\\nAssistant: ...'
    Returns None for empty input."""
    if not messages:
        return None
    lines = [
        f"{m.get('role', 'unknown').capitalize()}: {_message_content_to_str(m.get('content'))}"
        for m in messages
    ]
    return "\n".join(lines)


def _extract_text(row: dict, text_field: str) -> str | None:
    if text_field == "messages":
        return _render_messages(row[text_field])
    return row[text_field]


def list_available_subsets(hf_path: str) -> list[str]:
    try:
        return get_dataset_config_names(hf_path)
    except ValueError as e:
        raise ValueError(f"Could not list subsets for {hf_path}.") from e

def load_liars_bench(
    hf_path: str,
    subset: str,
    split: str,
    max_examples_per_class: int | None,
    seed: int,
) -> list[ProbingExample]:
    try:
        raw: Dataset = load_dataset(hf_path, subset, split=split)
    except ValueError as e:
        available = list_available_subsets(hf_path)
        raise ValueError(
            f"Subset '{subset}' not found for {hf_path}. "
            f"Available subsets: {available}"
        ) from e

    columns = raw.column_names
    text_field = _resolve_field(columns, _TEXT_FIELD_CANDIDATES, "text")
    label_field = _resolve_field(columns, _LABEL_FIELD_CANDIDATES, "label")

    examples = []
    for row in raw:
        text = _extract_text(row, text_field)
        if text is not None:
            examples.append(ProbingExample(text=text, label=int(bool(row[label_field]))))

    lies = [ex for ex in examples if ex.label == 1]
    honest = [ex for ex in examples if ex.label == 0]

    rng_shuffle = _deterministic_shuffle(seed)
    rng_shuffle(lies)
    rng_shuffle(honest)

    if max_examples_per_class is not None:
        lies = lies[:max_examples_per_class]
        honest = honest[:max_examples_per_class]

    if not lies or not honest:
        raise ValueError(
            f"After filtering, got {len(lies)} lie examples and {len(honest)} honest "
            f"examples -- probing needs both classes. Check the label field mapping."
        )

    balanced = lies + honest
    rng_shuffle(balanced)
    return balanced


def _deterministic_shuffle(seed: int):
    import random

    rng = random.Random(seed)

    def shuffle(lst: list) -> None:
        rng.shuffle(lst)

    return shuffle
