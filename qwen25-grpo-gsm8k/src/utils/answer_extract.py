from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


FINAL_ANSWER_PATTERNS = [
    re.compile(r"(?im)^\s*answer\s*:\s*([-+]?\d[\d,\s]*(?:\.\d+)?)\.?\s*$"),
    re.compile(r"(?im)^\s*final\s+answer\s*:\s*([-+]?\d[\d,\s]*(?:\.\d+)?)\.?\s*$"),
    re.compile(r"\\boxed\{\s*([-+]?\d[\d,\s]*(?:\.\d+)?)\s*\}"),
]

FALLBACK_NUMBER_PATTERN = re.compile(r"([-+]?\d[\d,\s]*(?:\.\d+)?)")


def normalize_numeric_text(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = text.strip().replace(",", "").replace(" ", "")
    cleaned = cleaned[:-1] if cleaned.endswith(".") else cleaned
    if not cleaned:
        return None
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return cleaned
    normalized = format(value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def extract_ground_truth_from_gsm8k(answer_text: str) -> str:
    if "####" not in answer_text:
        raise ValueError("Could not find '####' marker in GSM8K answer.")
    raw = answer_text.split("####")[-1].strip()
    normalized = normalize_numeric_text(raw)
    if normalized is None:
        raise ValueError(f"Could not normalize GSM8K ground truth: {raw}")
    return normalized


def extract_answer_from_text(text: str) -> str | None:
    for pattern in FINAL_ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            return normalize_numeric_text(match.group(1))

    if "####" in text:
        tail = text.split("####")[-1].strip()
        normalized = normalize_numeric_text(tail)
        if normalized:
            return normalized

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        match = FALLBACK_NUMBER_PATTERN.fullmatch(line.rstrip("."))
        if match:
            return normalize_numeric_text(match.group(1))
    return None


def has_valid_answer_format(text: str) -> bool:
    return any(pattern.search(text) for pattern in FINAL_ANSWER_PATTERNS[:2]) or bool(
        FINAL_ANSWER_PATTERNS[2].search(text)
    )


def answers_equal(prediction: str | None, ground_truth: str | None) -> bool:
    pred = normalize_numeric_text(prediction)
    gt = normalize_numeric_text(ground_truth)
    return pred is not None and gt is not None and pred == gt
