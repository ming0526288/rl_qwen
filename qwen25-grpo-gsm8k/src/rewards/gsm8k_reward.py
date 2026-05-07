from __future__ import annotations

from dataclasses import dataclass, asdict

from src.utils.answer_extract import answers_equal, extract_answer_from_text, has_valid_answer_format


@dataclass
class RewardBreakdown:
    reward: float
    answer_reward: float
    format_reward: float
    length_penalty: float
    correct: bool
    format_ok: bool
    extracted_answer: str | None
    ground_truth: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compute_length_penalty(
    completion_length: int,
    overlong_512: float = -0.1,
    overlong_768: float = -0.2,
) -> float:
    if completion_length > 768:
        return overlong_768
    if completion_length > 512:
        return overlong_512
    return 0.0


def compute_gsm8k_reward(
    completion_text: str,
    ground_truth: str,
    completion_length: int,
    answer_correct: float = 1.0,
    format_correct: float = 0.2,
    overlong_512: float = -0.1,
    overlong_768: float = -0.2,
) -> RewardBreakdown:
    extracted_answer = extract_answer_from_text(completion_text)
    correct = answers_equal(extracted_answer, ground_truth)
    format_ok = has_valid_answer_format(completion_text)

    answer_reward = answer_correct if correct else 0.0
    format_reward = format_correct if format_ok else 0.0
    length_penalty = compute_length_penalty(
        completion_length=completion_length,
        overlong_512=overlong_512,
        overlong_768=overlong_768,
    )
    reward = answer_reward + format_reward + length_penalty

    return RewardBreakdown(
        reward=reward,
        answer_reward=answer_reward,
        format_reward=format_reward,
        length_penalty=length_penalty,
        correct=correct,
        format_ok=format_ok,
        extracted_answer=extracted_answer,
        ground_truth=ground_truth,
    )
